import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
import time
from ray.rllib.agents import ppo
# from ray.rllib.algorithms import ppo
import learn_bathing
import matplotlib.pyplot as plt
import pickle

def setup_config(env, algo, render, coop=False, seed=0, extra_configs={}):
    # num_processes = multiprocessing.cpu_count()
    if render:
        num_processes = 1
    else:
        num_processes = max(1, int(multiprocessing.cpu_count() / 2))  # Use half of your cores

    print(f'num_processes: {num_processes}')
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 19200
        config['num_sgd_iter'] = 50
        config['sgd_minibatch_size'] = 128
        config['lambda'] = 0.95
        config['model']['fcnet_hiddens'] = [100, 100]
        # config = (
        #     ppo.PPOConfig()
        #     .training(
        #         train_batch_size=19200,
        #         num_sgd_iter=50,
        #         sgd_minibatch_size=128,
        #         lambda_=0.95,
        #         model={"fcnet_hiddens": [100, 100]},
        #     )
        # )
    else:
        raise ValueError('invalid algorithm')
    
    #config['num_workers'] = num_processes
    #config['num_cpus_per_worker'] = 0
    config['seed'] = seed
    config['log_level'] = 'ERROR'

    config["framework"] = "torch"
    config['num_workers'] = 0
    config['num_cpus_per_worker'] = 0
    config['num_gpus'] = 0
    config["eager_tracing"] = False  # avoids TF traces confusion
    config["simple_optimizer"] = True
    # config = (
    #     config.rollouts(num_rollout_workers=num_processes)
    #     .resources(num_gpus=0, num_cpus_per_worker=0)
    #     .framework("tf")
    #     .evaluation(evaluation_config={"seed": seed})
    #     .debugging(log_level="ERROR")
    # )

    return {**config, **extra_configs}

import pickle

def _restore_policy_weights_only(agent, checkpoint_path: str):
    """
    Load policy weights from RLlib checkpoint WITHOUT restoring optimizer state.
    Supports multiple RLlib 1.x checkpoint layouts.
    """
    import pickle

    with open(checkpoint_path, "rb") as f:
        outer = pickle.load(f)

    worker_blob = outer.get("worker", None)
    if worker_blob is None:
        raise ValueError("Checkpoint missing 'worker' key")

    worker = pickle.loads(worker_blob) if isinstance(worker_blob, (bytes, bytearray)) else worker_blob
    state = worker.get("state", worker)

    # Layout A: state["policy_states"][pid]["weights"]
    if isinstance(state, dict) and "policy_states" in state:
        policy_states = state["policy_states"]
        pid = "default_policy" if "default_policy" in policy_states else next(iter(policy_states.keys()))
        pstate = policy_states[pid]

    # Layout B (your case): state[pid]["weights"]
    elif isinstance(state, dict) and "default_policy" in state:
        pid = "default_policy"
        pstate = state[pid]

    # Layout C: state is dict of policies but not named default_policy
    elif isinstance(state, dict) and len(state) > 0:
        pid = next(iter(state.keys()))
        pstate = state[pid]

    else:
        raise ValueError(
            f"Unrecognized checkpoint layout. worker keys: {list(worker.keys())}, "
            f"state type: {type(state)}, state keys: {list(state.keys()) if isinstance(state, dict) else None}"
        )

    weights = pstate.get("weights", None)
    if weights is None:
        raise ValueError(f"Policy state missing 'weights'. pid={pid}, keys: {list(pstate.keys())}")

    agent.get_policy(pid).set_weights(weights)
    return pid
def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, extra_configs={}, render=False):
    if algo == 'ppo':
        agent = ppo.PPOTrainer(setup_config(env, algo, render, coop, seed, extra_configs), env_name)
    else:
        raise ValueError('invalid algorithm')

    if policy_path != '':
        # 1) If a checkpoint file path is provided directly
        if policy_path is not None and 'checkpoint' in policy_path:
            # Weights-only restore first: full agent.restore() crashes on
            # mac/RLlib 1.x when converting optimizer state
            # (TypeError: numpy.object_). Same workaround evaluate_policy uses.
            # Cost: Adam optimizer moments reset on resume (negligible).
            try:
                _restore_policy_weights_only(agent, policy_path)
                print(f'[load_policy] weights-only restore OK: {policy_path}')
            except Exception as e:
                print(f'[load_policy] weights-only restore failed ({e}); trying full restore')
                agent.restore(policy_path)
            return agent, policy_path

        # 2) Otherwise, find most recent checkpoint in directory
        directory = os.path.join(policy_path, algo, env_name)
        files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
        files_ints = [int(f) for f in files]

        if files:
            checkpoint_max = max(files_ints)
            checkpoint_num = files_ints.index(checkpoint_max)
            checkpoint_path = os.path.join(
                directory,
                'checkpoint_%s' % files[checkpoint_num],
                'checkpoint-%d' % checkpoint_max
            )
            print(f'checkpoint_path: {checkpoint_path}')

            if render:
                _restore_policy_weights_only(agent, checkpoint_path)
            else:
                agent.restore(checkpoint_path)

            return agent, checkpoint_path

        return agent, None

    return agent, None
    
def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make(env_name)
    else:
        raise ValueError('coop is not supported')
    
    env.seed(seed)
    return env

def train(env_name, algo, timesteps_total=1000000, save_dir='./trained_models/', load_policy_path='', coop=False, seed=0, extra_configs={}):
    render = False
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False, _node_ip_address="127.0.0.1")
    env = make_env(env_name)
    agent, checkpoint_path = load_policy(env, algo, env_name, load_policy_path, coop, seed, extra_configs, render)
    env.close()

    timesteps = 0
    start_time = time.time()

    while timesteps < timesteps_total:
        result = agent.train()
        timesteps = result['timesteps_total']

        print(f"Iteration: {result['training_iteration']}, total timesteps: {result['timesteps_total']}, total time: {result['time_total_s']:.1f}, FPS: {result['timesteps_total']/result['time_total_s']:.1f}, mean reward: {result['episode_reward_mean']:.1f}, min/max reward: {result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f}")
        sys.stdout.flush()

        # Delete the old saved policy
        if checkpoint_path is not None:
            shutil.rmtree(os.path.dirname(checkpoint_path), ignore_errors=True)
        # Save the recently trained policy
        checkpoint_path = agent.save(os.path.join(save_dir, algo, env_name))

    training_time = time.time() - start_time
    print(f"training time: {training_time} s")

    return checkpoint_path

def _follow_end_effector(env):

    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    tool_pos = base_env.bc.getLinkState(
        base_env.tool, 1,
        computeForwardKinematics=True,
        physicsClientId=base_env.bc._client
    )[0]

    # SIDE VIEW — not behind robot
    p.resetDebugVisualizerCamera(
        cameraDistance=2.4,   # less zoom
        cameraYaw=90,         # pure side view
        cameraPitch=-35,      # more top-down
        cameraTargetPosition=tool_pos
    )

def render_policyx(env, env_name, algo, policy_path, coop=False, colab=False, seed=0, n_episodes=1, extra_configs={}):
    render = True
    ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False, _node_ip_address="127.0.0.1")
    if env is None:
        env = make_env(env_name, seed=seed)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs, render)

    if not colab:
        env.render()

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            if coop:
                raise ValueError('coop not implemented')
            else:
                # Compute the next action using the trained policy
                action = test_agent.compute_action(obs)
                # Step the simulation forward using the action from our trained policy
                obs, reward, done, info = env.step(action)

                number_of_targets_cleared = info['number_of_targets_cleared']
                task_percentage = info['task_percentage']
                total_target_count = info['total_target_count']
                feasible_target_count = info['feasible_target_count']
        
            print(f'number_of_targets_cleared: {number_of_targets_cleared}/{feasible_target_count}, task_percentage: {task_percentage}, reward: {reward}')
    
    env.close()

def render_policy(env, env_name, algo, policy_path, coop=False, colab=False, seed=0, n_episodes=1, extra_configs={}):
    import pybullet as p
    import numpy as np

    render = True
    ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False, _node_ip_address="127.0.0.1")

    if env is None:
        env = make_env(env_name, seed=seed)

    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs, render)

    if not colab:
        env.render()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # remove side panels
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)

    # Camera helper
    def _follow_end_effector(env):
    
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    
        tool_pos = base_env.bc.getLinkState(
            base_env.tool, 1,
            computeForwardKinematics=True,
            physicsClientId=base_env.bc._client
        )[0]
    
        # SIDE VIEW — not behind robot
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,   # less zoom
            cameraYaw=150,         # pure side view
            cameraPitch=-33,      # more top-down
        cameraTargetPosition=tool_pos)


    for episode in range(n_episodes):
        obs = env.reset()
        _follow_end_effector(env)

        done = False
        while not done:
            if coop:
                raise ValueError('coop not implemented')
            else:
                action = test_agent.compute_action(obs)
                obs, reward, done, info = env.step(action)

                _follow_end_effector(env)

                number_of_targets_cleared = info['number_of_targets_cleared']
                task_percentage = info['task_percentage']
                feasible_target_count = info['feasible_target_count']

            print(f'number_of_targets_cleared: {number_of_targets_cleared}/{feasible_target_count}, task_percentage: {task_percentage:.2f}, reward: {reward:.3f}')
            print(f"d_arm={info['d_arm']:.3f} force={info['normal_force']:.2f} near={info['near_target_contact']} clears={info['clears_this_step']} rem={info['n_remaining_targets']} reward={reward:.3f}")

    env.close()


# def evaluate_policy(env_name, algo, policy_path, n_episodes=100, coop=False, seed=0, verbose=False, extra_configs={}):
#     render = False
#     ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
#     env = make_env(env_name, coop, seed=seed)
#     test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs, render)

#     rewards = []
#     task_successes = []
#     number_of_targets_cleared_list = []
#     for episode in range(n_episodes):
#         obs = env.reset()
#         done = False
#         reward_total = 0.0
#         task_success = 0.0
#         while not done:
#             if coop:
#                 raise ValueError('coop not implemented')
#             else:
#                 action = test_agent.compute_action(obs)
#                 obs, reward, done, info = env.step(action)
#             reward_total += reward
#             task_success = info['task_success']
#             number_of_targets_cleared = info['number_of_targets_cleared']

#         rewards.append(reward_total)
#         task_successes.append(task_success)
#         number_of_targets_cleared_list.append(number_of_targets_cleared)
#         if verbose:
#             print('Reward total: %.2f, task success: %r, number of targets cleared %r' % (reward_total, task_success, number_of_targets_cleared))
#         sys.stdout.flush()
#     env.close()

#     print('\n', '-'*50, '\n')
#     # print('Rewards:', rewards)
#     print('Reward Mean:', np.mean(rewards))
#     print('Reward Std:', np.std(rewards))

#     # print('Task Successes:', task_successes)
#     print('Task Success Mean:', np.mean(task_successes))
#     print('Task Success Std:', np.std(task_successes))

#     print('Number of Targets Cleared Mean:', np.mean(number_of_targets_cleared_list))
#     print('Number of Targets Cleared Std:', np.std(number_of_targets_cleared_list))
#     sys.stdout.flush()

def evaluate_policy(env_name, algo, policy_path, n_episodes=100, coop=False, seed=0, verbose=False, extra_configs={}):
    # We only need policy weights for evaluation (optimizer restore can crash on mac/RLlib 1.x),
    # so we force the "weights-only restore" path by setting render=True.
    render = True

    # Evaluation should be single-process and deterministic-ish
    extra_configs = {**extra_configs, "num_workers": 0, "num_gpus": 0}

    ray.init(num_cpus=multiprocessing.cpu_count(),
             ignore_reinit_error=True,
             log_to_driver=False,
             _node_ip_address="127.0.0.1")

    env = make_env(env_name, coop, seed=seed)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs, render)

    rewards = []
    task_successes = []
    number_of_targets_cleared_list = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        reward_total = 0.0
        task_success = 0.0
        number_of_targets_cleared = 0

        while not done:
            if coop:
                raise ValueError('coop not implemented')
            else:
                action = test_agent.compute_action(obs)
                obs, reward, done, info = env.step(action)

            reward_total += reward
            task_success = info.get('task_success', task_success)
            number_of_targets_cleared = info.get('number_of_targets_cleared', number_of_targets_cleared)

        rewards.append(reward_total)
        task_successes.append(task_success)
        number_of_targets_cleared_list.append(number_of_targets_cleared)

        if verbose:
            print('Reward total: %.2f, task success: %r, number of targets cleared %r'
                  % (reward_total, task_success, number_of_targets_cleared))
        sys.stdout.flush()

    env.close()

    # Summaries
    rewards_mean = float(np.mean(rewards)) if len(rewards) else 0.0
    rewards_std = float(np.std(rewards)) if len(rewards) else 0.0
    success_mean = float(np.mean(task_successes)) if len(task_successes) else 0.0
    success_std = float(np.std(task_successes)) if len(task_successes) else 0.0
    cleared_mean = float(np.mean(number_of_targets_cleared_list)) if len(number_of_targets_cleared_list) else 0.0
    cleared_std = float(np.std(number_of_targets_cleared_list)) if len(number_of_targets_cleared_list) else 0.0

    print('Evaluation over %d episodes:' % n_episodes)
    print('  reward mean/std: %.3f / %.3f' % (rewards_mean, rewards_std))
    print('  task_success mean/std: %.3f / %.3f' % (success_mean, success_std))
    print('  targets_cleared mean/std: %.3f / %.3f' % (cleared_mean, cleared_std))

    return {
        "reward_mean": rewards_mean,
        "reward_std": rewards_std,
        "task_success_mean": success_mean,
        "task_success_std": success_std,
        "targets_cleared_mean": cleared_mean,
        "targets_cleared_std": cleared_std,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='learn_bathing:WipingEnv-v0',
                        help='Environment to train on (default: WipingEnv-v0)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--entropy-coeff', type=float, default=None,
                        help='PPO entropy_coeff override (e.g. 0.01 to encourage '
                             'exploration when plateaued). If omitted, uses RLlib '
                             'default (0.0).')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train a new policy')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    parser.add_argument('--train-timesteps', type=int, default=1000000,
                        help='Number of simulation timesteps to train a policy (default: 1000000)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--render-episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='Whether rendering should generate an animated png rather than open a window (e.g. when using Google Colab)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    args = parser.parse_args()
    
    coop = False
    checkpoint_path = None

    extra_configs = {}
    if args.entropy_coeff is not None:
        extra_configs['entropy_coeff'] = args.entropy_coeff
        print(f'[learn] entropy_coeff override = {args.entropy_coeff}')

    if args.train:
        checkpoint_path = train(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=args.save_dir, load_policy_path=args.load_policy_path, coop=coop, seed=args.seed, extra_configs=extra_configs)
    if args.render:
        render_policy(None, args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, coop=coop, colab=args.colab, seed=args.seed, n_episodes=args.render_episodes)
    if args.evaluate:
        evaluate_policy(args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, n_episodes=args.eval_episodes, coop=coop, seed=args.seed, verbose=args.verbose)
