Place the two checkpoint filed under /trained_models/ppo/learn_bathing:WipingEnv-v0/checkpoint_000058/
python learn.py --train --algo ppo --load-policy-path ./trained_models/ppo/learn_bathing:WipingEnv-v0/checkpoint_000058/checkpoint-58

python learn.py --render --algo ppo --load-policy-path ./trained_models/ppo/learn_bathing:WipingEnv-v0/checkpoint_000058/checkpoint-58
