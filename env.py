import numpy as np
import time
import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

import gym
from gym import spaces
from gym.utils import seeding

# Agents
from .agents.humanoid_supine import Humanoid
from .agents.humanoid_sitting import HumanoidSeated
from .agents.pybullet_ur5.robot import UR5Robotiq85

# Utils
from .utils.util import Util
from .utils.collision_utils import get_collision_fn


class AssistiveEnv(gym.Env):
    def __init__(self, frame_skip=5, time_step=0.02, action_robot_len=6, obs_robot_len=90):
        # Start the bullet physics server
        self.bc = BulletClient(connection_mode=p.DIRECT)
        self.gui = False
        self.restore_state = False

        # Execute actions at 10 Hz by default. A new action every 0.1 seconds
        self.frame_skip = frame_skip
        self.time_step = time_step
        self.setup_timing()
        self.seed(1001)

        self.action_robot_len = action_robot_len
        self.obs_robot_len = obs_robot_len
        self.action_space = spaces.Box(low=np.array([-1.0]*(self.action_robot_len)), high=np.array([1.0]*(self.action_robot_len)), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-30.0]*(self.obs_robot_len)), high=np.array([30.0]*(self.obs_robot_len)), dtype=np.float32)
        self.util = Util(self.bc, self.np_random)

    def create_world(self):
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bc.setGravity(0, 0, 0, physicsClientId=self.bc._client) 

        # Load environment
        plane_id = self.bc.loadURDF("plane.urdf", (0, 0, -0.04), physicsClientId=self.bc._client)
        bed_id = self.bc.loadURDF("learn_bathing/envs/urdf/bed_0.urdf", (0.0, -0.1, 0.0), useFixedBase=True, physicsClientId=self.bc._client)

        # Load human
        human_base_pos = (0, 0, 0.3)
        human_base_orn = self.bc.getQuaternionFromEuler((0, 1.57, 0))
        self.humanoid = Humanoid(self.bc, human_base_pos, human_base_orn)

        # Load robot
        self.robot_base_pose = ((0.65, 0, 0.25), (0, 0, -1.57))
        cube_id = p.loadURDF("learn_bathing/envs/urdf/cube_0.urdf", 
                            (self.robot_base_pose[0][0], self.robot_base_pose[0][1], self.robot_base_pose[0][2]-0.15), useFixedBase=True,
                            physicsClientId=self.bc._client)
        self.robot = UR5Robotiq85(self.bc, self.robot_base_pose[0], self.robot_base_pose[1])
        self.robot.load()
        self.robot.reset()

        self.bed_id = bed_id
        self.cube_id = cube_id

        # initialize collision checker
        obstacles = [self.bed_id, self.cube_id, self.humanoid._humanoid]
        self.robot_in_collision = get_collision_fn(self.robot.id, self.robot.arm_controllable_joints, obstacles=obstacles,
                                                     attachments=[], self_collisions=True,
                                                     disabled_collisions=set(), client_id=self.bc._client)
        
        # compute target_to_eef
        world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        target_orn = self.util.rotate_quaternion_by_axis(world_to_eef[1], axis='z', degrees=180)
        world_to_target = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.3], target_orn]
        world_to_target_closer = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.2], target_orn]   #####

        self.target_orn = target_orn
        target_to_world = self.bc.invertTransform(world_to_target[0], world_to_target[1], physicsClientId=self.bc._client)
        target_closer_to_world = self.bc.invertTransform(world_to_target_closer[0], world_to_target_closer[1], physicsClientId=self.bc._client)   #####
        self.target_to_eef = self.bc.multiplyTransforms(target_to_world[0], target_to_world[1],
                                                        world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)
        self.target_closer_to_eef = self.bc.multiplyTransforms(target_closer_to_world[0], target_closer_to_world[1],
                                                               world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)   #####

    def generate_random_q_H(self):
        q_H_1 = np.random.uniform(1.5, 3.0)
        q_H_2 = np.random.uniform(0.0, 1.0)
        q_H_3 = np.random.uniform(-2.6, -2.0)
        q_H_4 = np.random.uniform(0.40, 1.5)
        return [q_H_1, q_H_2, q_H_3, q_H_4]
    
    def human_in_collision(self):
        """Check if any part of the human arm collides with other objects."""
        contact_points = self.bc.getContactPoints(bodyA=self.humanoid._humanoid, physicsClientId=self.bc._client)
        for point in contact_points:
            if (point[2] in [self.bed_id, self.cube_id, self.robot.id]):
                return True
        return False

    def reset_and_check(self):
        """Reset the human arm and check for collisions until no collision is detected."""
        while True:
            q_H = self.generate_random_q_H()
            self.reset_human_arm(q_H)
            self.bc.stepSimulation(physicsClientId=self.bc._client)
            if not self.human_in_collision():
                self.lock_human_joints(q_H)
                #print(f'q_H: {q_H}')
                break
    
    def reset_human_arm(self, q_human):
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, j, q_human[i], physicsClientId=self.bc._client)

    def init_tool(self):
        # close gripper
        for _ in range(50):
            self.robot.move_gripper(0.04)
            self.bc.stepSimulation(physicsClientId=self.bc._client)

        # initialize tool for wiping task
        world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        eef_to_world = self.bc.invertTransform(world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)
        world_to_tool = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.14], [0,0,0,1]]
        eef_to_tool = self.bc.multiplyTransforms(positionA=eef_to_world[0], orientationA=eef_to_world[1], 
                                                 positionB=world_to_tool[0], orientationB=world_to_tool[1], physicsClientId=self.bc._client)
        self.eef_to_tool = eef_to_tool
        self.tool = self.bc.loadURDF("learn_bathing/envs/urdf/wiper.urdf", basePosition=world_to_tool[0], baseOrientation=world_to_tool[1], physicsClientId=self.bc._client)

        # disable collisions between the tool and robot
        for j in self.robot.arm_controllable_joints:
            for tj in list(range(self.bc.getNumJoints(self.tool, physicsClientId=self.bc._client))) + [-1]:
                self.bc.setCollisionFilterPair(self.robot.id, self.tool, j, tj, False, physicsClientId=self.bc._client)
        
    def attach_tool(self):
        # create constraint that keeps the tool in the gripper
        self.cid = self.bc.createConstraint(parentBodyUniqueId=self.robot.id,
                            parentLinkIndex=self.robot.eef_id,
                            childBodyUniqueId=self.tool,
                            childLinkIndex=-1,
                            jointType=p.JOINT_FIXED,
                            jointAxis=(0, 0, 0),
                            parentFramePosition=self.eef_to_tool[0],
                            parentFrameOrientation=self.eef_to_tool[1],
                            childFramePosition=(0, 0, 0),
                            childFrameOrientation=(0, 0, 0),
                            physicsClientId=self.bc._client)

    def slow_time(self):
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)
        self.last_sim_time = time.time()

    def setup_timing(self):
        self.total_time = 0
        self.last_sim_time = None
        self.iteration = 0

    def lock_human_joints(self, q_H):
        # Make all joints on the person static by setting mass of each link (joint) to 0
        for j in range(self.bc.getNumJoints(self.humanoid._humanoid, physicsClientId=self.bc._client)):
            self.bc.changeDynamics(self.humanoid._humanoid, j, mass=0, physicsClientId=self.bc._client)
        # Set arm joints velocities to 0
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, jointIndex=j, targetValue=q_H[i], targetVelocity=0, physicsClientId=self.bc._client)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    # ### TODO
    # def store_current_state(self):

    # def restore_state(self):


    def take_step(self, action, gains=0.05, forces=1):
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)

        self.iteration += 1
        if self.last_sim_time is None:
            self.last_sim_time = time.time()

        action *= 0.05
        action_robot = action
        
        robot_joint_states = self.bc.getJointStates(self.robot.id, jointIndices=self.robot.arm_controllable_joints, 
                                                    physicsClientId=self.bc._client)
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])
        
        # Joints after applying action
        for _ in range(self.frame_skip):
            action_robot[robot_joint_positions + action_robot < self.robot_lower_limits] = 0
            action_robot[robot_joint_positions + action_robot > self.robot_upper_limits] = 0
            robot_joint_positions += action_robot

        # Setting Joints
        self.bc.setJointMotorControlArray(self.robot.id, jointIndices=self.robot.arm_controllable_joints, 
                                          controlMode=p.POSITION_CONTROL, targetPositions=robot_joint_positions,
                                          positionGains=np.array([gains]*self.action_robot_len), forces=[forces]*self.action_robot_len, 
                                          physicsClientId=self.bc._client)

        # Update robot position
        for _ in range(self.frame_skip):
            self.bc.stepSimulation(physicsClientId=self.bc._client)
            # self.update_targets()
            if self.gui:
                self.slow_time()

    def render(self, mode='human'):
        if not self.gui:
            self.gui = True
            self.bc.disconnect()
            self.bc = BulletClient(connection_mode=p.GUI)
            self.setup_timing()
            self.create_world()
            self.util = Util(self.bc._client, self.np_random)
