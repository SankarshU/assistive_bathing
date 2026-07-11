# MIT License

# Copyright (c) 2019 Healthcare Robotics Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Modified by: Yubin Koh (koh22@purdue.edu)

import numpy as np
import pybullet as p
import random
import time

class TargetsUtil:
    def __init__(self, pid, util):
        self.pid = pid
        self.util = util

    def init_targets_util(self, humanoid_id, shoulder, elbow, human_arm,
                          robot_w, tool,
                          target_closer_to_eef, robot_w_in_collision):
        self.humanoid_id = humanoid_id
        self.shoulder = shoulder
        self.elbow = elbow
        self.human_arm = human_arm

        self.robot_w = robot_w
        self.tool = tool

        self.target_closer_to_eef = target_closer_to_eef
        self.robot_w_in_collision = robot_w_in_collision

    def generate_new_targets_pose(self):
        self.target_indices_to_ignore = []
        self.upperarm_length, self.upperarm_radius = 0.144000+0.036000, 0.036000
        self.forearm_length, self.forearm_radius = 0.108000+0.028000*2, 0.028000
        self.distance_between_points = 0.03

        # generate capsule points
        self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), 
                                                                radius=self.upperarm_radius, distance_between_points=self.distance_between_points)
        self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, -0.01]), p2=np.array([0, 0, -self.forearm_length-0.01]), 
                                                                radius=self.forearm_radius, distance_between_points=self.distance_between_points)
    
        # generate upperarm targets orientation
        degree = np.rad2deg(self.distance_between_points/self.upperarm_radius)
        col_offset = int(self.upperarm_length//self.distance_between_points)  # 6
        row_offset = int(len(self.targets_pos_on_upperarm)/col_offset)
        self.targets_orn_on_upperarm = []
        for _ in range(col_offset):
            for i in range(row_offset):
                self.targets_orn_on_upperarm.append(self.util.rotate_quaternion_by_axis([0,0,0,1], axis='z', degrees=-degree*i-90))

        # generate forearm targets orientation
        degree = np.rad2deg(self.distance_between_points/self.forearm_radius)
        col_offset = int(self.forearm_length//self.distance_between_points)  # 5
        row_offset = int(len(self.targets_pos_on_forearm)/col_offset)
        self.targets_orn_on_forearm = []
        for _ in range(col_offset):
            for i in range(row_offset):
                self.targets_orn_on_forearm.append(self.util.rotate_quaternion_by_axis([0,0,0,1], axis='z', degrees=-degree*i-90))

        assert len(self.targets_pos_on_upperarm) == len(self.targets_orn_on_upperarm)
        assert len(self.targets_pos_on_forearm) == len(self.targets_orn_on_forearm)

    def initialize_deleted_targets_list(self):
        # to keep track of deleted targets
        self.deleted_targets_indices_on_upperarm = []
        self.deleted_targets_indices_on_forearm = []

    def generate_targets(self):
        assert self.targets_pos_on_upperarm is not None
        assert self.targets_pos_on_forearm is not None

        # create target points
        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.pid)
        self.targets_upperarm = []
        self.targets_forearm = []
        for _ in range(len(self.targets_pos_on_upperarm)):
            self.targets_upperarm.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, 
                                                           basePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.pid))
        for _ in range(len(self.targets_pos_on_forearm)):
            self.targets_forearm.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, 
                                                          basePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.pid))
        self.total_target_count = len(self.targets_upperarm) + len(self.targets_forearm)

        # move targets to initial positions
        self.update_targets()
    
    def remove_targets(self):
        # remove upperarm targets
        remove_targets_upperarm = []
        for i in self.deleted_targets_indices_on_upperarm:
            remove_targets_upperarm.append(self.targets_upperarm[i])
        self.targets_pos_on_upperarm = [t for i, t in enumerate(self.targets_pos_on_upperarm) if i not in self.deleted_targets_indices_on_upperarm]
        self.targets_orn_on_upperarm = [t for i, t in enumerate(self.targets_orn_on_upperarm) if i not in self.deleted_targets_indices_on_upperarm]
        self.targets_pos_upperarm_world = [t for i, t in enumerate(self.targets_pos_upperarm_world) if i not in self.deleted_targets_indices_on_upperarm]
        self.targets_orn_upperarm_world = [t for i, t in enumerate(self.targets_orn_upperarm_world) if i not in self.deleted_targets_indices_on_upperarm]
        self.targets_upperarm = [t for i, t in enumerate(self.targets_upperarm) if i not in self.deleted_targets_indices_on_upperarm]

        assert len(self.targets_pos_on_upperarm) == len(self.targets_pos_upperarm_world) == len(self.targets_orn_upperarm_world) == len(self.targets_upperarm)

        # remove forearm targets
        remove_targets_forearm = []
        for i in self.deleted_targets_indices_on_forearm:
            remove_targets_forearm.append(self.targets_forearm[i])
        self.targets_pos_on_forearm = [t for i, t in enumerate(self.targets_pos_on_forearm) if i not in self.deleted_targets_indices_on_forearm]
        self.targets_orn_on_forearm = [t for i, t in enumerate(self.targets_orn_on_forearm) if i not in self.deleted_targets_indices_on_forearm]
        self.targets_pos_forearm_world = [t for i, t in enumerate(self.targets_pos_forearm_world) if i not in self.deleted_targets_indices_on_forearm]
        self.targets_orn_forearm_world = [t for i, t in enumerate(self.targets_orn_forearm_world) if i not in self.deleted_targets_indices_on_forearm]
        self.targets_forearm = [t for i, t in enumerate(self.targets_forearm) if i not in self.deleted_targets_indices_on_forearm]

        assert len(self.targets_pos_on_forearm) == len(self.targets_pos_forearm_world) == len(self.targets_orn_forearm_world) == len(self.targets_forearm)

        self.initialize_deleted_targets_list()  # reset to empty list

    def remove_all_targets(self):
        for target in self.targets_upperarm:
            p.resetBasePositionAndOrientation(target, [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.pid)
        for target in self.targets_forearm:
            p.resetBasePositionAndOrientation(target, [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.pid)

        self.targets_pos_on_upperarm = []
        self.targets_orn_on_upperarm = []
        self.targets_upperarm = []

        self.targets_pos_on_forearm = []
        self.targets_orn_on_forearm = []
        self.targets_forearm = []

    def update_targets(self):
        # upperarm targets position & orientation
        upperarm_pos, upperarm_orient = p.getLinkState(self.humanoid_id, self.shoulder, computeForwardKinematics=True, physicsClientId=self.pid)[4:6]
        upperarm_orient = self.util.rotate_quaternion_by_axis(upperarm_orient, axis='x', degrees=-90)
        self.targets_pos_upperarm_world = []
        self.targets_orn_upperarm_world = []
        for target_pos_on_arm, target_orn_on_arm, target in zip(self.targets_pos_on_upperarm, self.targets_orn_on_upperarm, self.targets_upperarm):
            target_pos, target_orn = p.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_arm, target_orn_on_arm, physicsClientId=self.pid)
            self.targets_pos_upperarm_world.append(target_pos)
            self.targets_orn_upperarm_world.append(target_orn)
            p.resetBasePositionAndOrientation(target, target_pos, target_orn, physicsClientId=self.pid)

        # forearm targets position & orientation
        forearm_pos, forearm_orient = p.getLinkState(self.humanoid_id, self.elbow, computeForwardKinematics=True, physicsClientId=self.pid)[4:6]
        forearm_orient = self.util.rotate_quaternion_by_axis(forearm_orient, axis='x', degrees=-90)
        self.targets_pos_forearm_world = []
        self.targets_orn_forearm_world = []
        for target_pos_on_arm, target_orn_on_arm, target in zip(self.targets_pos_on_forearm, self.targets_orn_on_forearm, self.targets_forearm):
            target_pos, target_orn = p.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_arm, target_orn_on_arm, physicsClientId=self.pid)
            self.targets_pos_forearm_world.append(target_pos)
            self.targets_orn_forearm_world.append(target_orn)
            p.resetBasePositionAndOrientation(target, target_pos, target_orn, physicsClientId=self.pid)

    def mark_feasible_targets(self):
        # change color for feasible targets
        for target in self.feasible_targets:
            p.changeVisualShape(target, -1, rgbaColor=[0, 0.2, 1, 1], physicsClientId=self.pid)

    def unmark_feasible_targets(self):
        # change color for feasible targets
        for target in self.feasible_targets:
            p.changeVisualShape(target, -1, rgbaColor=[0, 1, 1, 1], physicsClientId=self.pid)
        
    def reorder_feasible_targets(self, targeted_arm):
        if targeted_arm == 'upperarm':
            col_offset = int(self.upperarm_length//self.distance_between_points)
        elif targeted_arm == 'forearm':
            col_offset = int(self.forearm_length//self.distance_between_points)
        else:
            raise ValueError('invalid targeted arm! must be upperarm or forearm..')

        feasible_targets_pos_world = []
        feasible_targets_orn_world = []
        feasible_targets = []
        feasible_targets_indices = []
        row_offset = len(self.feasible_targets)//col_offset

        for i in range(row_offset):
            for j in range(col_offset):
                index = i + row_offset*j
                feasible_targets_pos_world.append(self.feasible_targets_pos_world[index])
                feasible_targets_orn_world.append(self.feasible_targets_orn_world[index])
                feasible_targets.append(self.feasible_targets[index])
                feasible_targets_indices.append(self.feasible_targets_indices[index])

        def reverse_alternate_rows(feasible_targets, col_offset):
            total_rows = row_offset
            new_feasible_targets = []
            
            for row in range(total_rows):
                start_index = row * col_offset
                end_index = start_index + col_offset
                segment = feasible_targets[start_index:end_index]
                
                # Reverse every odd-indexed row
                if row % 2 != 0:
                    segment = segment[::-1]
                
                new_feasible_targets.extend(segment)
            
            # Handle any remaining elements if the list length isn't a perfect multiple of col_offset
            if len(feasible_targets) % col_offset != 0:
                remaining_segment = feasible_targets[total_rows * col_offset:]
                new_feasible_targets.extend(remaining_segment)
            
            return new_feasible_targets
        
        feasible_targets_pos_world = reverse_alternate_rows(feasible_targets_pos_world, col_offset)
        feasible_targets_orn_world = reverse_alternate_rows(feasible_targets_orn_world, col_offset)
        feasible_targets = reverse_alternate_rows(feasible_targets, col_offset)
        feasible_targets_indices = reverse_alternate_rows(feasible_targets_indices, col_offset)

        self.feasible_targets_pos_world = feasible_targets_pos_world
        self.feasible_targets_orn_world = feasible_targets_orn_world
        self.feasible_targets = feasible_targets
        self.feasible_targets_indices = feasible_targets_indices

    def get_feasible_targets_pos(self, targeted_arm):
        if targeted_arm == 'upperarm':
            if len(self.targets_pos_upperarm_world) == 0:
                feasible_targets_found = False
                return feasible_targets_found
        elif targeted_arm == 'forearm':
            if len(self.targets_pos_forearm_world) == 0:
                feasible_targets_found = False
                return feasible_targets_found
        else:
            raise ValueError('invalid targeted arm! must be upperarm or forearm..')
        
        # randomize order of trial
        self.target_order_flags = {'upperarm_front': False,
                                   'upperarm_back': False,
                                   'forearm_front': False,
                                   'forearm_back': False}
        if targeted_arm == 'upperarm':
            self.target_trial_order = ['upperarm_front', 'upperarm_back']
        elif targeted_arm == 'forearm':
            self.target_trial_order = ['forearm_front', 'forearm_back']
        else:
            raise ValueError('invalid targeted arm! must be upperarm or forearm..')
        
        self.target_trial_order = random.sample(self.target_trial_order, len(self.target_trial_order))
        self.target_axis_trial_order = [0, 1]
        self.target_axis_trial_order = random.sample(self.target_axis_trial_order, len(self.target_axis_trial_order))

        # flag
        feasible_targets_found = False

        # split targets to front and back
        def split_targets(targets_pos, axis):
            """Splits targets into front and back halves based on the x or y coordinate. (axis = 0 or 1)"""
            z_positions = np.array([pos[axis] for pos in targets_pos])
            mid_point = np.median(z_positions)
            front_indices = [i for i, pos in enumerate(targets_pos) if pos[axis] > mid_point]
            back_indices = [i for i, pos in enumerate(targets_pos) if pos[axis] <= mid_point]
            return front_indices, back_indices

        def split_half(targets_pos, axis=0):
            """Split targets into front and back, returning indices."""
            front_half_indices, back_half_indices = split_targets(targets_pos, axis)
            return front_half_indices, back_half_indices

        # check both x and y axis
        for axis in self.target_axis_trial_order:
            if len(self.targets_pos_on_upperarm) <= 15:  ## 1/3 left
                front_targets_on_upperarm_indices = [i for i in range(len(self.targets_pos_on_upperarm))]
                back_targets_on_upperarm_indices = front_targets_on_upperarm_indices
            else:
                front_targets_on_upperarm_indices, back_targets_on_upperarm_indices = split_half(self.targets_pos_on_upperarm, axis)
            if len(self.targets_pos_on_forearm) <= 10:   ## 1/3 left
                front_targets_on_forearm_indices = [i for i, pos in enumerate(self.targets_pos_on_forearm)]
                back_targets_on_forearm_indices = front_targets_on_forearm_indices
            else:
                front_targets_on_forearm_indices, back_targets_on_forearm_indices = split_half(self.targets_pos_on_forearm, axis)

            self.target_pos_world_dict = {'upperarm_front': np.array(self.targets_pos_upperarm_world)[front_targets_on_upperarm_indices],
                                          'upperarm_back': np.array(self.targets_pos_upperarm_world)[back_targets_on_upperarm_indices],
                                          'forearm_front': np.array(self.targets_pos_forearm_world)[front_targets_on_forearm_indices],
                                          'forearm_back': np.array(self.targets_pos_forearm_world)[back_targets_on_forearm_indices]}
            self.target_orn_world_dict = {'upperarm_front': np.array(self.targets_orn_upperarm_world)[front_targets_on_upperarm_indices],
                                          'upperarm_back': np.array(self.targets_orn_upperarm_world)[back_targets_on_upperarm_indices],
                                          'forearm_front': np.array(self.targets_orn_forearm_world)[front_targets_on_forearm_indices],
                                          'forearm_back': np.array(self.targets_orn_forearm_world)[back_targets_on_forearm_indices]}
            self.target_dict = {'upperarm_front': np.array(self.targets_upperarm)[front_targets_on_upperarm_indices],
                                'upperarm_back': np.array(self.targets_upperarm)[back_targets_on_upperarm_indices],
                                'forearm_front': np.array(self.targets_forearm)[front_targets_on_forearm_indices],
                                'forearm_back': np.array(self.targets_forearm)[back_targets_on_forearm_indices]}
            self.target_indices_dict = {'upperarm_front': front_targets_on_upperarm_indices,
                                        'upperarm_back': back_targets_on_upperarm_indices,
                                        'forearm_front': front_targets_on_forearm_indices,
                                        'forearm_back': back_targets_on_forearm_indices}
            
            for order_key in self.target_trial_order:
                # reset robot
                self.robot_w.reset()

                # set flag & targets
                self.target_order_flags[order_key] = True
                targets_pos_world = self.target_pos_world_dict[order_key]
                targets_orn_world = self.target_orn_world_dict[order_key]
                targets = self.target_dict[order_key]
                targets_indices = self.target_indices_dict[order_key]

                if len(targets_pos_world) == 0:
                    continue

                # compute world_to_target_point
                world_to_target_points = [[targets_pos_world[0], targets_orn_world[0]],
                                          [targets_pos_world[len(targets_pos_world)//4], targets_orn_world[len(targets_orn_world)//4]],
                                          [targets_pos_world[len(targets_pos_world)//2], targets_orn_world[len(targets_orn_world)//2]],
                                          [targets_pos_world[(len(targets_pos_world)*3)//4], targets_orn_world[(len(targets_orn_world)*3)//4]],
                                          [targets_pos_world[-1], targets_orn_world[-1]]]

                for world_to_target_point in world_to_target_points:
                    # compute desired world_to_eef (check if it can get closer to the target point)
                    world_to_eef = p.multiplyTransforms(world_to_target_point[0], world_to_target_point[1],
                                                        self.target_closer_to_eef[0], self.target_closer_to_eef[1], physicsClientId=self.pid)

                    # set robot initial joint state
                    q_robot_w_closer = p.calculateInverseKinematics(self.robot_w.id, self.robot_w.eef_id, world_to_eef[0], world_to_eef[1],
                                                                lowerLimits=self.robot_w.arm_lower_limits, upperLimits=self.robot_w.arm_upper_limits, 
                                                                jointRanges=self.robot_w.arm_joint_ranges, restPoses=self.robot_w.arm_rest_poses,
                                                                maxNumIterations=40, physicsClientId=self.pid)
                    q_robot_w_closer = [q_robot_w_closer[i] for i in range(len(self.robot_w.arm_controllable_joints))]

                    # invalid joint
                    if min(q_robot_w_closer) < min(self.robot_w.arm_lower_limits) or max(q_robot_w_closer) > max(self.robot_w.arm_upper_limits):
                        # reset flag
                        self.target_order_flags[order_key] = False
                        continue

                    for i, joint_id in enumerate(self.robot_w.arm_controllable_joints):
                        p.resetJointState(self.robot_w.id, joint_id, q_robot_w_closer[i], physicsClientId=self.pid)

                    # check if config is valid
                    eef_pos = p.getLinkState(self.robot_w.id, self.robot_w.eef_id, computeForwardKinematics=True, physicsClientId=self.pid)[0]
                    dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pos))
                    if dist > 0.03 or self.robot_w_in_collision(q_robot_w_closer):
                        # reset flag
                        self.target_order_flags[order_key] = False
                        continue

                    # valid, save this state
                    break

                self.feasible_targets_pos_world = targets_pos_world
                self.feasible_targets_orn_world = targets_orn_world
                self.feasible_targets = targets
                self.feasible_targets_count = len(targets)
                self.feasible_targets_indices = targets_indices
                self.init_q_R = q_robot_w_closer
                self.arm_side = order_key
                feasible_targets_found = True
                break

            if feasible_targets_found:
                break

        return feasible_targets_found
    
    def get_feasible_targets_lists(self):
        return (self.feasible_targets_pos_world, self.feasible_targets_orn_world, 
                self.feasible_targets_count, self.feasible_targets_indices, self.init_q_R, self.arm_side)

    def set_feasible_targets_lists(self, feasible_targets_pos_world, feasible_targets_orn_world,
                                   feasible_targets, feasible_targets_count, feasible_targets_indices,
                                   init_q_R, arm_side):
        self.feasible_targets_pos_world = feasible_targets_pos_world
        self.feasible_targets_orn_world = feasible_targets_orn_world
        self.feasible_targets = feasible_targets
        self.feasible_targets_count = feasible_targets_count
        self.feasible_targets_indices = feasible_targets_indices
        self.init_q_R = init_q_R
        self.arm_side = arm_side

    def get_feasible_targets_given_indices(self, feasible_targets_indices, arm_side):
        if 'upperarm' in arm_side:
            return np.array(self.targets_upperarm)[feasible_targets_indices]
        elif 'forearm' in arm_side:
            return np.array(self.targets_forearm)[feasible_targets_indices]
        else:
            raise ValueError('invalid targeted arm! must be upperarm or forearm..')

    def get_new_contact_points(self, targeted_arm):
        new_contact_points = 0
        indices_to_delete = []
        tool_endtip_pos = p.getLinkState(self.tool, 1, physicsClientId=self.pid)[0]
        # Check feasible targets
        for i, (target_pos_world, target) in enumerate(zip(self.feasible_targets_pos_world, self.feasible_targets)):
            if np.linalg.norm(tool_endtip_pos - target_pos_world) < 0.03:
                # The robot made contact with a point on the person's arm 
                new_contact_points += 1
                indices_to_delete.append(i)

        return new_contact_points, indices_to_delete
    
    def remove_contacted_feasible_targets(self, indices_to_delete, targeted_arm):
        for i in indices_to_delete:
            p.resetBasePositionAndOrientation(self.feasible_targets[i], [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.pid)
            if targeted_arm == 'upperarm':
                self.deleted_targets_indices_on_upperarm.append(self.feasible_targets_indices[i])
            elif targeted_arm == 'forearm':
                self.deleted_targets_indices_on_forearm.append(self.feasible_targets_indices[i])
            else:
                raise ValueError('invalid targeted arm! must be upperarm or forearm..')
            
        self.feasible_targets = [t for i, t in enumerate(self.feasible_targets) if i not in indices_to_delete]
        self.feasible_targets_pos_world = [t for i, t in enumerate(self.feasible_targets_pos_world) if i not in indices_to_delete]
        self.feasible_targets_indices = [t for i, t in enumerate(self.feasible_targets_indices) if i not in indices_to_delete]

    ### arm config NN dataset generation
    def reorder_targets(self):
        # helper function
        def reverse_alternate_rows(targets, col_offset):
            total_rows = row_offset
            new_targets = []
            
            for row in range(total_rows):
                start_index = row * col_offset
                end_index = start_index + col_offset
                segment = targets[start_index:end_index]
                
                # reverse every odd-indexed row
                if row % 2 != 0:
                    segment = segment[::-1]
                
                new_targets.extend(segment)
            
            # handle any remaining elements if the list length isn't a perfect multiple of col_offset
            if len(targets) % col_offset != 0:
                remaining_segment = targets[total_rows * col_offset:]
                new_targets.extend(remaining_segment)
            
            return new_targets

        # 1. reorder upperarm targets
        targets_pos_on_upperarm = []
        targets_orn_on_upperarm = []
        targets_pos_upperarm_world = []
        targets_orn_upperarm_world = []
        targets_upperarm = []
        col_offset = int(self.upperarm_length//self.distance_between_points)
        row_offset = len(self.targets_upperarm)//col_offset

        for i in range(row_offset):
            for j in range(col_offset):
                index = i + row_offset*j
                targets_pos_on_upperarm.append(self.targets_pos_on_upperarm[index])
                targets_orn_on_upperarm.append(self.targets_orn_on_upperarm[index])
                targets_pos_upperarm_world.append(self.targets_pos_upperarm_world[index])
                targets_orn_upperarm_world.append(self.targets_orn_upperarm_world[index])
                targets_upperarm.append(self.targets_upperarm[index])
        
        targets_pos_on_upperarm = reverse_alternate_rows(targets_pos_on_upperarm, col_offset)
        targets_orn_on_upperarm = reverse_alternate_rows(targets_orn_on_upperarm, col_offset)
        targets_pos_upperarm_world = reverse_alternate_rows(targets_pos_upperarm_world, col_offset)
        targets_orn_upperarm_world = reverse_alternate_rows(targets_orn_upperarm_world, col_offset)
        targets_upperarm = reverse_alternate_rows(targets_upperarm, col_offset)

        self.targets_pos_on_upperarm = targets_pos_on_upperarm
        self.targets_orn_on_upperarm = targets_orn_on_upperarm
        self.targets_pos_upperarm_world = targets_pos_upperarm_world
        self.targets_orn_upperarm_world = targets_orn_upperarm_world
        self.targets_upperarm = targets_upperarm

        # 2. reorder forearm targets
        targets_pos_on_forearm = []
        targets_orn_on_forearm = []
        targets_pos_forearm_world = []
        targets_orn_forearm_world = []
        targets_forearm = []
        col_offset = int(self.forearm_length//self.distance_between_points)
        row_offset = len(self.targets_forearm)//col_offset

        for i in range(row_offset):
            for j in range(col_offset):
                index = i + row_offset*j
                targets_pos_on_forearm.append(self.targets_pos_on_forearm[index])
                targets_orn_on_forearm.append(self.targets_orn_on_forearm[index])
                targets_pos_forearm_world.append(self.targets_pos_forearm_world[index])
                targets_orn_forearm_world.append(self.targets_orn_forearm_world[index])
                targets_forearm.append(self.targets_forearm[index])
        
        targets_pos_on_forearm = reverse_alternate_rows(targets_pos_on_forearm, col_offset)
        targets_orn_on_forearm = reverse_alternate_rows(targets_orn_on_forearm, col_offset)
        targets_pos_forearm_world = reverse_alternate_rows(targets_pos_forearm_world, col_offset)
        targets_orn_forearm_world = reverse_alternate_rows(targets_orn_forearm_world, col_offset)
        targets_forearm = reverse_alternate_rows(targets_forearm, col_offset)

        self.targets_pos_on_forearm = targets_pos_on_forearm
        self.targets_orn_on_forearm = targets_orn_on_forearm
        self.targets_pos_forearm_world = targets_pos_forearm_world
        self.targets_orn_forearm_world = targets_orn_forearm_world
        self.targets_forearm = targets_forearm

    def visualize_targets_order(self):
        # change color for targets
        for target in self.targets_upperarm:
            p.changeVisualShape(target, -1, rgbaColor=[0, 0.2, 1, 1], physicsClientId=self.pid)

        for target in self.targets_forearm:
            p.changeVisualShape(target, -1, rgbaColor=[0, 0.2, 1, 1], physicsClientId=self.pid)

    def restore_removed_targets(self):
        self.targets_pos_on_upperarm = self.targets_pos_on_upperarm_original
        self.targets_orn_on_upperarm = self.targets_orn_on_upperarm_original
        self.targets_upperarm = self.targets_upperarm_original
        
        self.targets_pos_on_forearm = self.targets_pos_on_forearm_original
        self.targets_orn_on_forearm = self.targets_orn_on_forearm_original
        self.targets_forearm = self.targets_forearm_original

        self.update_targets()

    def save_original_targets(self):
        # Save original lists for restoration
        self.targets_pos_on_upperarm_original = self.targets_pos_on_upperarm
        self.targets_orn_on_upperarm_original = self.targets_orn_on_upperarm
        self.targets_upperarm_original = self.targets_upperarm
        
        self.targets_pos_on_forearm_original = self.targets_pos_on_forearm
        self.targets_orn_on_forearm_original = self.targets_orn_on_forearm
        self.targets_forearm_original = self.targets_forearm

    def remove_some_targets(self, case, use_bias):
        # Save original lists for restoration
        self.save_original_targets()

        # Determine the fraction of targets to keep based on the case
        reduction_factors = {
            1: 1 / 2,
            2: 2 / 3,
            3: 3 / 4,
            4: 4 / 5,
            5: 1 / 3,
            6: 1 / 4,
            7: 1 / 5
        }
        reduction_factor = reduction_factors.get(case, 1)

        def truncate_targets_with_bias(targets, positions, orientations, fraction, bias_axis=0, bias_direction=1):
            """
            Truncates a list of targets with a bias toward one side of the arm.
            The bias is applied along the specified axis (0=x, 1=y, 2=z).
            """
            total_len = len(targets)
            keep_len = int(total_len * fraction)
            
            # Compute the midpoint of the target positions along the biased axis
            mid_value = sum(pos[bias_axis] for pos in positions) / total_len
            
            # Calculate a bias score for each target based on the specified axis and direction
            bias_scores = [
                (pos[bias_axis] - mid_value) * bias_direction for pos in positions
            ]
            
            # Sort targets by bias score to favor one side
            sorted_indices = sorted(range(total_len), key=lambda i: bias_scores[i], reverse=True)
            
            # Select indices to keep and remove
            indices_to_keep = sorted(sorted_indices[:keep_len])
            truncated_targets = [targets[i] for i in indices_to_keep]
            truncated_positions = [positions[i] for i in indices_to_keep]
            truncated_orientations = [orientations[i] for i in indices_to_keep]
            removed_targets = [targets[i] for i in range(total_len) if i not in indices_to_keep]
            
            return truncated_targets, truncated_positions, truncated_orientations, removed_targets

        def truncate_targets_without_bias(targets, positions, orientations, fraction):
            """
            Truncates a list of targets randomly without any bias.
            """
            total_len = len(targets)
            keep_len = int(total_len * fraction)
            indices_to_keep = sorted(random.sample(range(total_len), keep_len))
            
            truncated_targets = [targets[i] for i in indices_to_keep]
            truncated_positions = [positions[i] for i in indices_to_keep]
            truncated_orientations = [orientations[i] for i in indices_to_keep]
            removed_targets = [targets[i] for i in range(total_len) if i not in indices_to_keep]
            
            return truncated_targets, truncated_positions, truncated_orientations, removed_targets

        # Define a bias direction for asymmetry: -1 favors one side, +1 the other
        bias_direction = random.choice([-1, 1])

        # Truncate targets for the upper arm
        if use_bias:
            self.targets_upperarm, self.targets_pos_on_upperarm, self.targets_orn_on_upperarm, remove_targets_upperarm = truncate_targets_with_bias(
                self.targets_upperarm, self.targets_pos_on_upperarm, self.targets_orn_on_upperarm, 
                reduction_factor, bias_axis=1, bias_direction=bias_direction
            )
        else:
            self.targets_upperarm, self.targets_pos_on_upperarm, self.targets_orn_on_upperarm, remove_targets_upperarm = truncate_targets_without_bias(
                self.targets_upperarm, self.targets_pos_on_upperarm, self.targets_orn_on_upperarm, 
                reduction_factor
            )

        # Truncate targets for the forearm
        if use_bias:
            self.targets_forearm, self.targets_pos_on_forearm, self.targets_orn_on_forearm, remove_targets_forearm = truncate_targets_with_bias(
                self.targets_forearm, self.targets_pos_on_forearm, self.targets_orn_on_forearm, 
                reduction_factor, bias_axis=1, bias_direction=bias_direction
            )
        else:
            self.targets_forearm, self.targets_pos_on_forearm, self.targets_orn_on_forearm, remove_targets_forearm = truncate_targets_without_bias(
                self.targets_forearm, self.targets_pos_on_forearm, self.targets_orn_on_forearm, 
                reduction_factor
            )

        # Move "removed" targets far away in the simulation
        for target in remove_targets_upperarm:
            p.resetBasePositionAndOrientation(target, [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.pid)
        for target in remove_targets_forearm:
            p.resetBasePositionAndOrientation(target, [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.pid)

        # Update the targets after modification
        self.update_targets()

    def get_targets(self):
        return (self.targets_pos_on_upperarm, self.targets_orn_on_upperarm, self.targets_upperarm,
                self.targets_pos_on_forearm, self.targets_orn_on_forearm, self.targets_forearm)

    def set_targets(self, targets_pos_on_upperarm, targets_orn_on_upperarm, targets_upperarm,
                    targets_pos_on_forearm, targets_orn_on_forearm, targets_forearm):
        self.targets_pos_on_upperarm = targets_pos_on_upperarm
        self.targets_orn_on_upperarm = targets_orn_on_upperarm
        self.targets_upperarm = targets_upperarm

        self.targets_pos_on_forearm = targets_pos_on_forearm
        self.targets_orn_on_forearm = targets_orn_on_forearm
        self.targets_forearm = targets_forearm

    def move_away_all_targets(self):
        for target in self.targets_upperarm:
            p.resetBasePositionAndOrientation(target, [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.pid)
        for target in self.targets_forearm:
            p.resetBasePositionAndOrientation(target, [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.pid)