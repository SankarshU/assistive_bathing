from __future__ import print_function
from __future__ import division
import pybullet as p
import numpy as np
from collections import defaultdict, deque, namedtuple
from itertools import product, combinations

INF = np.inf
PI = np.pi
CIRCULAR_LIMITS = -PI, PI
MAX_DISTANCE = 0


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def circular_difference(theta2, theta1):
    return wrap_angle(theta2 - theta1)


def get_pose(body, client_id):
    return p.getBasePositionAndOrientation(body, physicsClientId=client_id)


def get_bodies(client_id):
    return [p.getBodyUniqueId(i, physicsClientId=client_id)
            for i in range(p.getNumBodies(physicsClientId=client_id))]


BodyInfo = namedtuple('BodyInfo', ['body_id', 'base_name', 'body_name'])


def get_body_info(body, client_id):
    return BodyInfo(body, *p.getBodyInfo(body, physicsClientId=client_id))


def get_base_name(body, client_id):
    return get_body_info(body, client_id).base_name.decode(encoding='UTF-8')


def get_body_name(body, client_id):
    return get_body_info(body, client_id).body_name.decode(encoding='UTF-8')


def get_name(body, client_id):
    name = get_body_name(body, client_id)
    if name == '':
        name = 'body'
    return '{}{}'.format(name, int(body))


def has_body(name, client_id):
    try:
        body_from_name(name, client_id)
    except ValueError:
        return False
    return True


def body_from_name(name, client_id):
    for body in get_bodies(client_id):
        if get_body_name(body, client_id) == name:
            return body
    raise ValueError(name)


def remove_body(body, client_id):
    return p.removeBody(body, physicsClientId=client_id)


JOINT_TYPES = {
    p.JOINT_REVOLUTE: 'revolute',  # 0
    p.JOINT_PRISMATIC: 'prismatic',  # 1
    p.JOINT_SPHERICAL: 'spherical',  # 2
    p.JOINT_PLANAR: 'planar',  # 3
    p.JOINT_FIXED: 'fixed',  # 4
    p.JOINT_POINT2POINT: 'point2point',  # 5
    p.JOINT_GEAR: 'gear',  # 6
}


def get_num_joints(body, client_id):
    return p.getNumJoints(body, physicsClientId=client_id)


def get_joints(body, client_id):
    return list(range(get_num_joints(body, client_id)))


def get_joint(body, joint_or_name, client_id):
    if type(joint_or_name) is str:
        return joint_from_name(body, joint_or_name, client_id)
    return joint_or_name


JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])


def get_joint_info(body, joint, client_id):
    return JointInfo(*p.getJointInfo(body, joint, physicsClientId=client_id))


def get_joint_name(body, joint, client_id):
    return get_joint_info(body, joint, client_id).jointName.decode('UTF-8')


def joint_from_name(body, name, client_id):
    for joint in get_joints(body, client_id):
        if get_joint_name(body, joint, client_id) == name:
            return joint
    raise ValueError(body, name)


def has_joint(body, name, client_id):
    try:
        joint_from_name(body, name, client_id)
    except ValueError:
        return False
    return True


def joints_from_names(body, names, client_id):
    return tuple(joint_from_name(body, name, client_id) for name in names)


JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])


def get_joint_state(body, joint, client_id):
    return JointState(*p.getJointState(body, joint, physicsClientId=client_id))


def get_joint_position(body, joint, client_id):
    return get_joint_state(body, joint, client_id).jointPosition


def get_joint_torque(body, joint, client_id):
    return get_joint_state(body, joint, client_id).appliedJointMotorTorque


def get_joint_positions(body, joints=None, client_id=0):
    return tuple(get_joint_position(body, joint, client_id) for joint in joints)


def set_joint_position(body, joint, value, client_id):
    p.resetJointState(body, joint, value, physicsClientId=client_id)


def set_joint_positions(body, joints, values, client_id):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        set_joint_position(body, joint, value, client_id)


def get_configuration(body, client_id):
    return get_joint_positions(body, get_movable_joints(body, client_id), client_id)


def set_configuration(body, values, client_id):
    set_joint_positions(body, get_movable_joints(body, client_id), values, client_id)


def get_full_configuration(body, client_id):
    # Cannot alter fixed joints
    return get_joint_positions(body, get_joints(body, client_id), client_id)


def get_joint_type(body, joint, client_id):
    return get_joint_info(body, joint, client_id).jointType


def is_movable(body, joint, client_id):
    return get_joint_type(body, joint, client_id) != p.JOINT_FIXED


def get_movable_joints(body, client_id):  # 45 / 87 on pr2
    return [joint for joint in get_joints(body, client_id) if is_movable(body, joint, client_id)]


def joint_from_movable(body, index, client_id):
    return get_joints(body, client_id)[index]


def is_circular(body, joint, client_id):
    joint_info = get_joint_info(body, joint, client_id)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    if joint_info.jointUpperLimit < joint_info.jointLowerLimit:
        raise ValueError("circular joint, check it out!")


def get_joint_limits(body, joint, client_id):
    if is_circular(body, joint, client_id):
        return CIRCULAR_LIMITS
    joint_info = get_joint_info(body, joint, client_id)
    return joint_info.jointLowerLimit, joint_info.jointUpperLimit


def get_joints_limits(body, joints, client_id):
    lower_limit = []
    upper_limit = []
    for joint in joints:
        lower_limit.append(get_joint_info(body, joint, client_id).jointLowerLimit)
        upper_limit.append(get_joint_info(body, joint, client_id).jointUpperLimit)
    return lower_limit, upper_limit


def get_min_limit(body, joint, client_id):
    return get_joint_limits(body, joint, client_id)[0]


def get_max_limit(body, joint, client_id):
    return get_joint_limits(body, joint, client_id)[1]


def get_max_velocity(body, joint, client_id):
    return get_joint_info(body, joint, client_id).jointMaxVelocity


def get_max_force(body, joint, client_id):
    return get_joint_info(body, joint, client_id).jointMaxForce


def get_joint_q_index(body, joint, client_id):
    return get_joint_info(body, joint, client_id).qIndex


def get_joint_v_index(body, joint, client_id):
    return get_joint_info(body, joint, client_id).uIndex


def get_joint_axis(body, joint, client_id):
    return get_joint_info(body, joint, client_id).jointAxis


def get_joint_parent_frame(body, joint, client_id):
    joint_info = get_joint_info(body, joint, client_id)
    return joint_info.parentFramePos, joint_info.parentFrameOrn


def violates_limit(body, joint, value, client_id):
    if not is_circular(body, joint, client_id):
        lower, upper = get_joint_limits(body, joint, client_id)
        if (value < lower) or (upper < value):
            return True
    return False


def violates_limits(body, joints, values, client_id):
    return any(violates_limit(body, joint, value, client_id) for joint, value in zip(joints, values))


def wrap_joint(body, joint, value, client_id):
    if is_circular(body, joint, client_id):
        return wrap_angle(value, client_id)
    return value


BASE_LINK = -1
STATIC_MASS = 0

get_num_links = get_num_joints
get_links = get_joints


def get_link_name(body, link, client_id):
    if link == BASE_LINK:
        return get_base_name(body, client_id)
    return get_joint_info(body, link, client_id).linkName.decode('UTF-8')


def get_link_parent(body, link, client_id):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link, client_id).parentIndex


LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                     'localInertialFramePosition', 'localInertialFrameOrientation',
                                     'worldLinkFramePosition', 'worldLinkFrameOrientation'])

LinkInfo = namedtuple('LinkInfo', ['linkName', 'linkIndex', 'jointName', 'jointIndex'])


def get_link_state(body, link, client_id):
    if p.getNumJoints(body, physicsClientId=client_id) == 0:
        raise ValueError('{} does not have any link!'.format(body))
    return LinkState(*p.getLinkState(body, link, physicsClientId=client_id))


def get_link_info(body, link, client_id):
    jointInfo = get_joint_info(body, link, client_id)
    linkName = jointInfo.linkName
    linkIndex = jointInfo.jointIndex
    jointName = jointInfo.jointName
    jointIndex = jointInfo.jointIndex
    return LinkInfo(linkName, linkIndex, jointName, jointIndex)


def get_com_pose(body, link, client_id):  # COM = center of mass
    link_state = get_link_state(body, link, client_id)
    return link_state.linkWorldPosition, link_state.linkWorldOrientation


def get_link_inertial_pose(body, link, client_id):
    link_state = get_link_state(body, link, client_id)
    return link_state.localInertialFramePosition, link_state.localInertialFrameOrientation


def get_link_pose(body, link, client_id):
    if link == BASE_LINK:
        return get_pose(body, client_id)
    # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
    link_state = get_link_state(body, link, client_id)
    return link_state.worldLinkFramePosition, link_state.worldLinkFrameOrientation


def get_all_link_parents(body, client_id):
    return {link: get_link_parent(body, link, client_id) for link in get_links(body, client_id)}


def get_all_link_children(body, client_id):
    children = {}
    for child, parent in get_all_link_parents(body, client_id).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children


def get_link_children(body, link, client_id):
    children = get_all_link_children(body, client_id)
    return children.get(link, [])


def get_link_ancestors(body, link, client_id):
    parent = get_link_parent(body, link, client_id)
    if parent is None:
        return []
    return get_link_ancestors(body, parent, client_id) + [parent]


def get_joint_ancestors(body, link, client_id):
    return get_link_ancestors(body, link, client_id) + [link]


def get_movable_joint_ancestors(body, link, client_id):
    return list(filter(lambda j: is_movable(body, j, client_id), get_joint_ancestors(body, link, client_id)))


def get_link_descendants(body, link, client_id):
    descendants = []
    for child in get_link_children(body, link, client_id):
        descendants.append(child)
        descendants += get_link_descendants(body, child, client_id)
    return descendants


def are_links_adjacent(body, link1, link2, client_id):
    return (get_link_parent(body, link1, client_id) == link2) or \
           (get_link_parent(body, link2, client_id) == link1)


def get_adjacent_links(body, client_id):
    adjacent = set()
    for link in get_links(body, client_id):
        parent = get_link_parent(body, link, client_id)
        adjacent.add((link, parent))
        # adjacent.add((parent, link))
    return adjacent


def get_adjacent_fixed_links(body, client_id):
    return list(filter(lambda item: not is_movable(body, item[0], client_id),
                       get_adjacent_links(body, client_id)))


def get_fixed_links(body, client_id):
    edges = defaultdict(list)
    for link, parent in get_adjacent_fixed_links(body, client_id):
        edges[link].append(parent)
        edges[parent].append(link)
    visited = set()
    fixed = set()
    for initial_link in get_links(body):
        if initial_link in visited:
            continue
        cluster = [initial_link]
        queue = deque([initial_link])
        visited.add(initial_link)
        while queue:
            for next_link in edges[queue.popleft()]:
                if next_link not in visited:
                    cluster.append(next_link)
                    queue.append(next_link)
                    visited.add(next_link)
        fixed.update(product(cluster, cluster))
    return fixed


def pairwise_collision(body1, body2, client_id, max_distance=MAX_DISTANCE):  # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  physicsClientId=client_id)) != 0  # getContactPoints


def pairwise_link_collision(body1, link1, body2, link2, client_id, max_distance=MAX_DISTANCE):  # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  linkIndexA=link1, linkIndexB=link2,
                                  physicsClientId=client_id)) != 0  # getContactPoints


def single_collision(body1, client_id, **kwargs):
    for body2 in get_bodies(client_id):
        if (body1 != body2) and pairwise_collision(body1, body2, client_id, **kwargs):
            return True
    return False


def all_collision(client_id, **kwargs):
    bodies = get_bodies(client_id)
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if pairwise_collision(bodies[i], bodies[j], client_id, **kwargs):
                return True
    return False


def get_moving_links(body, moving_joints, client_id):
    moving_links = list(moving_joints)
    for link in moving_joints:
        moving_links += get_link_descendants(body, link, client_id)
    return list(set(moving_links))


def get_moving_pairs(body, moving_joints, client_id):
    moving_links = get_moving_links(body, moving_joints, client_id)
    for i in range(len(moving_links)):
        link1 = moving_links[i]
        ancestors1 = set(get_joint_ancestors(body, link1, client_id)) & set(moving_joints)
        for j in range(i + 1, len(moving_links)):
            link2 = moving_links[j]
            ancestors2 = set(get_joint_ancestors(body, link2, client_id)) & set(moving_joints)
            if ancestors1 != ancestors2:
                yield link1, link2


def get_self_link_pairs(body, joints, disabled_collisions=set(), client_id=0):
    moving_links = get_moving_links(body, joints, client_id)
    fixed_links = list(set(get_links(body, client_id)) - set(moving_links))
    check_link_pairs = list(product(moving_links, fixed_links))
    if True:
        check_link_pairs += list(get_moving_pairs(body, joints, client_id))
    else:
        check_link_pairs += list(combinations(moving_links, 2))
    check_link_pairs = list(filter(lambda pair: not are_links_adjacent(body, *pair, client_id), check_link_pairs))
    check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                (pair[::-1] not in disabled_collisions), check_link_pairs))
    return check_link_pairs


def get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions, client_id):
    check_link_pairs = get_self_link_pairs(body, joints, disabled_collisions, client_id=client_id) if self_collisions else []
    moving_bodies = [body] + [attachment.child for attachment in attachments]
    if obstacles is None:
        obstacles = list(set(get_bodies(client_id=client_id)) - set(moving_bodies))
    check_body_pairs = list(product(moving_bodies, obstacles))  # + list(combinations(moving_bodies, 2))

    def collision_fn(q):
        if violates_limits(body, joints, q, client_id=client_id):
            return True
        set_joint_positions(body, joints, q, client_id=client_id)
        for attachment in attachments:
            attachment.assign()
        for link1, link2 in check_link_pairs:
            if pairwise_link_collision(body, link1, body, link2, client_id=client_id):
                return True
        return any(pairwise_collision(*pair, client_id=client_id) for pair in check_body_pairs)

    return collision_fn
