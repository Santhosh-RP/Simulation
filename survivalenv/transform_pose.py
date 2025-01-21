import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import matplotlib.pyplot as plt


def quat_angle(quat):
    q_w, q_x, q_y, q_z = quat
    angle = np.arctan2(2 * (q_w * q_z + q_x * q_y), 1 - 2 * (q_y ** 2 + q_z ** 2))
    return angle


def transform(x, y, angle):
    rotation_matrix_2d = pr.active_matrix_from_angle(2, angle)
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0:3] = rotation_matrix_2d
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 3] = y
    return transformation_matrix


def get_pose(learner_pose, other_pose):
    agent_transform = transform(*learner_pose)
    other_transform = transform(*other_pose)
    other_inverse = pt.invert_transform(other_transform)
    relative_matrix = np.dot(other_inverse, agent_transform)
    final_pose = [relative_matrix[0, 3], relative_matrix[1, 3],
                  np.arctan2(relative_matrix[1, 0], relative_matrix[0, 0])]
    return final_pose


def get_pose_qpos(learner_qpos, other_qpos):
    agent_angle = quat_angle(learner_qpos[3:])
    other_angle = quat_angle(other_qpos[3:])
    agent_transform = transform(learner_qpos[0], learner_qpos[1], agent_angle)
    other_transform = transform(other_qpos[0], other_qpos[1], other_angle)
    other_inverse = pt.invert_transform(other_transform)

    relative_matrix = np.dot(other_inverse, agent_transform)
    final_pose = [relative_matrix[0, 3], relative_matrix[1, 3],
                  np.arctan2(relative_matrix[1, 0], relative_matrix[0, 0])]
    return final_pose

# learner_qpos = [-1.00000000e+00,  0.00000000e+00,  6.99333876e-02,  9.23879533e-01,
#   6.31031096e-05, -1.52344383e-04,  3.82683432e-01]
#
# other_qpos = [ 1.00000000e+00, 0.00000000e+00,  6.99297389e-02,  3.82683432e-01,
#   5.42397542e-04,-2.24668418e-04,  9.23879533e-01]
#


def dist_pose(pose, distance):
    x, y, angle = pose
    new_x = x + np.cos(angle) * distance
    new_y = y + np.sin(angle) * distance
    return np.array([new_x, new_y, angle])


def dist_qpos(qpos, dist):
    ang = quat_angle(qpos[3:])
    pose = [qpos[0], qpos[1], ang]
    new_pose = dist_pose(pose, dist)
    new_qpos = np.copy(qpos)
    new_qpos[:2] = new_pose[:2]
    new_quat = pr.quaternion_from_angle(2, new_pose[2])
    new_qpos[3:] = new_quat

    return new_qpos



def visualise(ax, pose, color, label):
    x, y, angle = pose
    dx = np.cos(angle)
    dy = np.sin(angle)
    ax.plot(x, y, 'o', color=color)
    ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc=color, ec=color)
    if label:
        ax.text(x + 0.1, y + 0.1, label, color=color)



