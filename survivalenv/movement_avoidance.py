import numpy as np
from .transform_pose import get_pose_qpos


def attract(agent_qpos, goal_qpos):
    relative_goal_pos = get_pose_qpos(goal_qpos, agent_qpos)
    goal_rel_coords = relative_goal_pos[:2]

    attract_angle = np.arctan2(goal_rel_coords[1], goal_rel_coords[0])
    attract_angle = attract_angle % (2 * np.pi)

    return attract_angle


def repulse(agent_qpos, object_qpos, max_dist, intensity=1.0):
    relative_food_pos = get_pose_qpos(object_qpos, agent_qpos)
    object_rel_coords = relative_food_pos[:2]
    angle = np.arctan2(object_rel_coords[1], object_rel_coords[0])
    if angle < 0:
        angle += 2 * np.pi
    repulse_angle = angle + np.pi
    repulse_angle = repulse_angle % (2 * np.pi)

    object_dist = np.linalg.norm(object_rel_coords)
    repulse_magnitude = (max_dist - (object_dist * intensity)) / max_dist
    if repulse_magnitude < 0:
        repulse_magnitude = 0

    return repulse_angle, repulse_magnitude


def resultant(attract_angle, repulse_angle, repulse_magnitude, max_repulse_magnitude=1.0):
    repulse_magnitude = repulse_magnitude / max_repulse_magnitude
    delta_angle = np.arctan2(np.sin(repulse_angle - attract_angle), np.cos(repulse_angle - attract_angle))

    resultant_angle = attract_angle + (delta_angle * repulse_magnitude)
    resultant_angle = resultant_angle % (2 * np.pi)

    return resultant_angle

def angle_to_wheel(target_angle, threshold=np.radians(10)):

    norm_angle = (target_angle + np.pi) % (2 * np.pi) - np.pi
    abs_error = abs(norm_angle)
    if abs_error < threshold:
        return [1, 1]
    else:
        if norm_angle > 0:
            # Rotating towards the left
            return [-1 * abs_error, 1 * abs_error]
        else:
            # Rotating towards the right
            return [1 * abs_error, -1 * abs_error]


# agent_test_qpos = [1, 1, 0.06999969, 0.38268343, 0, 0, 0.92387953]
# food_test_pos = [2.00000000e+00, 2.00000000e+00, 3.09632818e-01, 1.00000000e+00, 1.10641257e-18, 0.00000000e+00, 0.00000000e+00]
# goal_test_pos = [3.00000000e+00, 3.00000000e+00, 3.09632818e-01, 1.00000000e+00, 1.10641257e-18, 0.00000000e+00, 0.00000000e+00]
# repulse_ang, repulse_mag = repulse(agent_test_qpos, food_test_pos, 2.828427)
# repulse_ang_d = repulse_ang * 180 / np.pi
# attract_ang = attract(agent_test_qpos, goal_test_pos)
# attract_ang_d = attract_ang * 180 / np.pi
# resultant_ang2 = resultant(attract_ang, repulse_ang, repulse_mag)
# resultant_ang2_d = resultant_ang2 * 180 / np.pi











