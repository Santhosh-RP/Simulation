import numpy as np
from .movement_avoidance import attract, repulse, resultant, angle_to_wheel
from .transform_pose import dist_qpos


class Helper(object):
    def __init__(self, food_name, dist_threshold):
        self.helper_state = "helping"
        self.current_wait_steps = 0
        self.food_placed = False
        self.food_name = food_name
        self.curriculum_learning_dist = dist_threshold



    def set_success(self):
        self.helper_state = "waiting"
        self.food_placed = True

    def get_action(self, data, wait_pos, max_wait_steps):

        learner_qpos = self.get_learner_qpos(data)
        goal_qpos = dist_qpos(learner_qpos, 1)
        if self.helper_state == "helping":
            placed = self.place_food(data, goal_qpos)
            if placed:
                self.set_success()
                return np.array([0, 0])*30
            else:
                self.carry_food(data)
                return np.array(self.move_to_goal(data, goal_qpos, 2.0))*30
        elif self.helper_state == "waiting":
            if self.current_wait_steps >= max_wait_steps:
                self.helper_state = "helping"
                self.current_wait_steps = 0
                self.food_placed = False
                return np.array([0, 0])*30
            else:
                self.current_wait_steps += 1
                return np.array(self.wait(data, wait_pos))*30

    def move_to_goal(self, data, goal_qpos, max_dist, intensity=1.0):
        obstacles_qpos = self.get_obstacles_qpos(data)
        helper_qpos = self.get_helper_qpos(data)
        attract_angle = attract(helper_qpos, goal_qpos)
        comb_repulse_vector = np.array([0.0, 0.0])

        for obstacle_qpos in obstacles_qpos:
            repulse_angle, repulse_magnitude = repulse(helper_qpos, obstacle_qpos, max_dist, intensity)
            repulse_vector = np.array(
                [np.cos(repulse_angle) * repulse_magnitude, np.sin(repulse_angle) * repulse_magnitude])
            comb_repulse_vector += repulse_vector

        if np.linalg.norm(comb_repulse_vector) > 0:
            comb_repulse_angle = np.arctan2(comb_repulse_vector[1], comb_repulse_vector[0])
        else:
            comb_repulse_angle = attract_angle

        resultant_angle = resultant(attract_angle, comb_repulse_angle, np.linalg.norm(comb_repulse_vector))
        wheel_actions = angle_to_wheel(resultant_angle, threshold=np.radians(30))

        return wheel_actions

    def carry_food(self, data):
        helper_qpos = self.get_helper_qpos(data)
        food_helper_qpos = dist_qpos(helper_qpos, 0.8)
        food_helper_qpos[2] = 0.75
        self.set_food_qpos(data, self.food_name, food_helper_qpos)

    def in_range(self, current_qpos, target_pos, range):
        lower_left = [target_pos[0] - range, target_pos[1] - range]
        upper_right = [target_pos[0] + range, target_pos[1] + range]
        in_range = lower_left[0] <= current_qpos[0] <= upper_right[0] and lower_left[1] <= current_qpos[1] <= upper_right[1]
        return in_range

    def place_food(self, data, goal_qpos):
        food_qpos = self.get_food_qpos(data)
        in_range = self.in_range(food_qpos, goal_qpos, self.curriculum_learning_dist)
        return in_range

    def wait(self, data, wait_pos):
        helper_qpos = self.get_helper_qpos(data)
        wait_qpos = np.copy(helper_qpos)
        wait_qpos[:2] = wait_pos
        action = self.move_to_goal(data, wait_qpos, 1)
        if self.in_range(helper_qpos, wait_pos, self.curriculum_learning_dist):
            return np.array([0, 0])
        else:
            return action

    def get_helper_qpos(self, data):
        return data.joint('Hroot').qpos

    def get_learner_qpos(self, data):
        return data.joint('root').qpos

    def get_food_qpos(self, data):
        return data.joint(self.food_name).qpos

    def set_food_qpos(self, data, food_name, qpos):
        data.joint(food_name).qpos = qpos

    def get_obstacles_qpos(self, data):
        obstacles_qpos = []
        for i in range(1, 11):
            food = 'food_free_' + str(i)
            obstacles_qpos.append(data.joint(food).qpos)
        if self.food_placed:
            obstacles_qpos.append(data.joint("food_free_11").qpos)

        return obstacles_qpos
