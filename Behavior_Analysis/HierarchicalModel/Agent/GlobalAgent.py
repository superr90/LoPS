'''
Description:
    Global agent.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Dec. 31 2020
'''
import numpy as np

import sys
sys.path.append("../../../Utils")
from Utils.ComputationUtils import scaleOfNumber, makeChoice

# ==================================================

class SimpleGlobal:

    def __init__(self, adjacent_data, locs_df, reward_amount, root, energizer_data, bean_data, ghost_data, reward_type,
                 fruit_pos, ghost_status, last_dir,
                 depth=10, ignore_depth=0, ghost_attractive_thr=34, ghost_repulsive_thr=10, fruit_attractive_thr=10,
                 randomness_coeff=1.0, laziness_coeff=1.0, reward_coeff=1.0, risk_coeff=1.0):
        '''
        Initialization.
        :param adjacent_data: Map adjacent data (dict).
        :param locs_df: Locations distance (dict).
        :param reward_amount: Reward amount (dict).
        :param root: Pacman position of 2-tuple.
        :param energizer_data: A list of positions of energizers. Each position should be a 2-tuple.
        :param bean_data: A list of positions of bens. Each position should be a 2-tuple.
        :param ghost_data: A list of positions of ghosts. Each position should be a 2-tuple. If no ghost exists, pass np.nan.
        :param reward_type: The type pf reward (int).
        :param fruit_pos: The position of fruit. Should be a 2-tuple.
        :param ghost_status: A list of ghost status. Each status should be either 1(normal) or 4 (scared). If no ghost exists, pass np.nan.
        :param depth: The maximum depth of tree.
        :param ignore_depth: Ignore this depth of nodes.
        :param ghost_attractive_thr: Ghost attractive threshold.
        :param ghost_repulsive_thr: Ghost repulsive threshold.
        :param fruit_attractive_thr: Fruit attractive threshold.
        :param reward_coeff: Coefficient for the reward.
        :param risk_coeff: Coefficient for the risk.
        '''
        # Parameter type check
        if not isinstance(root, tuple):
            raise TypeError("The root should be a 2-tuple, but got a {}.".format(type(root)))
        if not isinstance(depth, int):
            raise TypeError("The depth should be a integer, but got a {}.".format(type(depth)))
        if depth <= 0:
            raise ValueError("The depth should be a positive integer.")
        # Pacman Pos
        self.cur_pos = root
        # The maximize depth (i.e., the path length)
        self.depth = depth
        # The ignore depth (i.e., exclude this depth of nodes)
        self.ignore_depth = ignore_depth
        # Game status
        self.energizer_data = [tuple(each) for each in energizer_data] if (
                    not isinstance(energizer_data, float) and energizer_data is not None) else np.nan
        self.bean_data = [tuple(each) for each in bean_data] if (
                    not isinstance(bean_data, float) or bean_data is None) else np.nan
        self.ghost_data = [tuple(each) for each in ghost_data]
        self.ghost_status = ghost_status
        # Fruit data
        self.reward_type = reward_type
        self.fruit_pos = fruit_pos
        # Pre-defined thresholds for utility computation
        self.ghost_attractive_thr = ghost_attractive_thr
        self.fruit_attractive_thr = fruit_attractive_thr
        self.ghost_repulsive_thr = ghost_repulsive_thr
        # Other pre-computed data
        self.adjacent_data = adjacent_data
        self.locs_df = locs_df
        self.reward_amount = reward_amount
        self.existing_bean = bean_data
        self.existing_energizer = energizer_data
        self.existing_fruit = fruit_pos
        # Utility (Q-value) for every direction
        self.Q_value = [0, 0, 0, 0]
        # Direction list
        self.dir_list = ['left', 'right', 'up', 'down']
        self.available_dir = []
        self.adjacent_pos = adjacent_data[self.cur_pos]
        for dir in ["left", "right", "up", "down"]:
            if None != self.adjacent_pos[dir] and not isinstance(self.adjacent_pos[dir], float):
                self.available_dir.append(dir)
        if 0 == len(self.available_dir) or 1 == len(self.available_dir):
            raise ValueError("The position {} has {} adjacent positions.".format(self.cur_pos, len(self.available_dir)))
        self.adjacent_pos = [self.adjacent_pos[each] for each in self.available_dir]
        # Last direction
        self.last_dir = last_dir
        # For randomness and laziness
        self.randomness_coeff = randomness_coeff
        self.laziness_coeff = laziness_coeff


    def _dirArea(self,dir):
        # x: 1~28 | y: 1~33
        left_bound = 1
        right_bound = 28
        upper_bound = 1
        lower_bound = 33
        # Area corresponding to the direction
        if dir == "left":
            area = [
                (left_bound, upper_bound),
                (max(1, self.cur_pos[0]-1), lower_bound)
            ]
        elif dir == "right":
            area = [
                (min(right_bound, self.cur_pos[0]+1), upper_bound),
                (right_bound, lower_bound)
            ]
        elif dir == "up":
            area = [
                (left_bound, upper_bound),
                (right_bound, min(lower_bound, self.cur_pos[1]+1))
            ]
        elif dir == "down":
            area = [
                (left_bound, min(lower_bound, self.cur_pos[1]+1)),
                (right_bound, lower_bound)
            ]
        else:
            raise ValueError("Undefined direction {}!".format(dir))
        return area


    def _countBeans(self, upper_left, lower_right):
        area_loc = []
        # Construct a grid area
        for i in range(upper_left[0], lower_right[0]+1):
            for j in range(upper_left[1], lower_right[1]+1):
                area_loc.append((i,j))
        if isinstance(self.bean_data, float) or self.bean_data is None:
            return 0
        else:
            beans_num = 0
            for each in self.bean_data:
                if each in area_loc:
                    beans_num += 1
            return beans_num


    def nextDir(self, return_Q=False):
        available_directions_index = [self.dir_list.index(each) for each in self.available_dir]
        self.Q_value = [0.0, 0.0, 0.0, 0.0]
        for dir in self.available_dir:
            area = self._dirArea(dir)
            beans_num = self._countBeans(area[0], area[1])
            self.Q_value[self.dir_list.index(dir)] = beans_num
        self.Q_value = np.array(self.Q_value, dtype=np.float32)
        # self.Q_value[available_directions_index] += 1.0 # avoid 0 utility
        # Add randomness and laziness
        Q_scale = scaleOfNumber(np.max(np.abs(self.Q_value)))
        if len(available_directions_index) > 0:
            # randomness = np.random.normal(loc=0, scale=0.1, size=len(available_directions_index)) * Q_scale
            randomness = np.random.uniform(low=0, high=0.1, size=len(available_directions_index)) * Q_scale
            self.Q_value[available_directions_index] += (self.randomness_coeff * randomness)
        if self.last_dir is not None and self.dir_list.index(self.last_dir) in available_directions_index:
            self.Q_value[self.dir_list.index(self.last_dir)] += (self.laziness_coeff * Q_scale)
        if return_Q:
            return makeChoice(self.Q_value), self.Q_value
        else:
            return makeChoice(self.Q_value)


# if __name__ == '__main__':
#     from Utils.FileUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath
#     from Utils.ComputationUtils import makeChoice
# 
#     # Read data
#     locs_df = readLocDistance("../../../Data/Constantsdij_distance_map.csv")
#     adjacent_data = readAdjacentMap("../../../Data/Constants/adjacent_map.csv")
#     adjacent_path = readAdjacentPath("../../../Data/Constants/dij_distance_map.csv")
#     reward_amount = readRewardAmount()
#     print("Finished reading auxiliary data!")
# 
#     # An example of data
#     cur_pos = (14, 27)
#     ghost_data = [(13, 27), (15, 27)]
#     ghost_status = [1, 1]
#     energizer_data = [(7, 5), (17, 5), (7, 26), (24, 30)]
#     bean_data = [(2, 5), (4, 5), (5, 5), (16, 5), (18, 5), (20, 5), (24, 5), (25, 5), (16, 6), (7, 7), (13, 7), (27, 7),
#                  (2, 8), (16, 8), (22, 8), (2, 9), (6, 9), (8, 9), (9, 9), (13, 9), (14, 9), (16, 9), (17, 9), (19, 9),
#                  (24, 9), (26, 9), (27, 9), (10, 10), (22, 10), (2, 11), (10, 11), (22, 11), (5, 12), (7, 12), (22, 12),
#                  (22, 13), (7, 14), (13, 14), (16, 14), (7, 15), (7, 17), (22, 17), (7, 19), (10, 23), (2, 24), (3, 24),
#                  (6, 24), (8, 24), (9, 24), (13, 24), (17, 24), (20, 24), (22, 24), (25, 24), (7, 25), (22, 25), (2, 26),
#                  (27, 27), (2, 28), (27, 28), (10, 29), (22, 29), (2, 30), (7, 30), (11, 30), (16, 30), (18, 30), (19, 30),
#                  (22, 30), (27, 30), (2, 32), (5, 33), (9, 33), (12, 33), (14, 33), (15, 33), (16, 33), (17, 33), (18, 33),
#                  (20, 33), (24, 33), (25, 33), (26, 33)]
#     reward_type = 5
#     fruit_pos = (3, 30)
#     last_dir = "left"
# 
#     # Global agent
#     agent = SimpleGlobal(
#         adjacent_data,
#         locs_df,
#         reward_amount,
#         cur_pos,
#         energizer_data,
#         bean_data,
#         ghost_data,
#         reward_type,
#         fruit_pos,
#         ghost_status,
#         last_dir,
#         15,
#         5,
#         34,
#         34,
#         34,
#         reward_coeff=1.0, risk_coeff=0.0,
#         randomness_coeff=1.0, laziness_coeff=1.0
#     )
#     _, Q = agent.nextDir(return_Q=True)
#     choice = agent.dir_list[makeChoice(Q)]
#     print("Global Choice : ", choice, Q)