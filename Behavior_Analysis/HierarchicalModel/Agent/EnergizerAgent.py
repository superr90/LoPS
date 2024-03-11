'''
Description:
    Energizer agent which moves towards energizers.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>
'''
import numpy as np
import anytree
from collections import deque
import copy

import sys
sys.path.append("../../../Utils")
from Utils.ComputationUtils import scaleOfNumber, makeChoice

# ==================================================

class EnergizerTree:

    def __init__(self, adjacent_data, locs_df, reward_amount, root, energizer_data, bean_data, ghost_data, reward_type, fruit_pos, ghost_status, last_dir,
                 depth = 10, ignore_depth = 0, ghost_attractive_thr = 34, ghost_repulsive_thr = 10, fruit_attractive_thr = 10,
                 randomness_coeff = 1.0, laziness_coeff = 1.0, reward_coeff = 1.0, risk_coeff = 1.0):
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
        # The maximize depth (i.e., the path length)
        self.depth = depth
        # The ignore depth (i.e., exclude this depth of nodes)
        self.ignore_depth = ignore_depth
        # Game status
        self.energizer_data = [tuple(each) for each in energizer_data] if (not isinstance(energizer_data, float) and energizer_data is not None) else np.nan
        self.bean_data = [tuple(each) for each in bean_data] if (not isinstance(bean_data, float) or bean_data is None) else np.nan
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
        # Last direction
        self.last_dir = last_dir
        # Trade-off between risk and reward
        self.reward_coeff = reward_coeff
        self.risk_coeff = risk_coeff
        # For randomness and laziness
        self.randomness_coeff = randomness_coeff
        self.laziness_coeff = laziness_coeff
        # Pacman is eaten? If so, the path will be ended
        self.is_eaten = False
        # The root node is the path starting point.
        # Other tree nodes should contain:
        #   (1) location ("name")
        #   (2) parent location ("parent")
        #   (3) the direction from its to parent to itself ("dir_from_parent")
        #   (4) utility of this node, reward and  risk are separated ("cur_reward", "cur_risk", "cur_utility")
        #   (5) the cumulative utility so far, reward and risk are separated ("cumulative_reward", "cumulative_risk", "cumulative_utility")
        self.root = anytree.Node(root,
                                 cur_utility=0.0,
                                 cumulative_utility=0.0,
                                 cur_reward=0.0,
                                 cumulative_reward=0.0,
                                 cur_risk=00.,
                                 cumulative_risk=0.0,
                                 existing_beans=copy.deepcopy(self.bean_data),
                                 existing_energizers = copy.deepcopy(self.energizer_data),
                                 existing_fruit = copy.deepcopy(self.fruit_pos),
                                 ghost_status = copy.deepcopy(self.ghost_status),
                                 exact_reward_list = [],
                                 ghost_potential_reward_list = [],
                                 fruit_potential_reward_list = [],
                                 exact_risk_list = [],
                                 potential_risk_list = []
                                 )
        # The current node
        self.current_node = self.root
        # A queue used for append nodes on the tree
        self.node_queue = deque()
        self.node_queue.append(self.root)


    def _construct(self):
        '''
        Construct the utility tree.
        :return: The tree root node (anytree.Node).
        '''
        # construct the first layer firstly (depth = 1)
        self._attachNode(cur_depth = 1, ignore = True if self.ignore_depth > 0 else False) # attach all children of the root (depth = 1)
        self.node_queue.append(None) # the end of layer with depth = 1
        self.node_queue.popleft()
        self.current_node = self.node_queue.popleft()
        cur_depth = 2
        # construct the other parts
        while cur_depth <= self.depth:
            if cur_depth <= self.ignore_depth:
                ignore = True
            else:
                ignore = False
            while None != self.current_node :
                self._attachNode(cur_depth = cur_depth, ignore = ignore)
                self.current_node = self.node_queue.popleft()
            self.node_queue.append(None)
            if 0 == len(self.node_queue):
                break
            self.current_node = self.node_queue.popleft()
            cur_depth += 1

        # Add potential reward/risk for every path
        for each in self.root.leaves:
            each.path_utility = (
                    each.cumulative_utility
                    + self.reward_coeff * (np.mean(each.ghost_potential_reward_list) + np.mean(each.fruit_potential_reward_list))
                    + self.risk_coeff * np.mean(each.potential_risk_list))
        # Find the best path with the highest utility
        best_leaf = self.root.leaves[0]
        for leaf in self.root.leaves:
            if leaf.path_utility > best_leaf.path_utility:
                best_leaf = leaf
        highest_utility = best_leaf.path_utility
        best_path = best_leaf.ancestors
        best_path = [(each.name, each.dir_from_parent) for each in best_path[1:]]
        if best_path == []: # only one step is taken
            best_path = [(best_leaf.name, best_leaf.dir_from_parent)]
        return self.root, highest_utility, best_path


    def _attachNode(self, cur_depth = 0, ignore = False):
        '''
        Attach a node to the utility tree.
        '''
        if 0 == cur_depth:
            raise ValueError("The depth should not be 0!")
        # Find adjacent positions and the corresponding moving directions for the current node
        tmp_data = self.adjacent_data[self.current_node.name]
        for each in ["left", "right", "up", "down"]:
            # do not walk on the wall or walk out of boundary
            # do not turn back
            if None == self.current_node.parent and isinstance(tmp_data[each], float):
                continue
            elif None != self.current_node.parent and \
                    (isinstance(tmp_data[each], float)
                     or tmp_data[each] == self.current_node.parent.name):
                continue
            else:
                # Compute utility
                cur_pos = tmp_data[each]
                if ignore:
                    exact_reward = 0.0
                    ghost_potential_reward = 0.0
                    fruit_potential_reward = 0.0
                    exact_risk = 0.0
                    potential_risk = 0.0
                    existing_beans = copy.deepcopy(self.current_node.existing_beans)
                    existing_energizers = copy.deepcopy(self.current_node.existing_energizers)
                    existing_fruit = copy.deepcopy(self.current_node.existing_fruit)
                    ghost_status = copy.deepcopy(self.current_node.ghost_status)
                else:
                    # Compute reward
                    exact_reward, ghost_potential_reward, fruit_potential_reward,\
                    existing_beans, existing_energizers, existing_fruit, ghost_status = self._computeReward(cur_pos)
                    # Compute risk
                    # if the position is visited before, do not add up the risk to cumulative
                    if cur_pos in [each.name for each in self.current_node.path]:
                        exact_risk = 0.0
                        potential_risk = 0.0
                    else:
                        exact_risk, potential_risk = self._computeRisk(cur_pos)
                # Construct the new node
                exact_reward_list = copy.deepcopy(self.current_node.exact_reward_list)
                ghost_potential_reward_list = copy.deepcopy(self.current_node.ghost_potential_reward_list)
                fruit_potential_reward_list = copy.deepcopy(self.current_node.fruit_potential_reward_list)
                exact_risk_list = copy.deepcopy(self.current_node.exact_risk_list)
                potential_risk_list = copy.deepcopy(self.current_node.potential_risk_list)

                exact_reward_list.append(exact_reward)
                ghost_potential_reward_list.append(ghost_potential_reward)
                fruit_potential_reward_list.append(fruit_potential_reward)
                exact_risk_list.append(exact_risk)
                potential_risk_list.append(potential_risk)

                new_node = anytree.Node(
                        cur_pos,
                        parent = self.current_node,
                        dir_from_parent = each,
                        cur_utility = {
                            "exact_reward":exact_reward,
                            "ghost_potential_reward":ghost_potential_reward,
                            "fruit_potential_reward":fruit_potential_reward,
                            "exact_risk":exact_risk,
                            "potential_risk":potential_risk
                        },
                        cur_reward = {
                            "exact_reward":exact_reward,
                            "ghost_potential_reward":ghost_potential_reward,
                            "fruit_potential_reward":fruit_potential_reward,
                        },
                        cur_risk = {
                            "exact_risk":exact_risk,
                            "potential_risk":potential_risk
                        },
                        cumulative_reward = self.current_node.cumulative_reward + exact_reward,
                        cumulative_risk = self.current_node.cumulative_risk + exact_risk,
                        cumulative_utility = self.current_node.cumulative_utility + self.reward_coeff * exact_reward + self.risk_coeff * exact_risk,
                        existing_beans = existing_beans,
                        existing_energizers = existing_energizers,
                        existing_fruit = existing_fruit,
                        ghost_status = ghost_status,
                        exact_reward_list = exact_reward_list,
                        ghost_potential_reward_list = ghost_potential_reward_list,
                        fruit_potential_reward_list = fruit_potential_reward_list,
                        exact_risk_list = exact_risk_list,
                        potential_risk_list = potential_risk_list,
                        )
                # If the Pacman is eaten, end this path
                if self.is_eaten:
                    self.is_eaten = False
                else:
                    self.node_queue.append(new_node)


    def _computeReward(self, cur_position):
        '''
        Compute rewards for a position.
        '''
        existing_beans = copy.deepcopy(self.current_node.existing_beans)
        existing_energizers = copy.deepcopy(self.current_node.existing_energizers)
        existing_fruit = copy.deepcopy(self.current_node.existing_fruit)
        ghost_status = copy.deepcopy(self.current_node.ghost_status)
        exact_reward = 0.0
        # Energizer reward
        if isinstance(existing_energizers, float) or cur_position not in existing_energizers:
            exact_reward += 0.0
        elif cur_position in existing_energizers:
            # Reward for eating the energizer
            exact_reward += self.reward_amount[2]
            existing_energizers.remove(cur_position)
            ghost_status = [4 if each != 3 else 3 for each in ghost_status]  # change ghost status
        else:
            pass
        ghost_potential_reward = 0.0
        fruit_potential_reward = 0.0
        return exact_reward, ghost_potential_reward, fruit_potential_reward, existing_beans, existing_energizers, existing_fruit, ghost_status


    def _computeRisk(self, cur_position):
        '''
        Compute risks for a position.
        '''
        ghost_status =  copy.deepcopy(self.current_node.ghost_status)
        # Compute ghost risk when ghosts are normal
        ifscared1 = ghost_status[0] if not isinstance(ghost_status[0], float) else 0
        ifscared2 = ghost_status[1] if not isinstance(ghost_status[1], float) else 0
        exact_risk = 0.0
        potential_risk = 0.0
        if ifscared1 <= 2 or ifscared2 <= 2:  # ghosts are normal; use "or" for dealing with dead ghosts
            if 3 == ifscared1:
                # Pacman is eaten
                if cur_position == self.ghost_data[1]:
                    exact_risk = -self.reward_amount[9]
                    self.is_eaten = True

                    exact_risk = 0.0 #TODO: for testing single ghost pessimistic

                    return exact_risk, potential_risk
                ghost_dist = self.locs_df[cur_position][self.ghost_data[1]]
            elif 3 == ifscared2:
                # Pacman is eaten
                if cur_position == self.ghost_data[0]:
                    exact_risk = -self.reward_amount[9]
                    self.is_eaten = True
                    return exact_risk, potential_risk
                ghost_dist = self.locs_df[cur_position][self.ghost_data[0]]
            else:
                # Pacman is eaten
                if cur_position == self.ghost_data[0] or cur_position == self.ghost_data[1]:
                    exact_risk = -self.reward_amount[9]
                    self.is_eaten = True

                    if cur_position == self.ghost_data[1]:
                        exact_risk = 0.0  # TODO: for testing single ghost pessimistic

                    return exact_risk, potential_risk
                # Potential risk
                else:
                    ghost_dist = min(
                        self.locs_df[cur_position][self.ghost_data[0]],
                        self.locs_df[cur_position][self.ghost_data[1]]
                    )
            if ghost_dist < self.ghost_repulsive_thr:
                # risk = -self.reward_amount[9] * 1 / ghost_dist
                # risk = -self.reward_amount[9] * (self.ghost_repulsive_thr / ghost_dist - 1)
                # reward += self.reward_amount[int(self.reward_type)] * ( self.fruit_attractive_thr/ fruit_dist - 1)
                R = self.reward_amount[8]
                T = self.ghost_repulsive_thr
                if ghost_dist <= (self.ghost_repulsive_thr / 2):
                    potential_risk = -((-R / T) * ghost_dist + R)
                else:
                    potential_risk = -((R * T) / (2 * ghost_dist) - R / 2)
            else:
                pass
        # Ghosts are not scared
        else:
            pass

        # TODO: For excluding potential risk
        potential_risk = 0.0
        return exact_risk, potential_risk


    def _descendantUtility(self, node):
        '''
        Compute path utility.
        '''
        leaves_utility = []
        for each in node.leaves:
            leaves_utility.append(each.path_utility)
        return sum(leaves_utility) / len(leaves_utility)


    def nextDir(self, return_Q = False):
        '''
        Estimate the moving direction.
        '''
        _, highest_utility, best_path = self._construct()
        available_directions = [each.dir_from_parent for each in self.root.children]
        available_dir_utility = np.array([self._descendantUtility(each) for each in self.root.children])
        for index, each in enumerate(available_directions):
            self.Q_value[self.dir_list.index(each)] = available_dir_utility[index]
        self.Q_value = np.array(self.Q_value)
        available_directions_index = [self.dir_list.index(each) for each in available_directions]
        # self.Q_value[available_directions_index] += 1.0 # avoid 0 utility
        # Add randomness and laziness
        Q_scale = scaleOfNumber(np.max(np.abs(self.Q_value)))
        # randomness = np.random.normal(loc=0, scale=0.1, size=len(available_directions_index)) * Q_scale
        randomness = np.random.uniform(low=0, high=0.1, size=len(available_directions_index)) * Q_scale
        self.Q_value[available_directions_index] += (self.randomness_coeff * randomness)
        if self.last_dir is not None and self.dir_list.index(self.last_dir) in available_directions_index:
            self.Q_value[self.dir_list.index(self.last_dir)] += (self.laziness_coeff * Q_scale)
        if return_Q:
            return best_path[0][1], self.Q_value
        else:
            return best_path[0][1]



# if __name__ == '__main__':
#     from Utils.FileUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath
#     from Utils.ComputationUtils import makeChoice
# 
#     # Read data
#     locs_df = readLocDistance("../../../Data/Constants/dij_distance_map.csv")
#     adjacent_data = readAdjacentMap("../../../Data/Constants/adjacent_map.csv")
#     adjacent_path = readAdjacentPath("../../../Data/Constants/dij_distance_map.csv")
#     reward_amount = readRewardAmount()
#     print("Finished reading auxiliary data!")
# 
#     # An example of data
#     cur_pos = (14, 27)
#     ghost_data = [(13, 27), (15, 27)]
#     ghost_status = [3, 3]
#     energizer_data = [(12, 27), (13, 27), (15, 27), (19, 27)]
#     bean_data = [(2, 5), (4, 5), (5, 5), (16, 5), (18, 5), (20, 5), (24, 5), (25, 5), (16, 6), (7, 7), (13, 7), (27, 7),
#                  (2, 8), (16, 8), (22, 8), (2, 9), (6, 9), (8, 9), (9, 9), (13, 9), (14, 9), (16, 9), (17, 9), (19, 9),
#                  (24, 9), (26, 9), (27, 9), (10, 10), (22, 10), (2, 11), (10, 11), (22, 11), (5, 12), (7, 12), (22, 12),
#                  (22, 13), (7, 14), (13, 14), (16, 14), (7, 15), (7, 17), (22, 17), (7, 19), (10, 23), (2, 24), (3, 24),
#                  (6, 24), (8, 24), (9, 24), (13, 24), (17, 24), (20, 24), (22, 24), (25, 24), (7, 25), (22, 25), (2, 26),
#                  (27, 27), (2, 28), (27, 28), (10, 29), (22, 29), (2, 30), (7, 30), (11, 30), (16, 30), (18, 30), (19, 30),
#                  (22, 30), (27, 30), (2, 32), (5, 33), (9, 33), (12, 33), (14, 33), (15, 33), (16, 33), (17, 33), (18, 33),
#                  (20, 33), (24, 33), (25, 33), (26, 33)]
#     reward_type = 5
#     fruit_pos = (3,30)
#     last_dir = None
# 
#     # Energizer agent
#     agent = EnergizerTree(
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
#         5,
#         0,
#         0,
#         0,
#         0,
#         reward_coeff=1.0, risk_coeff=0.0,
#         randomness_coeff=1.0, laziness_coeff=1.0
#     )
#     _, Q = agent.nextDir(return_Q=True)
#     choice = agent.dir_list[makeChoice(Q)]
#     print("Energizer Tree Choice : ", choice, Q)
