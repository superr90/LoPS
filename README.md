# Language of Problem Solving

**Quickstart**

The code execution order is 
1.BasicStrategy 
2.FeatureExtractor 
3.GrammarInduction
4.DrawImg

The first three items are data analysis codes, and each step is divided into two parts: human processing and monkey processing. 
The order of step within each item is indicated by the file number. 
We visulize and save all the figures in the paper in "DrawImg".

Additionally, we provide two examples of data files required to run the scripts. 

**Data Description**

| name | mean                                                         | description                                                  |
| --- | --- | --- |
| Unnamed: 0   | frame number |  |
| DayTrial | trial name | game-trial-day-month-year |
| Step | serial number in each trial |  |
| pacmanPos | the coordinates of pacman | tuple |
| ghost1Pos | the coordinates of ghost 1 | tuple |
| ghost2Pos | the coordinates of ghost 2 | tuple |
| ghost3Pos | the coordinates of ghost 3 | tuple |
| ghost4Pos | the coordinates of ghost 4 | tuple |
| ifscared1 | the mode of ghost 1 | 12 is a living ghost, 4 is a scared ghost, 5 is a flash scared |
| ifscared2 | the mode of ghost 2 |  |
| ifscared3 | the mode of ghost 3 |  |
| ifscared4 | the mode of ghost 4 |  |
| pacman_dir | the movement direction of pacman (actually the movement direction of the previous step) | nan means not moved |
| JoyStick | joystick direction | nan means the joystick has not been moved |
| Map | current map | string representation |
| beans | location of beans | multiple tuples in the list represent the position of each bean |
| energizers | elocation of energizers | multiple tuples in the list represent the position of each energizer |
| strategy_Q | strategy utility in four directions | the strategy can be global, local, evade_blinky, evade_clyde, evade_3, evade_4, approach, energizer, no_energizer |
| strategy_Q_norm | normalization utility of strategy in four directions | 1. The strategy can be global, local, evade_blinky, evade_clyde, evade_3, evade_4, approach, energizer, no_energizer; 2. If there are negative numbers, shift to the right first, let the minimum value be 0 and then divide by the maximum value |
| next_pacman_dir_fill | pacman’s current movement direction |  |
| available_dir | is the current movement direction feasible? | True or False |
| weight | the weights of the nine strategies at the current moment | a list of length 9 |
| contribution | normalized weights of nine strategies at the current moment | max-min normalization |
| predict_dir | current direction predicted based on weight | 1234 means up, down, left and right respectively. |
| is_correct | is the current direction prediction correct? |  |
| trial_context | the context that belongs to the current trial at the current moment | tuple represents the time before and after the current conext |
| eat_energizer | whether the energizer has been eaten at the current moment |  |
| eat_ghost | whether the ghost has been eaten at the current moment |  |
| is_stay | whether the current moment has not moved | True means not moved |
| is_vague | is the current strategy uncertain? | True indicates an uncertain strategy (accuracy is less than 0.5 using any other single strategy) |
| revise_weight | the revised strategy weight |  |
| revise_is_correct | is the current prediction correct after revising the weights? |  |
| strategy | current strategy | { “global”: 0, “local”: 1, “evade_blinky”: 2, “evade_clyde”: 3, “evade_3”: 4, “evade_4”: 5, “approach”: 6, “energizer”: 7, “no_energizer”: 8, “vague”: 9, “stay”: 10 } |
| gram | grammar at the current moment |  |
| gram_num | the sequence number of the strategy at the current moment in the current  grammar | counting starts from 0 |
| gramStart | whether the current moment is the starting moment of a new grammar | 1 means yes 0 means no |
| gramLen | the length of the current grammar |  |
