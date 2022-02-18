
## Table of Contents

- [Background](#background)
- [Environment](#environment)
- [Configuration](#configuration)
- [Usage](#usage)
	- [Usage Example 1](#usage-example-1)
	- [Usage Example 2](#usage-example-2)
	- [Usage Example 3](#usage-example-3)
- [Results](#results)

## Background
This is the program of UCT, parallel UCT and Genetic search.

## Environment

Operating system: Linux  
Python: Python3.6, Python 3.7 or Python 3.8  
Package: ngspice, networkx, numpy and matplotlib
```sh
$ sudo apt install ngspice
$ pip3 install networkx numpy matplotlib
```

## Configuration

**max_episode_length**: The max allowed step for simulator to take action  
**deterministic**: True if when a state take an action, the next state is deterministic. 
False if the next state can be one of several different states  
**ucb_scalar**: the ucb scalar, currently it should be set in range 5 to 20   
**gamma**: The parameter we multiplied when we do the back propagation in UCT.
In this application we set it as 1.  
**leaf_value**: The reward we assign to the leaf node in a Monte Carlo tree.
In this application we set as 0.  
**end_episode_value**: The reward we assign to the terminal state. In this application
 we set as 0.  
**algorithm**: The name of the algorithm, if we want to test UCT, it should be set as UCF. If we want
 to test Genetic Search, it should be set as GS  
**root_parallel**: True if we want to run parallel UCT, False if we want to run serial UCT
**act_selection**: The way we find the action through several different trees in root parallelization.
The best choice is Pmerge now.  
**tree_num**: The number of the tree for root parallelization.  
**thread_num**: The number of the thread for root parallelization. Currently it should be set the same
as tree_num  
If we want to test the UCT, we must set tree_num and thread_num as 1  
**position_num**: Initial state number. Currently it can only be 1.  
**game_num**: The time we want to generate topologies using the algorithms.  
**dep_start**: The start number of the depth list we want to test  
**dep_end**: The end number of the depth list we want to test  
**dep_step_len**: The step of the depth list  
The depth can only be -1, which representing that we continuously extend
the topology until we meet the terminal state in the simulation process of UCT.
So we set dep_start as -1, dep_end as 0 and dep_step_len as 1(without blank)  
**traj_start**: The start number of the trajectory list we want to test  
**traj_end**: The end number of the trajectory list we want to test  
**traj_step_len**: The step of the trajectory list  
If we just what to test UCT when trajectory is 100, we can set traj_start as 100, 
traj_end as 101 and traj_step_len as 1.  
**sys_os**: The operating system. Currently it should be set as linux  
**output**: False if we do not need to print out several addition information.  
**num_component**: The number of component. Currently it should be set as 6  
**freq, vin, D**: Fixed parameters set as 200000, 50 and 0.48  
**mutate_num**: Number of child in every mutation generation  
**mutate_generation**: Total number of mutation in genetic search  

## Usage
There are two steps we need to follow to run the program: first we need to change file "config" to set the parameters.
Then we need to run "main.py". Usually the running time is about 10 to 30 minutes, so we recommend you to run the program at the background:
```sh
$ nohup python3 -u main.py > out.log 2<%1 $
```
The printed message will be saved in "out.log".

### Usage Example: UCT
If we want to test the UCT with trajectory as 512, we can set the config as:  
deterministic=True  
ucb_scalar=10  
gamma=1  
***algorithm***=UCF  
***root_parallel***=False  
tree_num=1  
thread_num=1  
dep_start=-1  
dep_end=0  
dep_step_len=1  
***traj_start***=512  
***traj_end***=513  
traj_step_len=1  
sys_os=linux  
output=False  

Then run:
```sh
$ nohup python3 -u main.py > out.log 2<%1 $
```

### Usage Example: Parallel UCT
If we want to test the root-Parallel UCT with trajectory as 512 and tree number as 4,
we can set the config as:  
deterministic=True  
ucb_scalar=10  
gamma=1  
algorithm=UCF  
***root_parallel***=True  
***act_selection***=Pmerge  
***tree_num***=4  
***thread_num***=4  
dep_start=-1  
dep_end=0  
dep_step_len=1  
traj_start=512  
traj_end=513  
traj_step_len=1  
sys_os=linux  
output=False  

Then run:
```sh
$ nohup python3 -u main.py > out.log 2<%1 $
```

### Usage Example: Genetic Search
If we want to test Genetic search with mutating 100 times, we can set config as:  
***algorithm***=GS  
mutate_num=5  
***mutate_generation***=100  

Then run:
```sh
$ nohup python3 -u main.py > out.log 2<%1 $
```


### Results
There are two kinds of results, the topology figures and the result saved in txt file.   
The figures are saved in folder "figures", the txt result are saved in folder "Results".
The last two lines printed by the program (saved in out.log) are the folder name of the current program's
topology figures and result, both of them will be the program starting time.  
In the figures' folder, for UCT algorithms, it will store the topology generation process(one action by one action).
For genetic search, it will store the initialized state and the finally generated topology.
The last few lines of the txt file saved finally efficiency, printed topology, total running time and other information.  
We use the date and time as the name of folders and files. For example, the last two lines of program could be:
```sh
figures are saved in:figures/2020-11-13-17-42-08/
outputs are saved in:Results/mutitest-2020-11-13-17-42-08.txt
``` 
Then we can open "figures/2020-11-13-17-42-08/" to check the visualized topology generation process and 
"Results/mutitest-2020-11-13-17-42-08.txt" to check outputted information.  