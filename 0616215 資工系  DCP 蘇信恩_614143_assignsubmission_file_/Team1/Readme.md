# Readme

## Dependency
matplotlib, gym, pickle, numpy, scipy

## Taxi-v2
`python3 taxi.py`

if you want different parameter of different softmax function, you have to modify code in main() manually, examples are presented in main() below exit(). Simply by modifying parameters on line 289 can achieve different testing scenarios. 

After running the program, you can call graph.py to graph out the result.

### Guide
#### Policy 
To test different policy sampler, modify parameter `action_selector` to either `eps_greedy`, `mm_sample` or `softmax_sample`.
#### Parameter Range
The 1~3 parameter of function `test_range` specifies the parameter range to be tested, for example: `test_range(1, 10, 0.5, ...)` for Mellowmax will return mean rewards of gamma ranging from 1 to 10, with step size 0.5, 

### Graphing
After running taxi.py, a file containing mean returns will be created, filename is specified in `test_range` by using `test_range(..., filename='filename')`

With the filename, modify the desired filename on line 5 in `graph.py` and run `python3 graph.py`, this will create the graph. The graph will show the mean returns of different parameters.

## Lunar
### Simulation
Simply run `./run.sh`

This command will train the agent with Boltzmann softmax and Mellowmax softmax with beta and gamma ranging from 1 ~ 10, with step size 0.5. And the result will be output to the respective files, for details please refer to `run.sh`, it's short and self descriptive. You will see that the shell script basically calls `reinforce.py` in loops for different parameters, detail discussion of these parameter will be provided below.


## Graphing
The folloing commands with take the result of `./run.sh` and graph it out, make sure all files are in the same folder.

`python3 graph_rewards.py` & `python3 graph_rewards.py --mm `

`python3 graph_runtime.py` & `python3 graph_runtime.py --mm`

to obtain different graphs.
The graph y axis will be either the reward or the running time, and the x axis will be the value of different parameters (omega or beta).


## Customize Parameters
`python3 reinforce.py` can be combined with different parameters, we can look into `run.sh` to see some example.

- `--mm` determines the softmax function to use, if enabled, Mellowmax softmax is used.
- `--omega {value}` , if `--mm` is used, we set the omega to the *value*.
- `--beta {value}`, if `--mm` is not used, we set the beta of softmax function to *value*.
- `--episode {n_episode}`, set the max episode of the program to *n_episode*.



