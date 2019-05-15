# High Dimensional Bayesian Optimization
This repository includes several implementations of high dimensional BO, mainly the projection methods. There are two main category of projection based BO methods, REMBO and HeSBO. There are also the implementations of HeSBO embedding method on both BLOSSOM and Cornell-MOE which are the state-of-the-art low dimensional BO methods. To use those algorithms, please go to the corresponding folder.

## Installing the requirements
The codes are written in python 3.6, so it is recommended to use this version of python to run the scripts. To install the requirements one can simply use this line:
```bash
pip3 install -r requirements.txt
```
## Running different BO methods
There are HeSBO and three different variants of REMBO implemented in this code. Three REMBO variants are called K<sub>Y</sub>, K<sub>X</sub>, and K<sub><img src="https://latex.codecogs.com/gif.latex?{\psi}" title="{\psi}" /></sub>. To run any of these algorithms, one should run this line:
```bash
python experiments.py [algorithm] [first_job_id] [last_job_id] [test_function] [num_of_steps] [low_dim] [high_dim] [num_of_initial_sample] [noise_variance] [REMBO_variant]
```
To determine the algorithm, use `REMBO` or `HeSBO` input for the python script. If REMBO algorithm is selected to be run, the REMBO variant must be determined by `X`, `Y`, or `psi` as the last argument. If none of them is picked, all of those variants will be run.