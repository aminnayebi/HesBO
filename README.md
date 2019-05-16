# A Framework for Bayesian Optimization in Embedded Subspaces

## What is high-dimensional Bayesian optimization?
Bayesian optimization (BO) has recently emerged as powerful method for the global optimization of expensive-to-evaluate black-box functions. However, these methods are usually limited to about 15 input parameters (levers). 
In the paper "A Framework for Bayesian Optimization in Embedded Subspaces" (to appear at ICML'19), [Munteanu](https://www.statistik.tu-dortmund.de/munteanu.html "Alexander Munteanu"), Nayebi, and [Poloczek](http://www.sie.arizona.edu/poloczek "Matthias Poloczek") propose a non-adaptive probabilistic subspace embedding that can be combined with many BO algorithms to enable them to higher dimensional problems.

This repository provides Python implementations of several algorithms that extend BO to problems with high input dimensions:

* The HeSBO algorithm proposed by Munteanu, Nayebi, and Poloczek (ICML '19) (see below for the citation) combined with

	* The Knowledge Gradient (KG) algorithm of [Cornell-MOE](https://github.com/wujian16/Cornell-MOE "Cornell-MOE") (Wu & Frazier NIPS'16; Wu, Poloczek, Wilson, and Frazier NIPS'17)
	
	* The [BLOSSOM algorithm](https://github.com/markm541374/gpbo "BLOSSOM") of McLeod, Osborne, and Roberts (ICML '18)
	
	* Expected improvement, e.g., see Jones, Schonlau, and Welch (JGO '98)
		
* The REMBO method using 

	* the K<sub>X</sub>and K<sub>y</sub> kernels of Wang et al. (JMLR '18) and 
	
	* the K<sub><img src="https://latex.codecogs.com/gif.latex?{~_\psi}" title="{\psi}" /></sub> kernel of Binois, Ginsbourger and Roustant (LION '15).  

## Installing the requirements
The codes are written in python 3.6, so it is recommended to use this version of python to run the scripts. To install the requirements one can simply use this line:
```bash
pip3 install -r requirements.txt
```
## Running different BO methods
There are HeSBO and three different variants of REMBO implemented in this code. Three REMBO variants are called K<sub>y</sub>, K<sub>X</sub>, and K<sub><img src="https://latex.codecogs.com/gif.latex?{~_\psi}" title="{\psi}" /></sub> . These algorithms can be run as follows.

```bash
python experiments.py [algorithm] [first_job_id] [last_job_id] [test_function] [num_of_steps] [low_dim] [high_dim] [num_of_initial_sample] [noise_variance] [REMBO_variant]
```
To determine the algorithm, use `REMBO` or `HeSBO` input for the python script. If REMBO algorithm is selected to be run, the REMBO variant must be determined by `X`, `Y`, or `psi` as the last argument. If none of those variants is picked, all of those variants will be run.
Here is an example of running HeSBO-EI on 100 dim noise-free Branin with 4 low dimensions:
```bash
python experiments.py HeSBO 1 1 Branin 80 4 100 10 0
```
To collect the output data, you must have a folder named "results". Here is a plot for running REMBO-K<sub><img src="https://latex.codecogs.com/gif.latex?{~_\psi}" title="{\psi}" /></sub> and HeSBO-EI on the Branin function.
<center><img src="https://github.com/aminnayebi/HesBO/blob/master/Branin_D100_d4.jpg" height="350" width="350"></center>

## Citation
```bash
@inproceedings{HeSBO19,
  author    = {Alex Munteanu and
               Amin Nayebi and
			   Matthias Poloczek},
  title     = {A Framework for Bayesian Optimization in Embedded Subspaces},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning, {(ICML)}},
  year      = {2019},
  note={Accepted for publication. The code is available at https://github.com/aminnayebi/HesBO.}
}
```
