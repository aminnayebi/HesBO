# A Framework for Bayesian Optimization in Embedded Subspaces
Bayesian optimization (BO) has recently emerged as powerful method for the global optimization of expensive-to-evaluate black-box functions. This repository provides Python implementations of several algorithms that extend BO to problems with high input dimensions, including:

* The HesBO algorithm proposed by Munteanu, Nayebi, and Poloczek (ICML '19) (see below for the citation) combined with

	* The Knowledge Gradient algorithm of Cornell-MOE (Wu & Frazier NIPS'16; Wu, Poloczek, Wilson, and Frazier NIPS'17)
	
	* The BLOSSOM algorithm of McLeoad, Osborne, and Roberts (ICML '18)
	
	* Expected improvement, e.g., see Jones, Schonlau, and Welch '98)
		
* The REMBO method using 

	* the K<sub>X</sub>and K<sub>y</sub> kernels of Wang et al. '16 and 
	
	* K<sub>{~_\psi}</sub> of Binois, Ginsbourger and Roustant '15.  

## Installing the requirements
The codes are written in python 3.6, so it is recommended to use this version of python to run the scripts. To install the requirements one can simply use this line:
```bash
pip3 install -r requirements.txt
```
## Running different BO methods
There are HeSBO and three different variants of REMBO implemented in this code. Three REMBO variants are called K<sub>Y</sub>, K<sub>X</sub>, and K<sub><img src="https://latex.codecogs.com/gif.latex?{\psi}" title="{\psi}" /></sub>. To run any of these algorithms, one should run this line:

### An example running HesBO-EI on a benchmark
Run
```bash
(VIRT_ENV) $ python run_misoKG.py miso_rb_benchmark_mkg 0 0
```
for the Rosenbrock function proposed by Lam, Allaire, and Willcox (2015), or 
```bash
(VIRT_ENV) $ python run_misoKG.py miso_rb_benchmark_mkg 1 0
```
for the noisy variant proposed in the MISO paper.

The results are stored in a pickled dictionary in the current working directory. The filename is output when the program starts.
Note that the last parameter is a nonnegative integer that is used for the filename, e.g., when running multiple replications.

```bash
python experiments.py [algorithm] [first_job_id] [last_job_id] [test_function] [num_of_steps] [low_dim] [high_dim] [num_of_initial_sample] [noise_variance] [REMBO_variant]
```
To determine the algorithm, use `REMBO` or `HeSBO` input for the python script. If REMBO algorithm is selected to be run, the REMBO variant must be determined by `X`, `Y`, or `psi` as the last argument. If none of those variants is picked, all of those variants will be run.

## Citation
```bash
@inproceedings{HesBO19,
  author    = {Alex Munteanu and
               Amin Nayebi and
			   Matthias Poloczek},
  title     = {A Framework for Bayesian Optimization in Embedded Subspaces},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning, {(ICML)}},
  year      = {2019},
  note={Accepted for publication}
}
```
