# HeSBO embedding on BLOSSOM
One first needs to install the BLOSSOM package on their machine. To do so, we refer the readers to the [BLOSSOM github page][1]. After installing the package, one simply can use the below line to run the BLOSSOM using the HeSBO embedding.
```bash
python blossom_run.py [test_function] [num_of_steps] [low_dim] [high_dim] [num_of_initial_sample] [job_id] [noise_variance]
```
Please note to change the line 9 in the blossom_run.py file and give the path of the directory in which the results would be collected.

[1]: https://github.com/markm541374/gpbo
