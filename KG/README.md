# HeSBO embedding on KG
To run the knowledge gradient based algorithm with HeSBO embedding, we first need to install the Corenell-MOE package on our machine. To do so, we refer the readers to the [Cornell-MOE github page][1]. After installing the package, one simply can use the below line to run the KG using the HeSBO embedding.
```bash
python moe_run.py [test_function] [KG] [num_of_steps] [low_dim] [high_dim] [num_of_initial_sample] [job_id] [noise_variance]
```

[1]: https://github.com/wujian16/Cornell-MOE