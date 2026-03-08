# Template base code for pytorch

This repository contains a template base code for a complete pytorch pipeline.

This is a template because it works on fake data but aims to illustrate some pythonic syntax allowing the pipeline to be modular.

More specifically, this template base code aims to target :

- modularity : allowing to change models/optimizers/ .. and their hyperparameters easily
- reproducibility : saving the commit id, ensuring every run saves its assets in a different directory, recording a summary card for every experiment, building a virtual environnement

For the last point, if you ever got a good model as an orphane pytorch tensor whithout being able to remember in which conditions, with which parameters and so on you got, you see what I mean. 

## Usage

### Local experimentation

For a local experimentation, you start by setting up the environment :

```
python3 -m virtualenv venv
source venv/bin/activate
python -m pip install .
```

Then you can run a training, by editing the yaml file, then 

```
python -m torchtmpl.main config.yml train
```

And for testing (**not yet implemented**)

```
python main.py path/to/your/run test
```

### Cluster experimentation (**not yet implemented**)

For running the code on a cluster, we provide an example script for starting an experimentation on a SLURM based cluster.

The script we provide is dedicated to a use on our clusters and you may need to adapt it to your setting. 

Then running the simulation can be as simple as :

```
python3 submit.py
```

## Testing the functions

Every module/script is equiped with some test functions. Although these are not unitary tests per se, they nonetheless illustrate how to test the provided functions.

For example, you can call :


```
python3 -m virtualenv venv
source venv/bin/activate
python -m pip install .
python -m torchtmpl.models
```

and this will call the test functions in the `torchtmpl/models/__main__.py` script.

## 
Launching a job. To launch a job on the dgx use the submit-slurm-dgx-sweep.py function 

you must call uv run python -u submit-slurm-dgx-sweep.py config/file/.yaml nruns function/to/be/called 

The -u flag is good practice to get accurate logs (the stderr and stdout buffers are dumped and logs are seen in real time which helps to locate the erros)

### Explicative video

You will find a video (in French) explaining and detailing our project what we did and why did it following this link: 
https://www.youtube.com/watch?v=1JTjhFDgrtw