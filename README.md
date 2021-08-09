# Into the Unknown

This repository contains the implementation and data used in the paper [Into the Unknown: Active Monitoring of Neural Networks](https://arxiv.org/pdf/2009.06429), published at [RV 2021](https://uva-mcps-lab.github.io/RV21/).
To cite the paper, use:

```
@inproceedings{intotheunknown21,
  author    = {Anna Lukina and
               Christian Schilling and
               Thomas A. Henzinger},
  title     = {Into the unknown: Active monitoring of neural networks},
  booktitle = {RV},
  year      = {2021}
}
```

# Installation

We use Python 3.6 but other Python versions may work as well.
The package requirements that need to be installed are found in the file `requirements.txt`.

Since the datasets are large and have mostly been used in our previous work, we do not include most of them here.
You need to manually download them (see the links below) and extract them to the `data` folder of this repository.

Modify the file called `paths.txt` in the base folder, which contains two lines that are the paths to the model and dataset folders:

```
.../models/
.../data/
```

Here replace the `...` with the absolute path to your clone of the repository.

## Links to dataset files

- [`MNIST`](https://github.com/VeriXAI/Outside-the-Box/tree/master/data/MNIST)
- [`Fashion MNIST`](https://github.com/VeriXAI/Outside-the-Box/tree/master/data/Fashion_MNIST)
- [`CIFAR-10`](https://github.com/VeriXAI/Outside-the-Box/tree/master/data/cifar-10-python/cifar-10-batches-py)
- [`GTSRB`](https://github.com/VeriXAI/Outside-the-Box/tree/master/data/GTSRB) (You need to manually extract the file `train.zip` because the content is too large for Github.)
- `EMNIST`: This dataset is already included in the repository.


# Recreation of the results

Below we describe how to obtain the results shown in the paper.

## Models

The repository contains the pretrained models used in the evaluation.
The models have been trained using the scripts `run/train_INSTANCE.py` where `INSTANCE` is the name of the model/data combination.

## Evaluation

The scripts to reproduce the figures and tables of the paper are found in the folder `run/`:

- `run_experiments_online.py` (This script runs all experiments, which can also be run individually by modifying the script accordingly.)
- `plot_experiments_online.py` (This script creates all plots and requires that all results from the previous script have been obtained.)

Intermediate results of the experiments are stored in `.csv` files and the final plots are stored as `.pdf` files in the `run/` folder.
