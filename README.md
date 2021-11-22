# Macbook Pro M1 setup
> How to setup python and DL libs on your new macbook pro

The new apple silicon is pretty amazing, it is fast and very power efficient, but does it work for data science? The first thing you need is to install python. To do so, refer to this [video](https://www.youtube.com/watch?v=w2qlou7n7MA&list=RDCMUCR1-GEpyOPzT2AO4D_eifdw&index=1) from the amazing Jeff Heaton.

## Install

- Install apple developer tools: Just git clone something, and the install will be triggered.
- Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html) or [miniforge](https://github.com/conda-forge/miniforge) and choose the ARM M1 binaries!!
- (Optional) Install [Mamba](https://github.com/mamba-org/mamba) on top of your miniconda/miniforge install, it makes everything faster.

## Work

You can now create you environment and start working!

```bash
#this will create an environment called wandb with python and all these pkgs
conda create --name=wandb python wandb pandas numpy matplotlib jupyterlab

# or with mamba
mamba create --name=wandb python wandb pandas numpy matplotlib jupyterlab
```

## Apple M1 Tensorflow
Apple has made binaries for tensorflow 2 that supports the GPU inside the M1 processor. This makes training way faster than CPU. You need can grab them [here](https://developer.apple.com/metal/tensorflow-plugin/) or use the environment files that are provided here. ([tf_apple.yml](tf_apple.yml)).

## Pytorch
Pytorch works straight out of the box, but only on CPU. There is a plan to release GPU support in the next months, follow [Soumith Chintala](https://twitter.com/soumithchintala) for up to date info on this.

## Benchmarks
You can check some runs on this [report](http://wandb.me/m1pro). To run the benchmark yourself in your new macbook pro:
- setup the enviroment:
```bash
conda env create --file=tf_apple.yml
conda activate tf
```
- You will need a [wandb][wandb.ai] account, follow instructions once the script is launched.
- Run the training:

```bash
python scripts/keras_pets.py
```
This script is based on an official keras example by fchollet.

I also provide a [pytorch training script](scripts/pytorch_wandb.py), but you will need to install pytorch first.