
# Macbook Pro M1 setup for Tensorflow
> How to setup python and DL libs on your new macbook pro

The new apple silicon is pretty amazing, it is fast and very power efficient, but does it work for data science? The first thing you need is to install python. To do so, refer to this [video](https://www.youtube.com/watch?v=w2qlou7n7MA&list=RDCMUCR1-GEpyOPzT2AO4D_eifdw&index=1) from the amazing Jeff Heaton.

## Install

- Install apple developer tools: Just git clone something, and the install will be triggered.
You have 2 options:
  - Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html): This is a small and light version of anaconda that only contains the command line iterface and no default packages.
  - Install [miniforge](https://github.com/conda-forge/miniforge): This is a fork of miniconda that has the default channel `conda-forge` (instead of conda default), this is a good thing as almost every package you want to use is in this channel.
> Note: As your ML packages are on `conda-forge` if you install miniconda3 you will always need to append the flag `-c conda-forge` to find/install your libraries.
- (Optional) Install [Mamba](https://github.com/mamba-org/mamba) on top of your miniconda/miniforge install, it makes everything faster.

> Note: Don't forget to choose the ARM M1 binaries

## Environment setup

You can now create you environment and start working!

```bash
#this will create an environment called wandb with python and all these pkgs
conda create -c conda-forge --name=wandb python wandb pandas numpy matplotlib jupyterlab

# or with mamba
mamba create -c conda-forge --name=wandb python wandb pandas numpy matplotlib jupyterlab

#if you use miniforge, you can skip the channel flag 
conda create --name=wandb python wandb pandas numpy matplotlib jupyterlab
```

> Note: To work inside the environment, you will need to call `conda activate env_name`.

## Apple M1 Tensorflow
Apple has made binaries for tensorflow 2 that supports the GPU inside the M1 processor. This makes training way faster than CPU. You need can grab tensorflow install intruction from the apple website [here](https://developer.apple.com/metal/tensorflow-plugin/) or use the environment files that are provided here. ([tf_apple.yml](tf_apple.yml)). I also provide a [linux env file](tf_linux.yml) in case you want to try.

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
python scripts/keras_cvp.py --hw "your_gpu_name" --repeat 3 --trainable
```
This will run the training script 3 times with all parameters trainable (not finetuning)

> Note: I also provide a [pytorch training script](scripts/pytorch_wandb.py), but you will need to install pytorch first. It may be useful once pytroch adds GPU support.

## NGC Docker

We can also run the benchmarks on linux using nvidia docker [containers](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#runcont):: 

- Install `docker` with [following official nvidia documentation](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html). Once the installation of docker and nvidia support is complete, you can run the tensorflow container. You cant test that your setup works correctly by running the dummy cuda container and yo should see the `nvidia-smi` output:

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

Now with the tensorflow container:

- Pull the container:

```bash
sudo docker pull nvcr.io/nvidia/tensorflow:21.11-tf2-py3
```
once the download has finished, you can run the container with:

- Run the containter, replace the `path_to_folder` with the path to this repository. This will link this folder inside the docker container to `/code` path.

```bash
sudo docker run --gpus all -it --rm -v path_to_folder/apple_m1_pro_python:/code nvcr.io/nvidia/tensorflow:21.11-tf2-py3
```

once inside the container install the missing libraries:

```bash
$ pip install wandb fastcore tensorflow_datasets
```

- And finally run the benchmark, replace `your_gpu_name` with your graphcis card name: `RTX3070m`, `A100`, etc... With modern Nvidia hardware (after RTX cards) you should enable the `--fp16` flag to use the tensor cores on your GPU.

```bash
python scripts/keras_cvp.py --hw "your_gpu_name" --trainable --fp16
```

> Note: You may need `sudo` to run docker.

> Note2: Using this method is substantially faster than installing the python libs one by one, please use the NV containers.