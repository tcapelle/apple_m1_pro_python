
# Macbook Pro M1 setup for Tensorflow
> How to setup python and DL libs on your new macbook pro

The new apple silicon is pretty amazing, it is fast and very power efficient, but does it work for data science? The first thing you need is to install python. ~~To do so, refer to this [video](https://www.youtube.com/watch?v=w2qlou7n7MA&list=RDCMUCR1-GEpyOPzT2AO4D_eifdw&index=1) from the amazing Jeff Heaton.~~

## Install

- Install apple developer tools: Just git clone something, and the install will be triggered.
- Install [miniforge](https://github.com/conda-forge/miniforge): This is a fork of miniconda that has the default channel `conda-forge` (instead of conda default), this is a good thing as almost every package you want to use is in this channel.
> Note: Don't forget to choose the ARM M1 binaries

## Environment setup (TLDR)

You can now create you environment and start working!

```bash
$ curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
$ sh Miniforge3-MacOSX-arm64.sh
# follow instructions and init conda at the end, restart your terminal
# this will create an environment called `tf` with python
$ conda create --name=tf "python<3.11"

# activate the environment
$ conda activate tf
$ conda install -c apple tensorflow-deps
$ pip install tensorflow-macos tensorflow-metal 

# install benchmark dependencies
$ pip install wandb transformers datasets tqdm scikit-learn
```

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

## Running the Benchmark on Oxford PETS Resnet50 Training
In your terminal run:

```bash
$ python train_pets.py --gpu_name="M1Pro GPU 16 Cores" #replace with your GPU name
```

- Pass the `--gpu_name` flag to group the runs, I am not able to detect this automatically on Apple.
- To run on cpu pass `--device="cpu"` or for CUDA `--device="cuda"` (you need a linux PC with an Nvidia GPU)
- You can also pass other params, and play with different `batch_size` and `model_name`.


## Bert Benchmark

In your terminal run:

```bash
$ python train_bert.py --gpu_name="M1Pro GPU 16 Cores" #replace with your GPU name
```


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
$ pip install wandb transformers datasets tqdm scikit-learn
```

- And finally run the benchmark, replace `your_gpu_name` with your graphcis card name: `RTX3070m`, `A100`, etc... With modern Nvidia hardware (after RTX cards) you should enable the `--fp16` flag to use the tensor cores on your GPU.

```bash
python train_pets.py --gpu_name="My_fancyNvidia_GPU"
```

> Note: You may need `sudo` to run docker.

> Note2: Using this method is substantially faster than installing the python libs one by one, please use the NV containers.