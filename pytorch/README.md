# Running PyTorch on the Apple M1's GPU

We will run PyTorch training on the M1 GPU

## Setup
You will need Python installed, the preferred way is using MiniForge

```bash
# The version of Anaconda may be different depending on when you are installing`
$ curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
$ sh Miniforge3-MacOSX-arm64.sh
# and follow the prompts. The defaults are generally good.
```

Then, create an environment to use Python and PyTorch:

```bash
$ conda create --name="metal" python


# activate the environment
$ conda activate metal


# or with pip
$ pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu


# install dependencies of this training script 😎
$ pip install wandb tqdm
```

For more details on installing Python check [this report](https://wandb.ai/tcapelle/apple_m1_pro/reports/Deep-Learning-on-the-M1-Pro-with-Apple-Silicon---VmlldzoxMjQ0NjY3).

> The `setup.sh` script takes care of installing all the above 

```bash
$ sh setup.sh
```

## Verify your Install
Run the following in python

```python
import torch


torch.__version__
>>> '1.12.0.dev20220518'


torch.tensor([1,2,3], device="mps")
```

## Running the Benchmark
In your terminal run:

```bash
$ python train_pets.py --device="mps" --gpu_name="M1Pro GPU 16 Cores"
```

- Pass the `--gpu_name` flag to group the runs, I am not able to detect this automatically on Apple.
- To run on cpu pass `--device="cpu"` or for CUDA `--device="cuda"` (you need a linux PC with an Nvidia GPU)
- You can also pass other params, and play with different `batch_size` and `model_name`.




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