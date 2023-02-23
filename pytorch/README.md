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
$ conda create --name="metal" "python<3.11"


# activate the environment
$ conda activate metal

# with conda
conda install pytorch torchvision -c pytorch

# or with pip
$ pip install torch torchvision


# install dependencies of this training script 😎
$ pip install wandb tqdm transformers datasets
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
