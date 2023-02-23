# Apple Silicon DL benchmarks

Currently we have PyTorch and Tensorflow that have Metal backend.

## Results 
Varied results across frameworks:
- [Apple M1Pro Pytorch Training Results](https://wandb.me/pytorch_m1)
- [Apple M1Pro Tensorflow Training Results](https://wandb.me/m1pro)

### Tensorflow Resnet50:
![tf_resnet_50results.png](images/tf_resnet50_results.png)

### PyTorch Resnet50:
- Difference between CPU and GPU
![gpu_vs_cpu.png](images/pt_gpu_vs_cpu.png)
- Comparing with Nvidia
![samples_sec.png](images/pt_samples_sec.png)

### PyTorch Bert
- Running a Bert from Huggingface
![pt_bert.png](images/pt_bert.png)


## Pytorch
We have official PyTorch support! check [pytorch](pytorch) folder to start running your benchmarks



## Tensorflow
You can run tensorflow benchmarks by going to the [tensorflow](tensorflow) folder.
