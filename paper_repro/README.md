## Installation
Needs
`pip install torch torchvision opacus dm-haiku`

For some reason JAX can't find the CUDA 11.0 version of 1.0.71 (though it definitely exists?) so we have to install it like this for the A100s
`pip install --upgrade jax jaxlib==1.0.70+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html`

## Running
Our benchmark will run the model (`--model` can be `cifar10` or `resnet18`) with batch size of 16, 32, 64, 128, 256 (`--batch_size` can be passed different numbers to adjust this) for 5 epochs (can be changed with `--epochs`). 

To run opacus on the A100s:
`gpurun python opacusdp.py --model {resnet18|cifar10}`

To run jax on the A100s:
`XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 gpurun python jaxdp.py -model resnet18`

We need the flag since the AWS cluster has a different driver than what's installed on the A100s. However, this flag means that the first epoch will be useless since it's not able to compile in parallel

**NOTE:** in order to get these to work with the same data loader, every batch load for JAX needs to copy the tensor to a numpy tensor (I tried to make this a dataloader transform but it didn't work). We can make this one use the TF data loader to get even better performance
