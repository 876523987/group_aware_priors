# Mind the GAP: Improving Robustness to Subpopulation Shifts with Group-Aware Priors


## Running Experiments


### Running Code with JSON Configs

To run the code using any of the JSON configs under `configs`, execute the following line (replacing CONFIG_NAME, CONFIG_ID, and SEED) from the repository root:

```
python trainer_nn.py --config configs/CONFIG_NAME.json --config_id CONFIG_ID --seed SEED
```

NB: When using the configs, you need to modify the `--cwd` arg.

```
python trainer_nn.py --config configs/NAME_OF_CONFIG.json --config_id CONFIG_ID --seed 0
```


## Environment Setup


### Installing JAX

First, set up the conda environment using the conda environment `.yml` files in the repository root. To install `jax` and `jaxlib`, use
```
pip install "jax[cuda11_cudnn86]==0.4.4" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Installing PyTorch (CPU)

We recommend install PyTorch for CPU-only to make sure that it does not interfere with JAX's memory allocation. To install the right versions of `pytorch` and `torchvision`, use

```
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cpu
```

## Troubleshooting

If you encounter an out-of-memory error, you may have to adjust the amount of pre-allocated memory used by jax. This can be done, for example, by setting

```
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.9"
```

Note that this is only one of many reasons why you may encounter an OOM error.
