## Standard libraries
import os
import numpy as np
from PIL import Image, ImageFile
import pickle
from typing import Any
from collections import defaultdict
import time
import tree
import random as random_py
import numpy.random as random_np
import functools
from functools import partial
from copy import copy, deepcopy
from typing import (Any, Callable, Iterable, Optional, Tuple, Union, Dict, List)
import warnings
import h5py
import argparse
from tqdm.auto import tqdm
import json
from pprint import pprint
import re
import pandas as pd

## Plotting
import matplotlib
# matplotlib.use('TKAgg') # TODO: Tmp changes

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
# %matplotlib inline
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg', 'pdf') # For export
rcParams['lines.linewidth'] = 2.0
# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman'] # need to have latex installed for this to work
rcParams['text.usetex'] = True
plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
import seaborn as sns
sns.reset_orig()

## To run JAX on TPU in Google Colab, uncomment the two lines below
# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

## JAX
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import jit
from jax.config import config
# config.update("jax_debug_nans", True)
# config.update('jax_platform_name', 'cpu')

## Flax
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import freeze, unfreeze
from flax.linen.initializers import lecun_normal

## JAX addons
import optax
import distrax
import neural_tangents as nt
import flaxmodels as fm
from flaxmodels.resnet import ops
from flaxmodels import utils

## Tensorflow
import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
tfd = tfp.distributions

## PyTorch
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST, MNIST, KMNIST, ImageNet, CelebA
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import datasets as sklearn_datasets
from sklearn.preprocessing import StandardScaler
import wandb

from pathlib import Path
from timm.data import create_dataset
from torch.utils.data import Dataset, random_split

## Huggingface Transformers
#from transformers import FlaxBertForSequenceClassification, BertConfig, AutoTokenizer

## Convert from CxHxW to HxWxC for Flax.
chw2hwc_fn = lambda img: img.permute(1, 2, 0)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='fmnist')
parser.add_argument('--group_dro', action="store_true", default=False)
parser.add_argument('--data_balancing', action="store_true", default=False)
parser.add_argument('--gdro_step_size', type=float, default=0.01)
parser.add_argument('--quick_eval', action="store_true", default=False)
parser.add_argument('--sensitive_attribute', type=str, default='sex') # sex, race, age
parser.add_argument('--fairness_eval', action="store_true", default=False)
parser.add_argument('--fairness_train', action="store_true", default=False)
parser.add_argument("--val_train_frac", type=float, default=1)
parser.add_argument("--dataset_group_scale", type=float, default=4)
parser.add_argument('--validation_training', action="store_true", default=False)
parser.add_argument('--final_layer_retraining', action="store_true", default=False)
parser.add_argument('--empirical_fairness_prior_scale', type=float, default=1)
parser.add_argument('--llm_dropout', type=float, default=0.1)
parser.add_argument('--prediction_type', type=str, default='classification')
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--context_batch_size", type=int, default=1)
parser.add_argument("--training_dataset_size", type=int, default=0)
parser.add_argument("--context_dataset_size", type=int, default=10000)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--lr_schedule_name", type=str, default='linear')
parser.add_argument("--learning_rate_scale_logvar", type=float, default=1)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument('--optimizer_name', type=str, default='sgd')
parser.add_argument('--model_name', type=str, default='ResNet18')
parser.add_argument('--method', type=str, default='fsmap')
parser.add_argument('--reg_type', type=str, default='function_prior')
parser.add_argument('--forward_points', type=str, default='train')
parser.add_argument('--reg_points', type=str, default='train')
parser.add_argument('--context_points', type=str, default='train')
parser.add_argument("--context_transform", action="store_true", default=False)
parser.add_argument('--ood_points', type=str, default='')
parser.add_argument("--mc_samples_llk", type=int, default=1)
parser.add_argument("--mc_samples_reg", type=int, default=1)
parser.add_argument("--mc_samples_eval", type=int, default=1)
parser.add_argument("--reg_scale", type=float, default=1)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--prior_mean", type=float, default=0)
parser.add_argument("--prior_var", type=float, default=0)
parser.add_argument("--prior_likelihood_scale", type=float, default=1)
parser.add_argument("--prior_likelihood_f_scale", type=float, default=1)
parser.add_argument("--prior_likelihood_cov_scale", type=float, default=0)
parser.add_argument("--prior_likelihood_cov_diag", type=float, default=0)
parser.add_argument("--prior_likelihood_mean", type=float, default=0)
parser.add_argument("--prior_likelihood_normalize_feature", action="store_true", default=False)
parser.add_argument("--likelihood_scale", type=float, default=1)
parser.add_argument("--output_var", action="store_true", default=False)
parser.add_argument("--rho_sam", type=float, default=0)
parser.add_argument("--rho_adversarial", type=float, default=0)
parser.add_argument("--dropout_rate_sam", type=float, default=0)
parser.add_argument("--prior_params_var", type=float, default=1)
parser.add_argument("--init_logvar", type=float, default=-50)
parser.add_argument("--init_final_layer_weights_logvar", type=float, default=-50)
parser.add_argument("--init_final_layer_bias_logvar", type=float, default=-50)
parser.add_argument("--prior_feature_logvar", type=float, default=-50)
parser.add_argument("--prior_precision", type=float, default=0)
parser.add_argument("--pretrained_prior", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--linearize", action="store_true", default=False)
parser.add_argument("--evaluate", action="store_true", default=False)
parser.add_argument("--full_eval", action="store_true", default=False)
parser.add_argument("--restore_checkpoint", action="store_true", default=False)
parser.add_argument('--checkpoint_dir', type=str, default='')
parser.add_argument("--final_layer_random_init", action="store_true", default=False)
parser.add_argument('--batch_stats_init_epochs', type=int, default=0)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--debug_print", action="store_true", default=False)
parser.add_argument("--debug_psd", action="store_true", default=False)
parser.add_argument("--log_frequency", type=int, default=2)
parser.add_argument("--log_frequency_steps", type=int, default=0)
parser.add_argument("--save_to_wandb", action="store_true", default=False)
parser.add_argument('--wandb_project', type=str, default='xxx')
parser.add_argument('--wandb_account', type=str, default='xxx')
parser.add_argument('--gpu_mem_frac', type=float, default=0)
parser.add_argument('--config', type=str, default='')
parser.add_argument('--config_id', type=int, default=0)
parser.add_argument('--config_name', type=str, default='')
parser.add_argument('--cwd', type=str, default='')

# SSM Arugments
parser.add_argument("--ssm", action="store_true", default=False)
parser.add_argument('--primary_type', type=str, default='')
parser.add_argument('--secondary_type', type=str, default='')
parser.add_argument('--tertiary_type', type=str, default='')

parser.add_argument('--optimizer_name_ssm', type=str, default='')
parser.add_argument('--learning_rate_ssm', type=float, default=1)
parser.add_argument('--momentum_ssm', type=float, default=1)
parser.add_argument('--alpha_ssm', type=float, default=1)
parser.add_argument('--weight_decay_ssm', type=float, default=0)

parser.add_argument("--init_d_model", type=int, default=128)
parser.add_argument("--init_n_layers", type=int, default=4)
parser.add_argument("--init_dropout", type=float, default=0.0) # TODO: original value: 0.0
parser.add_argument("--init_prenorm", type=bool, default=True)
parser.add_argument("--init_embedding", type=bool, default=False)

parser.add_argument("--init_layer_N", type=int, default=64)
parser.add_argument("--init_layer_l_max", type=int, default=784)
parser.add_argument("--init_layer_scaling", type=str, default='linear')
parser.add_argument("--log_step_dt_min", type=float, default=0.001)
parser.add_argument("--log_step_dt_max", type=float, default=0.1)

args = parser.parse_args()
args_dict = vars(args)

config_file = args.config
config_id = args.config_id
config_name = args.config_name

if config_file != '':
    with open(config_file, 'r') as f:
        config_json = json.load(f)

    configurations = config_json['configurations']
    if config_name == '':
        name = configurations[config_id]['name']
    else:
        name = config_name
    id = configurations[config_id]['id']
    cwd = configurations[config_id]['cwd']
    parser_args_list = configurations[config_id]['args']
    env_args = configurations[config_id]['env']

    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False
        
    parser_args = {}

    for i in range(len(parser_args_list)):
        if parser_args_list[i].startswith('--'):
            key = parser_args_list[i][2:]
            value = parser_args_list[i+1]
            parser_args[key] = value

    print(f"\nConfig name: {name}")
    print(f"\nConfig id: {id}")
    print(f"\nEnvironment args:\n\n{env_args}")

    for key in parser_args:
        args_dict[key] = parser_args[key]

    for key in parser_args:
        if isinstance(parser_args[key], int):
            args_dict[key] = int(parser_args[key])
        elif isinstance(parser_args[key], str) and parser_args[key].isnumeric():
            args_dict[key] = int(parser_args[key])
        elif isinstance(parser_args[key], str) and is_float(parser_args[key]):
            args_dict[key] = float(parser_args[key])
        elif parser_args[key] == 'True' or parser_args[key] == 'False':
            args_dict[key] = True if parser_args[key] == 'True' else False

    for key in env_args:
        os.environ[key] = env_args[key]

# if args_dict["dataset"] == "cifar10":
#     args_dict["init_layer_l_max"] = 1024
#     args_dict["init_d_model"] = 128
#     args_dict["init_n_layers"] = 6
#     args_dict["init_dropout"] = 0.25

dataset = args_dict["dataset"]
sensitive_attribute = args_dict["sensitive_attribute"]
group_dro = args_dict["group_dro"]
data_balancing = args_dict["data_balancing"]
gdro_step_size = args_dict["gdro_step_size"]
quick_eval = args_dict["quick_eval"]
fairness_eval = args_dict["fairness_eval"]
fairness_train = args_dict["fairness_train"]
val_train_frac = args_dict["val_train_frac"]
dataset_group_scale = args_dict["dataset_group_scale"]
validation_training = args_dict["validation_training"]
final_layer_retraining = args_dict["final_layer_retraining"]
empirical_fairness_prior_scale = args_dict["empirical_fairness_prior_scale"]
llm_dropout = args_dict["llm_dropout"]
prediction_type = args_dict["prediction_type"]
batch_size = args_dict["batch_size"]
context_batch_size = args_dict["context_batch_size"]
training_dataset_size = args_dict["training_dataset_size"]
context_dataset_size = args_dict["context_dataset_size"]
num_epochs = args_dict["num_epochs"]
learning_rate = args_dict["learning_rate"]
lr_schedule_name = args_dict["lr_schedule_name"]
learning_rate_scale_logvar = args_dict["learning_rate_scale_logvar"]
alpha = args_dict["alpha"]
momentum = args_dict["momentum"]
optimizer_name = args_dict["optimizer_name"]
model_name = args_dict["model_name"]
method = args_dict["method"]
reg_type = args_dict["reg_type"]
weight_decay = args_dict["weight_decay"]
context_points = args_dict["context_points"]
forward_points = args_dict["forward_points"]
reg_points = args_dict["reg_points"]
context_transform = args_dict["context_transform"]
ood_points = args_dict["ood_points"]
mc_samples_llk = args_dict["mc_samples_llk"]
mc_samples_reg = args_dict["mc_samples_reg"]
mc_samples_eval = args_dict["mc_samples_eval"]
reg_scale = args_dict["reg_scale"]
prior_mean = args_dict["prior_mean"]
prior_var = args_dict["prior_var"]
prior_likelihood_scale = args_dict["prior_likelihood_scale"]
prior_likelihood_f_scale = args_dict["prior_likelihood_f_scale"]
prior_likelihood_cov_scale = args_dict["prior_likelihood_cov_scale"]
prior_likelihood_cov_diag = args_dict["prior_likelihood_cov_diag"]
prior_likelihood_mean = args_dict["prior_likelihood_mean"]
prior_likelihood_normalize_feature = args_dict["prior_likelihood_normalize_feature"]
likelihood_scale = args_dict["likelihood_scale"]
output_var = args_dict["output_var"]
rho_sam = args_dict["rho_sam"]
rho_adversarial = args_dict["rho_adversarial"]
dropout_rate_sam = args_dict["dropout_rate_sam"]
prior_params_var = args_dict["prior_params_var"]
init_logvar = args_dict["init_logvar"]
init_final_layer_weights_logvar = args_dict["init_final_layer_weights_logvar"]
init_final_layer_bias_logvar = args_dict["init_final_layer_bias_logvar"]
prior_feature_logvar = args_dict["prior_feature_logvar"]
prior_precision = args_dict["prior_precision"]
pretrained_prior = args_dict["pretrained_prior"]
linearize = args_dict["linearize"]
seed = args_dict["seed"]
evaluate = args_dict["evaluate"]
full_eval = args_dict["full_eval"]
restore_checkpoint = args_dict["restore_checkpoint"]
checkpoint_dir = args_dict["checkpoint_dir"]
final_layer_random_init = args_dict["final_layer_random_init"]
batch_stats_init_epochs = args_dict["batch_stats_init_epochs"]
debug = args_dict["debug"]
debug_print = args_dict["debug_print"]
debug_psd = args_dict["debug_psd"]
log_frequency = args_dict["log_frequency"]
log_frequency_steps = args_dict["log_frequency_steps"]
save_to_wandb = args_dict["save_to_wandb"]
wandb_project = args_dict["wandb_project"]
wandb_account = args_dict["wandb_account"]
gpu_mem_frac = args_dict["gpu_mem_frac"]
cwd = args_dict["cwd"]

ssm = args_dict["ssm"]
primary_type = args_dict["primary_type"]
secondary_type = args_dict["secondary_type"]
tertiary_type = args_dict["tertiary_type"]

# SSM Arugments
init_d_model = args_dict["init_d_model"]
init_n_layers = args_dict["init_n_layers"]
init_dropout = args_dict["init_dropout"]
init_prenorm = args_dict["init_prenorm"]
init_embedding = args_dict["init_embedding"]

optimizer_name_ssm = args_dict["optimizer_name_ssm"]
learning_rate_ssm = args_dict["learning_rate_ssm"]
momentum_ssm = args_dict["momentum_ssm"]
alpha_ssm = args_dict["alpha_ssm"]
weight_decay_ssm = args_dict["weight_decay_ssm"]

init_layer_N = args_dict["init_layer_N"]
init_layer_l_max = args_dict["init_layer_l_max"]
init_layer_scaling = args_dict["init_layer_scaling"]

log_step_dt_min = args_dict["log_step_dt_min"]
log_step_dt_max = args_dict["log_step_dt_max"]

if cwd == '':
    ValueError("You need to define CWD in config!")

if 'bert' in model_name:
    try:
        from transformers import FlaxBertForSequenceClassification, BertConfig, BertTokenizer
    except:
        print("Installing huggingface transformers")            
        import subprocess
        subprocess.run(["pip", "install", "transformers"])
        from transformers import FlaxBertForSequenceClassification, BertConfig, BertTokenizer

print(f"\nParser args:\n\n{args_dict}")

print(f"\nCWD: {cwd}")
os.chdir(cwd)

# if gpu_mem_frac != 0:
#     os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(gpu_mem_frac)

# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = "data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "checkpoints"

# Seeding for random operations
print(f"\nSeed: {seed}")
main_rng = random.PRNGKey(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
random_py.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
torch.random.manual_seed(seed)

rng_key = main_rng

if debug:
    config.update('jax_disable_jit', True)

jitter = eps = 1e-6

print(f"\nDevice: {jax.devices()[0]}\n")


def calibration(y, p_mean, num_bins=10):
  """Compute the calibration.
  References:
  https://arxiv.org/abs/1706.04599
  https://arxiv.org/abs/1807.00263
  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_mean: numpy array, size (?, num_classes)
           containing the mean output predicted probabilities
    num_bins: number of bins
  Returns:
    ece: Expected Calibration Error
    mce: Maximum Calibration Error
  """
  # Compute for every test sample x, the predicted class.
  class_pred = np.argmax(p_mean, axis=1)
  # and the confidence (probability) associated with it.
  conf = np.max(p_mean, axis=1)
  # Convert y from one-hot encoding to the number of the class
  y = np.argmax(y, axis=1)
  # Storage
  acc_tab = np.zeros(num_bins)  # empirical (true) confidence
  mean_conf = np.zeros(num_bins)  # predicted confidence
  nb_items_bin = np.zeros(num_bins)  # number of items in the bins
  tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
  for i in np.arange(num_bins):  # iterate over the bins
    # select the items where the predicted max probability falls in the bin
    # [tau_tab[i], tau_tab[i + 1)]
    sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
    nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
    # select the predicted classes, and the true classes
    class_pred_sec, y_sec = class_pred[sec], y[sec]
    # average of the predicted max probabilities
    mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
    # compute the empirical confidence
    acc_tab[i] = np.mean(
        class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

  # Cleaning
  mean_conf = mean_conf[nb_items_bin > 0]
  acc_tab = acc_tab[nb_items_bin > 0]
  nb_items_bin = nb_items_bin[nb_items_bin > 0]

  # Expected Calibration Error
  ece = np.average(
      np.absolute(mean_conf - acc_tab),
      weights=nb_items_bin.astype(float) / np.sum(nb_items_bin))
  # Maximum Calibration Error
  mce = np.max(np.absolute(mean_conf - acc_tab))
  return ece, mce


@jax.jit
def accuracy(logits_or_p, Y):
    '''Compute accuracy
    Arguments:
        logits_or_p: (B, d)
        Y: (B,) integer labels.
    '''
    if len(Y) == 0:
        return 0.
    matches = jnp.argmax(logits_or_p, axis=-1) == Y
    return jnp.mean(matches)


@jax.jit
def categorical_nll(logits, Y):
    '''Negative log-likelihood of categorical distribution.
    '''
    return optax.softmax_cross_entropy_with_integer_labels(logits, Y)


@jax.jit
def categorical_nll_with_softmax(p, Y):
    '''Negative log-likelihood of categorical distribution.
    '''
    return -jnp.sum(jnp.log(p + 1e-10) * jax.nn.one_hot(Y, p.shape[-1]), axis=-1)


@jax.jit
def gaussian_nll(f, Y, likelihood_var):
    '''Negative log-likelihood of Gaussian distribution.
    '''
    likelihood = tfd.Normal(f, likelihood_var ** 0.5)
    nll = jnp.sum(-likelihood.log_prob(Y), -1)
    return nll


@jax.jit
def categorical_entropy(p):
    '''Entropy of categorical distribution.
    Arguments:
        p: (B, d)

    Returns:
        (B,)
    '''
    return -jnp.sum(p * jnp.log(p + eps), axis=-1)


# @jax.jit
def selective_accuracy(p, Y):
    '''Selective Prediction Accuracy
    Uses predictive entropy with T thresholds.
    Arguments:
        p: (B, d)

    Returns:
        (B,)
    '''

    thresholds = np.concatenate([np.linspace(100, 1, 100), np.array([0.1])], axis=0)

    predictions_test = p.argmax(-1)
    accuracies_test = predictions_test == Y
    scores_id = categorical_entropy(p)

    thresholded_accuracies = []
    for threshold in thresholds:
        p = np.percentile(scores_id, threshold)
        mask = np.array(scores_id <= p)
        thresholded_accuracies.append(np.mean(accuracies_test[mask]))
    values_id = np.array(thresholded_accuracies)

    auc_sel_id = 0
    for i in range(len(thresholds)-1):
        if i == 0:
            x = 100 - thresholds[i+1]
        else:
            x = thresholds[i] - thresholds[i+1]
        auc_sel_id += (x * values_id[i] + x * values_id[i+1]) / 2

    return auc_sel_id


def selective_accuracy_test_ood(p_id, p_ood, Y):
    thresholds = np.concatenate([np.linspace(100, 1, 100), np.array([0.1])], axis=0)

    predictions_test = p_id.argmax(-1)
    accuracies_test = predictions_test == Y
    scores_id = categorical_entropy(p_id)

    accuracies_ood = jnp.zeros(p_ood.shape[0])
    scores_ood = categorical_entropy(p_ood)

    accuracies = jnp.concatenate([accuracies_test, accuracies_ood], axis=0)
    scores = jnp.concatenate([scores_id, scores_ood], axis=0)

    thresholded_accuracies = []
    for threshold in thresholds:
        p = np.percentile(scores, threshold)
        mask = np.array(scores <= p)
        thresholded_accuracies.append(np.mean(accuracies[mask]))
    values = np.array(thresholded_accuracies)

    auc_sel = 0
    for i in range(len(thresholds)-1):
        if i == 0:
            x = 100 - thresholds[i+1]
        else:
            x = thresholds[i] - thresholds[i+1]
        auc_sel += (x * values[i] + x * values[i+1]) / 2

    return auc_sel


def auroc_logits(predicted_logits_test, predicted_logits_ood, score, rng_key):
    predicted_targets_test = jax.nn.softmax(predicted_logits_test, axis=-1)
    predicted_targets_ood = jax.nn.softmax(predicted_logits_ood, axis=-1)

    ood_size = predicted_targets_ood.shape[1]
    test_size = predicted_targets_test.shape[1]
    anomaly_targets = jnp.concatenate((np.zeros(test_size), np.ones(ood_size)))
    if score == "entropy":
        entropy_test = categorical_entropy(predicted_targets_test.mean(0))
        entropy_ood = categorical_entropy(predicted_targets_ood.mean(0))
        scores = jnp.concatenate((entropy_test, entropy_ood))
    if score == "expected entropy":
        entropy_test = categorical_entropy(predicted_targets_test).mean(0)
        entropy_ood = categorical_entropy(predicted_targets_ood).mean(0)
        scores = jnp.concatenate((entropy_test, entropy_ood))
    elif score == "mutual information":
        mutual_information_test = categorical_entropy(predicted_targets_test.mean(0)) - categorical_entropy(predicted_targets_test).mean(0)
        mutual_information_ood = categorical_entropy(predicted_targets_ood.mean(0)) - categorical_entropy(predicted_targets_ood).mean(0)
        scores = jnp.concatenate((mutual_information_test, mutual_information_ood))
    else:
        NotImplementedError
    fpr, tpr, _ = roc_curve(anomaly_targets, scores)
    auroc_score = roc_auc_score(anomaly_targets, scores)
    return auroc_score


def merge_params(params_1, params_2):
    flat_params_1 = flax.traverse_util.flatten_dict(params_1)
    flat_params_2 = flax.traverse_util.flatten_dict(params_2)
    flat_params = flat_params_1 | flat_params_2
    unflat_params = flax.traverse_util.unflatten_dict(flat_params)
    return unflat_params


def split_params(params, type="dense"):
    flat_params_fixed = flax.traverse_util.flatten_dict(params)
    flat_params_rest = flax.traverse_util.flatten_dict(params)
    keys = flat_params_fixed.keys()

    i = -1
    for key in list(keys):
        if "Dense" in str(key) and "kernel" in str(key):
            i += 1

    if type == "dense":
        for key in list(keys):
            if f"Dense_{i}" in str(key):  # first check if there may be two final dense layers
                flat_params_fixed.pop(key)
            else:
                flat_params_rest.pop(key)
    elif type == "batch_norm":
        for key in list(keys):
            if "BatchNorm" in str(key):
                flat_params_fixed.pop(key)
            else:
                flat_params_rest.pop(key)
    else:
        raise NotImplementedError

    unflat_params_fixed = flax.traverse_util.unflatten_dict(flat_params_fixed)
    unflat_params_fixed = unflat_params_fixed
    unflat_params_rest = flax.traverse_util.unflatten_dict(flat_params_rest)
    unflat_params_rest = unflat_params_rest

    return unflat_params_fixed, unflat_params_rest


def split_params_ssm(params, primary_type='', secondary_type='', tertiary_type=''):
    flat_params_fixed = flax.traverse_util.flatten_dict(params)
    flat_params_rest = flax.traverse_util.flatten_dict(params)
    keys = flat_params_fixed.keys()

    primary_types = primary_type.split(",")

    if secondary_type == '':
        secondary_check = True
    else:
        secondary_check = False

    if tertiary_type == '':
        teriaty_check = True
    else:
        teriaty_check = False

    for key in list(keys):
        if key[0] in primary_types or key[-1] in primary_types:
            if secondary_type in str(key) or secondary_check:
                if tertiary_type in str(key) or teriaty_check:
                    flat_params_fixed.pop(key)
                else:
                    flat_params_rest.pop(key)
            else:
                flat_params_rest.pop(key)
        else:
            flat_params_rest.pop(key)

    unflat_params_fixed = flax.traverse_util.unflatten_dict(flat_params_fixed)
    unflat_params_fixed = unflat_params_fixed
    unflat_params_rest = flax.traverse_util.unflatten_dict(flat_params_rest)
    unflat_params_rest = unflat_params_rest

    return unflat_params_fixed, unflat_params_rest


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def fair_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    tensors, targets, sensitives = zip(*batch)
    features = pad_sequence(tensors, batch_first=True)
    targets = np.stack(targets)
    sensitives = np.stack(sensitives)
    return np.array(features, dtype=np.float32), targets, sensitives

class CustomDataset(Dataset):
    def __init__(self, original_dataset, desired_size):
        self.original_dataset = original_dataset
        self.desired_size = desired_size

    def __len__(self):
        return self.desired_size

    def __getitem__(self, idx):
        idx = idx % len(self.original_dataset)  # wrap around the original dataset
        return self.original_dataset[idx]


class TabularDataset(Dataset):
    def __init__(self, X, Y, S, want_sensitive=False): # loading from tabular data
        # S: sensitive features
        # want_sensitive (bool): a flag indicating if sensitive features should be included in the output
        assert X.shape[0] == Y.shape[0] == S.shape[0] # ensure the first dimension (number of samples) is the same
        self.size = X.shape[0]
        self.X = X
        self.Y = Y
        self.S = S
        self.num_sensitive = np.unique(S).shape[0]
        self.want_sensitive = want_sensitive

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.want_sensitive:
            return (self.X[index], self.Y[index], self.S[index])
        else:
            return (self.X[index], self.Y[index])

class SubpopDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    SPLITS = {               # Default, subclasses may override
        'tr': 0,
        'va': 1,
        'te': 2
    }
    EVAL_SPLITS = ['te']     # Default, subclasses may override

    def __init__(self, root, split, metadata, transform, train_attr='yes', subsample_type=None, duplicates=None):
        df = pd.read_csv(metadata)
        df = df[df["split"] == (self.SPLITS[split])]

        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.y = df["y"].tolist()
        self.a = df["a"].tolist() if train_attr == 'yes' else [0] * len(df["a"].tolist())
        self.transform_ = transform
        self._count_groups()

        if subsample_type is not None:
            self.subsample(subsample_type)

        if duplicates is not None:
            self.duplicate(duplicates)

    def _count_groups(self):
        self.weights_g, self.weights_y = [], []
        self.num_attributes = len(set(self.a))
        self.num_labels = len(set(self.y))
        self.group_sizes = [0] * self.num_attributes * self.num_labels
        self.class_sizes = [0] * self.num_labels

        for i in self.idx:
            self.group_sizes[self.num_attributes * self.y[i] + self.a[i]] += 1
            self.class_sizes[self.y[i]] += 1

        for i in self.idx:
            self.weights_g.append(len(self) / self.group_sizes[self.num_attributes * self.y[i] + self.a[i]])
            self.weights_y.append(len(self) / self.class_sizes[self.y[i]])

    def subsample(self, subsample_type):
        assert subsample_type in {"group", "class"}
        perm = torch.randperm(len(self)).tolist()
        min_size = min(list(self.group_sizes)) if subsample_type == "group" else min(list(self.class_sizes))

        counts_g = [0] * self.num_attributes * self.num_labels
        counts_y = [0] * self.num_labels
        new_idx = []
        for p in perm:
            y, a = self.y[self.idx[p]], self.a[self.idx[p]]
            if (subsample_type == "group" and counts_g[self.num_attributes * int(y) + int(a)] < min_size) or (
                    subsample_type == "class" and counts_y[int(y)] < min_size):
                counts_g[self.num_attributes * int(y) + int(a)] += 1
                counts_y[int(y)] += 1
                new_idx.append(self.idx[p])

        self.idx = new_idx
        self._count_groups()

    def duplicate(self, duplicates):
        new_idx = []
        for i, duplicate in zip(self.idx, duplicates):
            new_idx += [i] * duplicate
        self.idx = new_idx
        self._count_groups()

    def __getitem__(self, index):
        i = self.idx[index]
        x = self.transform(self.x[i])
        y = self.y[i]
        a = self.a[i]
        return i, x, y, a

    def __len__(self):
        return len(self.idx)

class CivilComments(SubpopDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None,
                 granularity="coarse"):
        text = pd.read_csv(os.path.join(
            data_path, "civilcomments/civilcomments_{}.csv".format(granularity))
        )
        metadata = os.path.join(data_path, "civilcomments", "metadata_civilcomments_{}.csv".format(granularity))

        self.text_array = list(text["comment_text"])
        if model_name == 'bert-base-uncased':
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif model_name == 'bert-base-cased':
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        else:
            raise NotImplementedError
        self.data_type = "text"
        super().__init__("", split, metadata, self.transform, train_attr, subsample_type, duplicates)

    def transform(self, i):
        text = self.text_array[int(i)]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",
        )

        if len(tokens) == 3:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"]
                ), dim=2
                ), dim=0
            )
        else:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"]
                ), dim=2
                ), dim=0
            )
class CivilCommentsFine(CivilComments):
    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        super().__init__(data_path, split, hparams, train_attr, subsample_type, duplicates, "fine")


class UTKFace(Dataset):
    def __init__(self, root='./data/utkface', split='train', target='age', sensitive='race', want_sensitive=False, transform=None):
        if split == 'train':
            self.images = os.listdir(root + '/UTKFace')
            self.root = root + '/UTKFace'
        elif split == 'test':
            self.images = os.listdir(root + '/crop_part1')
            self.root = root + '/crop_part1'
        else:
            raise ValueError('Invalid split {}'.format(split))
        self.images = [img for img in self.images if len(img.split('_')) == 4]
        self.transform = transform
        # extract age from filename
        age = np.array([img.split('_')[0] for img in self.images], dtype=int)
        age[age <= 19] = 0
        age[(age > 19) & (age <= 40)] = 1
        age[age > 40] = 2
        self.age = age # categorize age to 0-19, 20-40, 41-116, just like how they did in MFD paper
        # extract gender from filename 
        self.gender = np.array([img.split('_')[1] for img in self.images], dtype=int)
        # extract race from filename
        self.race = np.array([img.split('_')[2] for img in self.images], dtype=int)
        # select the indices where race is 4
        other_race_idx = np.where(self.race == 4)[0]
        if sensitive == 'race':
            # remove from self.race and self.images the other_race_idx
            self.race = np.delete(self.race, other_race_idx)
            self.age = np.delete(self.age, other_race_idx)
            self.images = np.delete(np.array(self.images, dtype=str), other_race_idx)
            self.sensitive = self.race 
            self.num_sensitive = 4
        elif sensitive == 'sex':
            self.sensitive = self.gender 
            self.num_sensitive = 2
        else:
            raise ValueError('Invalid sensitive attribute: {}'.format(sensitive))

        self.images_list = self.images
        self.length = len(self.images)

        if os.path.exists(os.path.join(self.root, f'loaded_img_{split}.pkl')):
            self.images = pickle.load(open(os.path.join(self.root, f'loaded_img_{split}.pkl'), 'rb'))
        else:
            self.images = []
            for i in tqdm(range(self.length)):
                img = Image.open(os.path.join(self.root, self.images_list[i]))
                temp = img.copy()
                self.images.append(temp)
                img.close()
            pickle.dump(self.images, open(os.path.join(self.root, f'loaded_img_{split}.pkl'), 'wb'))
        self.images = np.vstack(self.images).reshape(-1, 200, 200, 3)

        self.target = self.age
        self.num_labels = 3
        self.want_sensitive = want_sensitive

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        image = Image.fromarray(self.images[index])
        if self.transform: image = self.transform(image)
        return image, self.target[index], self.sensitive[index]

    def __len__(self):
        return self.length

class MultiNLI(Dataset):
    """
    MNLI
    label_dict = {
        'contradiction' : 0,
        'entailment' : 1,
        'neutral' : 2
    }
    Negation words taken from https://arxiv.org/pdf/1803.02324.pdf
    negation_words = ['nobody', 'no', 'never', 'nothing']
    """
    def __init__(self, root='data', split='train', transform=None, want_sensitive=True, **kwargs):
        self.data_dir = os.path.join(root, 'MNLI')
        if not os.path.exists(self.data_dir):
            raise ValueError('Dataset not found at {}'.format(self.data_dir))
        self.transform = transform
        self.want_sensitive = want_sensitive

        self.target_name = 'gold_label_random'
        # self.target_name = 'gold_label_preset'
        # If 'preset', use the official train/val/test MultiNLI split
        # If 'random', randomly split 50%/20%/30% of the data to train/val/test

        self.confounder_names = 'sentence2_has_negation'

        type_of_split = self.target_name.split('_')[-1]
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata_{}.csv'.format(type_of_split)),
            index_col=0
        )

        self.target = self.metadata_df['gold_label'].values 
        self.num_labels = len(np.unique(self.target))
        self.sensitive = self.metadata_df[self.confounder_names].values 
        self.num_sensitive = len(np.unique(self.sensitive))

        self.num_groups = self.num_sensitive * self.num_labels
        self.group = (self.target*(self.num_groups/self.num_labels) + self.sensitive)

        self.split_array = self.metadata_df['split'].values
        self.split = split 
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        if split not in ['train', 'val', 'test']:
            raise ValueError('Invalid split {}'.format(split))

        import utils_glue
        self.features_array = []
        for feature_file in [
            'cached_train_bert-base-uncased_128_mnli',
            'cached_dev_bert-base-uncased_128_mnli',
            'cached_dev_bert-base-uncased_128_mnli-mm'
        ]:
            features = torch.load(
                os.path.join(self.data_dir, feature_file)
            )
            self.features_array += features
        
        self.all_input_ids = np.array([f.input_ids for f in self.features_array], dtype=np.float32)
        self.all_input_masks = np.array([f.input_mask for f in self.features_array], dtype=np.float32)
        self.all_segment_ids = np.array([f.segment_ids for f in self.features_array], dtype=np.float32)
        self.all_label_ids = np.array([f.label_id for f in self.features_array], dtype=np.float32)

        # self.inputs = np.stack((self.all_input_ids, self.all_input_masks, self.all_segment_ids), axis=1)
        self.inputs = np.stack((self.all_input_ids, self.all_segment_ids, self.all_input_masks), axis=1)

        self.inputs = self.inputs[self.split_array == self.split_dict[split]]
        self.target = torch.from_numpy(self.target[self.split_array == self.split_dict[split]])
        self.sensitive = torch.from_numpy(self.sensitive[self.split_array == self.split_dict[split]])
        self.group = torch.from_numpy(self.group[self.split_array == self.split_dict[split]])

        self.group_counts = (torch.arange(self.num_groups).unsqueeze(1)==self.group).sum(1).float().numpy()
        self.group_weights = len(self) / self.group_counts
        self.group_proportions = self.group_counts / self.group_counts.sum()
        self.weights = np.ones(self.num_groups) / self.num_groups

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]: 
        y = self.target[index]
        s = self.sensitive[index]
        g = self.group[index]
        x = self.inputs[index, ...]
        return x, y, s, g

class MultiNLIDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # input_ids, token_type_ids, attention mask
        all_data = self.dataset.__getitem__(idx)
        y = all_data.get('labels')
        all_data.pop('labels')
        all_input_ids = all_data.get('input_ids')
        all_token_type_ids =  all_data.get('token_type_ids')
        all_attention_mask = all_data.get('attention_mask')
        x = np.stack((
            all_input_ids,
            all_token_type_ids,
            all_attention_mask
        )) # batch, dim, feature_dim
        return np.array(x), np.array(y)

class Waterbirds(Dataset):
    def __init__(self, root='data', split='train', transform=None, want_sensitive=False, val_split=None, val_train_frac=1, idx=None, **kwargs):
        # Assumes dataset already downloaded and extracted from G-DRO github
        self.data_dir = os.path.join(root, 'waterbird_complete95_forest2water2')
        if not os.path.exists(self.data_dir):
            raise ValueError('Dataset not found at {}'.format(self.data_dir))
        self.transform = transform 
        self.want_sensitive = want_sensitive

        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))
        self.target = torch.from_numpy(self.metadata_df['y'].values)
        self.num_labels = 2
        self.sensitive = torch.from_numpy(self.metadata_df['place'].values)
        self.num_sensitive = 2
        self.num_groups = 4
        self.group = self.target*(self.num_groups/2) + self.sensitive

        # target_label * num_labels + sensitive_group]

        self.filename_array = self.metadata_df['img_filename'].values 
        self.split_array = self.metadata_df['split'].values
        self.split = split
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        if split not in ['train', 'val', 'test']:
            raise ValueError('Invalid split {}'.format(split))

        validation_dataset_size = 1199
        train_validation_dataset_size = int(val_train_frac * validation_dataset_size)  # NOTE: HARDCODED FRACTION!
        validation_validation_dataset_size = validation_dataset_size - train_validation_dataset_size
        
        if idx is not None:
            self.idx = idx

            self.idx_train_validation = self.idx[:train_validation_dataset_size]
            self.idx_validation_validation = self.idx[train_validation_dataset_size:]

        if val_split is None:
            self.filename_array = self.filename_array[self.split_array == self.split_dict[split]]
            self.target = self.target[self.split_array == self.split_dict[split]]
            self.sensitive = self.sensitive[self.split_array == self.split_dict[split]]
            self.group = (self.group[self.split_array == self.split_dict[split]])

        elif val_split == 'val_train':
            self.filename_array = self.filename_array[self.split_array == self.split_dict[split]][self.idx_train_validation]
            self.target = self.target[self.split_array == self.split_dict[split]][self.idx_train_validation]
            self.sensitive = self.sensitive[self.split_array == self.split_dict[split]][self.idx_train_validation]
            self.group = (self.group[self.split_array == self.split_dict[split]])[self.idx_train_validation]

        elif val_split == 'val_val':
            self.filename_array = self.filename_array[self.split_array == self.split_dict[split]][self.idx_validation_validation]
            self.target = self.target[self.split_array == self.split_dict[split]][self.idx_validation_validation]
            self.sensitive = self.sensitive[self.split_array == self.split_dict[split]][self.idx_validation_validation]
            self.group = (self.group[self.split_array == self.split_dict[split]])[self.idx_validation_validation]

        self.group_counts = (torch.arange(self.num_groups).unsqueeze(1)==self.group).sum(1).float().numpy()
        self.group_weights = len(self) / self.group_counts
        self.group_proportions = self.group_counts / self.group_counts.sum()
        self.weights = np.ones(self.num_groups) / self.num_groups

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        image = Image.open(
            os.path.join(self.data_dir, self.filename_array[index])
        ).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, self.target[index], self.sensitive[index], self.group[index]

    def __len__(self):
        return len(self.filename_array)

class CelebADataset(CelebA): # Barebones wrapper for CelebA dataset
    def __init__(self, want_sensitive=False, **kwargs):
        super().__init__(**kwargs)
        self.target = self.attr[:, 9]
        self.sensitive = self.attr[:, 20]
        self.want_sensitive = want_sensitive
        self.num_sensitive = 2
        self.num_labels = 2
        self.num_groups = self.num_sensitive * self.num_labels
        self.group = self.target*(self.num_groups/2) + self.sensitive
        self.group_counts = (torch.arange(self.num_groups).unsqueeze(1)==self.group).sum(1).float().numpy()
        self.group_weights = len(self) / self.group_counts 
        self.group_proportions = self.group_counts / self.group_counts.sum()
        self.weights = np.ones(self.num_groups) / self.num_groups
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, self.target[index], self.sensitive[index], self.group[index]
    def __len__(self):
        return len(self.target)


'''
class CelebADataset(Dataset): # Optimized wrapper for CelebA dataset
    def __init__(self, want_sensitive=False, root='data/celeba', transform=None, split='train', **kwargs):
        self.want_sensitive = want_sensitive
        self.transform = transform
        self.dataset = CelebA(root=root, transform=None, split=split)
        self.length = len(self.dataset)
        #1
        if os.path.exists(os.path.join(root, 'celeba', f'loaded_img_{split}.pkl')):
            self.images = pickle.load(open(os.path.join(root, 'celeba', f'loaded_img_{split}.pkl'), 'rb'))
            #self.images = np.load(os.path.join(root, 'celeba', 'loaded_img.npy'), allow_pickle=True)
        else:
            self.images = []
            for i in tqdm(range(self.length)):
                img = Image.open(os.path.join(root, 'celeba', 'img_align_celeba', self.dataset.filename[i]))
                temp = img.copy()
                self.images.append(temp)
                img.close()
            #np.save(os.path.join(root, 'celeba', 'loaded_img.npy'), self.images)
            pickle.dump(self.images, open(os.path.join(root, 'celeba', f'loaded_img_{split}.pkl'), 'wb'))
        self.images = np.vstack(self.images).reshape(-1, 218, 178, 3)
        self.target = self.dataset.attr[:, 9].numpy()
        self.sensitive = self.dataset.attr[:, 20].numpy()
        self.num_sensitive = 2
        self.num_labels = 2

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        image = Image.fromarray(self.images[index])
        if self.transform: image = self.transform(image)
        return image, self.target[index], self.sensitive[index]

    def __getitems__(self, indices: np.ndarray) -> Tuple[Any, Any, Any]:
        batch_images = self.images[indices]
        batch_targets = self.target[indices]
        batch_sensitive = self.sensitive[indices]
        # Transform images
        transformed_images = []
        for img_array in batch_images:
            img = Image.fromarray(img_array)
            if self.transform:
                img = self.transform(img)
            transformed_images.append(img)
        return np.array(transformed_images, dtype=np.float32), batch_targets, batch_sensitive

    def __len__(self):
        return self.length
'''

class VisionContextDataset(Dataset):
    def __init__(self, dataset, want_sensitive=False):
        self.dataset = dataset
        self.Y = dataset.target
        self.S = dataset.sensitive
        self.num_sensitive = dataset.num_sensitive
        self.num_labels = dataset.num_labels
        self.size = len(self.dataset)
        self.want_sensitive = want_sensitive

        # create a num_sensitive x num_labels matrix of indices
        self.indices = {}
        self.indices_length = {}
        for sensitive in range(self.num_sensitive):
            for label in range(self.num_labels):
                self.indices[(label, sensitive)] = np.where((self.Y == label) & (self.S != sensitive))[0]
                self.indices_length[(label, sensitive)] = len(self.indices[(label, sensitive)])
                print(f"Label: {label}, Sensitive: {sensitive}, Size: {len(self.indices[(label, sensitive)])}")

        #self.indices = np.empty((self.num_labels, self.num_sensitive), dtype=object)
        #self.indices_length = np.empty((self.num_labels, self.num_sensitive), dtype=int)
        #for sensitive in range(self.num_sensitive):
        #    for label in range(self.num_labels):
        #        self.indices[(label, sensitive)] = np.where((self.Y == label) & (self.S != sensitive))[0]
        #        self.indices_length[(label, sensitive)] = len(self.indices[(label, sensitive)])
        #        print(f"Label: {label}, Sensitive: {sensitive}, Size: {len(self.indices[(label, sensitive)])}")

        # create a hash table mapping indices to (label, sensitive) tuples
        self.num_to_tuple_hash = {}
        for tuple_idx in self.indices.keys():
            for idx in self.indices[tuple_idx]:
                self.num_to_tuple_hash[idx] = tuple_idx
        self.get_context_tuple = lambda idx: self.num_to_tuple_hash.get(idx, None)

        self.permute = lambda x: random_np.permutation(x)[0]
        self.get_batch = lambda i: self.dataset[i]

    def __len__(self) -> int:
        return self.size
        
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        tuple_idx = self.get_context_tuple(index)
        context_idx = self.permute(self.indices[tuple_idx])
        return self.dataset[context_idx]
        '''
        if self.want_sensitive:
            X, _, _ = self.dataset[index]
            X = np.array(X, dtype=np.float32)
            X, Y, S = X, self.Y[index], self.S[index]
            return (X, self.Y[index], self.S[index])
        else:
            X, _ = self.dataset[index]
            X = np.array(X, dtype=np.float32)
            return (X, self.Y[index])
        '''
    '''
    def get_fair_batch(self, input_batch: Tuple[Any, Any, Any]) -> Tuple[Any, Any, Any]:
        X, Y, S = input_batch
        index_array = self.indices[(Y.int(), S.int())]
        choices = self.permute(index_array)
        #choices = np.array(list(map(self.permute, index_array)), dtype=int)
        #batch = np.array(list(map(self.get_batch, choices)), dtype=object)
        batch = [self.dataset.dataset[choice] for choice in choices]
        X_prime, Y_prime, S_prime = zip(*batch)
        # X_prime, Y_prime, S_prime = self.dataset.__getitems__(choices)
        return X_prime, Y_prime, S_prime
        #X_prime, Y_prime, S_prime = zip(*batch)
        #X_prime = np.array(X_prime, dtype=np.float32)
        #Y_prime = np.array(Y_prime, dtype=np.float32)
        #S_prime = np.array(S_prime, dtype=np.float32)
        #return X_prime, Y_prime, S_prime 
    '''


class TwoMoons(torch.utils.data.Dataset):
    def __init__(self, size: int, inputs: np.ndarray, targets: np.ndarray):
        self.size = size
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input, target = self.inputs[index], int(self.targets[index][0].sum())
        ret = (input, target)
        return ret


class Regression(torch.utils.data.Dataset):
    def __init__(self, size: int, inputs: np.ndarray, targets: np.ndarray):
        self.size = size
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input, target = self.inputs[index], self.targets[index]
        ret = (input, target)
        return ret


def get_cifar10_test(root=None, v1=False, corr_config=None, batch_size=128, **_):
    _TEST_TRANSFORM = [
        transforms.ToTensor(),
        transforms.Normalize((.4914, .4822, .4465), (.247, .243, .261)),
    ]

    if dataset == 'cifar10-224':
        _TEST_TRANSFORM.append(transforms.Resize(224))

    _TEST_TRANSFORM.append(transforms.Lambda(chw2hwc_fn))

    if v1:
        test_data = create_dataset(name='tfds/cifar10_1', root=root, split='test',
                                    is_training=True, batch_size=batch_size,
                                    transform=transforms.Compose(_TEST_TRANSFORM), download=True)
    elif corr_config is not None:
        test_data = create_dataset(f'tfds/cifar10_corrupted/{corr_config}', root=root, split='test',
                                    is_training=True, batch_size=batch_size,
                                    transform=transforms.Compose(_TEST_TRANSFORM), download=True)
    else:
        test_data = create_dataset('torch/cifar10', root=root, split='test',
                                    transform=transforms.Compose(_TEST_TRANSFORM), download=True)

    return test_data

def create_sensitive_context_set(dataset):
    if isinstance(dataset, TabularDataset):
        X, Y, S = dataset.X, dataset.Y, dataset.S
        X_list, Y_list, S_list = [], [], []

        for i in range(dataset.num_sensitive):
            X_copy = X.copy()
            Y_copy = Y.copy()
            S_copy = S.copy()

            X_copy[:, -1] = i # assumes sensitive attribute is the last column
            S_copy[:] = i

            X_list.append(X_copy)
            Y_list.append(Y_copy)
            S_list.append(S_copy)

        X = np.concatenate(X_list)
        Y = np.concatenate(Y_list)
        S = np.concatenate(S_list)

        return TabularDataset(X, Y, S, want_sensitive=True)
    elif isinstance(dataset, MultiNLI):
        return dataset
    else:
        return VisionContextDataset(dataset, want_sensitive=True)

''' # This is conceptually wrong
def compute_worst_group_accuracy(y_true, y_pred, sensitive_features):
    unique_groups = jnp.unique(sensitive_features)

    group_accuracies = []
    for group in unique_groups:
        group_indices = (sensitive_features == group)
        group_y_true = y_true[group_indices]
        group_y_pred = y_pred[group_indices]

        accuracy = jnp.sum(group_y_true == group_y_pred) / len(group_y_true)
        group_accuracies.append(accuracy)

    worst_group_accuracy = jnp.min(jnp.array(group_accuracies))
    return worst_group_accuracy
'''

def compute_weighted_test_accuracy(y_true, y_pred, sensitive_features, proportions, num_labels=2, num_sensitive=2):
    group_accuracies = []
    group_test_counts = []
    
    for target_label in range(num_labels):
        for sensitive_group in range(num_sensitive):
            group_indices = (sensitive_features == sensitive_group) & (y_true == target_label)

            if jnp.sum(group_indices) == 0:
                group_accuracies.append(0.0)
                continue
            
            group_y_true = y_true[group_indices]
            group_y_pred = y_pred[group_indices]

            accuracy = jnp.sum(group_y_true == group_y_pred) / len(group_y_true)
            group_accuracies.append(accuracy)

    weighted_test_accuracy = jnp.sum(jnp.array(group_accuracies) * jnp.array(proportions))
    return weighted_test_accuracy

def compute_worst_group_accuracy(y_true, y_pred, sensitive_features, num_labels=2, num_sensitive=2):
    group_accuracies = []

    for target_label in range(num_labels):
        for sensitive_group in range(num_sensitive):
            group_indices = (sensitive_features == sensitive_group) & (y_true == target_label)

            # If no elements in the group, continue to the next iteration
            if jnp.sum(group_indices) == 0:
                continue

            group_y_true = y_true[group_indices]
            group_y_pred = y_pred[group_indices]

            accuracy = jnp.sum(group_y_true == group_y_pred) / len(group_y_true)
            group_accuracies.append(accuracy)
    
    worst_group_accuracy = jnp.min(jnp.array(group_accuracies))
    return worst_group_accuracy

'''
def compute_equality_of_opportunity(y_true, y_pred, sensitive_features):
    unique_groups = jnp.unique(sensitive_features)

    group_true_positive_rates = []
    for group in unique_groups:
        group_indices = (sensitive_features == group)
        group_y_true = y_true[group_indices]
        group_y_pred = y_pred[group_indices]

        true_positive_rate = jnp.sum((group_y_true == 1) & (group_y_pred == 1)) / jnp.sum(group_y_true == 1)
        group_true_positive_rates.append(true_positive_rate)

    diff_in_eop = jnp.max(jnp.array(group_true_positive_rates)) - jnp.min(jnp.array(group_true_positive_rates))
    return diff_in_eop
'''

def compute_equality_of_opportunity(y_true, y_pred, sensitive_features, num_labels=2, num_sensitive=2):
    # For normalization when more than two groups
    normalized_factor = num_sensitive * (num_sensitive - 1) / 2.0
    unique_groups = jnp.array(range(num_sensitive))

    eop = 0.0
    for i, group1 in enumerate(unique_groups):
        for group2 in unique_groups[i + 1:]:
            group1_indices = (sensitive_features == group1)
            group2_indices = (sensitive_features == group2)
            
            group1_y_true = y_true[group1_indices]
            group2_y_true = y_true[group2_indices]
            
            group1_y_pred = y_pred[group1_indices]
            group2_y_pred = y_pred[group2_indices]
            
            true_positive_rate_group1 = jnp.sum((group1_y_true == 1) & (group1_y_pred == 1)) / jnp.sum(group1_y_true == 1)
            true_positive_rate_group2 = jnp.sum((group2_y_true == 1) & (group2_y_pred == 1)) / jnp.sum(group2_y_true == 1)
            
            gap = true_positive_rate_group1 - true_positive_rate_group2
            eop += jnp.abs(gap)
            
    if num_sensitive > 2:
        eop /= normalized_factor

    return eop

'''
def compute_equality_of_odds(y_true, y_pred, sensitive_features):
    unique_groups = jnp.unique(sensitive_features)

    group_true_positive_rates = []
    group_true_negative_rates = []

    for group in unique_groups:
        group_indices = (sensitive_features == group)
        group_y_true = y_true[group_indices]
        group_y_pred = y_pred[group_indices]

        true_positive_rate = jnp.sum((group_y_true == 1) & (group_y_pred == 1)) / jnp.sum(group_y_true == 1)
        true_negative_rate = jnp.sum((group_y_true == 0) & (group_y_pred == 0)) / jnp.sum(group_y_true == 0)

        group_true_positive_rates.append(true_positive_rate)
        group_true_negative_rates.append(true_negative_rate)

    diff_in_eoo = max(jnp.max(jnp.array(group_true_positive_rates)) - jnp.min(jnp.array(group_true_positive_rates)),
                      jnp.max(jnp.array(group_true_negative_rates)) - jnp.min(jnp.array(group_true_negative_rates)))

    return diff_in_eoo
'''

def compute_equality_of_odds(y_true, y_pred, sensitive_features, num_labels=2, num_sensitive=2):
    # For normalization when more than two groups
    normalized_factor = num_sensitive * (num_sensitive - 1) / 2.0
    unique_groups = jnp.array(range(num_sensitive))

    eoo_tpr = 0.0  # Equality of Odds for True Positive Rates
    eoo_tnr = 0.0  # Equality of Odds for True Negative Rates

    for i, group1 in enumerate(unique_groups):
        for group2 in unique_groups[i + 1:]:
            group1_indices = (sensitive_features == group1)
            group2_indices = (sensitive_features == group2)

            group1_y_true = y_true[group1_indices]
            group2_y_true = y_true[group2_indices]

            group1_y_pred = y_pred[group1_indices]
            group2_y_pred = y_pred[group2_indices]

            tpr_group1 = jnp.sum((group1_y_true == 1) & (group1_y_pred == 1)) / jnp.sum(group1_y_true == 1)
            tpr_group2 = jnp.sum((group2_y_true == 1) & (group2_y_pred == 1)) / jnp.sum(group2_y_true == 1)

            tnr_group1 = jnp.sum((group1_y_true == 0) & (group1_y_pred == 0)) / jnp.sum(group1_y_true == 0)
            tnr_group2 = jnp.sum((group2_y_true == 0) & (group2_y_pred == 0)) / jnp.sum(group2_y_true == 0)

            gap_tpr = tpr_group1 - tpr_group2
            gap_tnr = tnr_group1 - tnr_group2

            eoo_tpr += jnp.abs(gap_tpr)
            eoo_tnr += jnp.abs(gap_tnr)

    if num_sensitive > 2:
        eoo_tpr /= normalized_factor
        eoo_tnr /= normalized_factor

    return max(eoo_tpr, eoo_tnr)

'''
def compute_predictive_rate_parity(y_true, y_pred, sensitive_features):
    unique_groups = jnp.unique(sensitive_features)

    group_ppv = []
    for group in unique_groups:
        group_indices = (sensitive_features == group)
        group_y_true = y_true[group_indices]
        group_y_pred = y_pred[group_indices]

        ppv = jnp.sum((group_y_true == 1) & (group_y_pred == 1)) / jnp.sum(group_y_pred == 1)
        group_ppv.append(ppv)

    diff_in_prp = jnp.max(jnp.array(group_ppv)) - jnp.min(jnp.array(group_ppv))
    return diff_in_prp
'''

def compute_predictive_rate_parity(y_true, y_pred, sensitive_features, num_labels=2, num_sensitive=2):
    # For normalization when more than two groups
    normalized_factor = num_sensitive * (num_sensitive - 1) / 2.0  # For normalization when more than two groups
    unique_groups = jnp.array(range(num_sensitive))
    total_ppv_disparity = 0.0 

    for i, group1 in enumerate(unique_groups):
        for group2 in unique_groups[i + 1:]:
            group1_indices = (sensitive_features == group1)
            group2_indices = (sensitive_features == group2)

            group1_y_true = y_true[group1_indices]
            group2_y_true = y_true[group2_indices]

            group1_y_pred = y_pred[group1_indices]
            group2_y_pred = y_pred[group2_indices]

            ppv_group1 = jnp.sum((group1_y_true == 1) & (group1_y_pred == 1)) / jnp.sum(group1_y_pred == 1)
            ppv_group2 = jnp.sum((group2_y_true == 1) & (group2_y_pred == 1)) / jnp.sum(group2_y_pred == 1)

            gap_ppv = ppv_group1 - ppv_group2
            total_ppv_disparity += jnp.abs(gap_ppv)  
    
    if num_sensitive > 2:
        total_ppv_disparity /= normalized_factor 

    return total_ppv_disparity

num_workers_test = 4
persistent_workers_test = True

if dataset == 'cifar10' or dataset == 'cifar10-224':
    # _train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
    _train_dataset = CIFAR10(root="./data/CIFAR10", train=True, download=True)
    input_dim = 32
    num_classes = 10
    dataset_size = 50000
    testset_size = 10000
    if training_dataset_size == 0:
        training_dataset_size = 50000
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size
    
    exmp_input = jnp.ones([1, input_dim, input_dim, 3])

    # DATA_MEANS = (_train_dataset.data / 255.0).mean(axis=(0,1,2))
    # DATA_STD = (_train_dataset.data / 255.0).std(axis=(0,1,2))
    # DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    # DATA_STD = np.array([0.2023, 0.1994, 0.2010])
    DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    DATA_STD = np.array([0.247, 0.243, 0.261])
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    if ssm:
        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img.reshape((img.shape[-2]**2, img.shape[-1]))
        def image_to_numpy_context(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS_CONTEXT) / DATA_STD_CONTEXT
            return img.reshape((img.shape[-2]**2, img.shape[-1]))

    else:
        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img
        def image_to_numpy_context(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS_CONTEXT) / DATA_STD_CONTEXT
            return img

    test_transform_list = [
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
    ]
    train_transform_list = [
        transforms.RandomCrop(input_dim, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
    ]

    if dataset == 'cifar10-224':
        test_transform_list.append(transforms.Resize(224))
        train_transform_list.append(transforms.Resize(224))

    test_transform_list.append(image_to_numpy)
    train_transform_list.append(image_to_numpy)

    test_transform = transforms.Compose(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    _train_dataset = CIFAR10(root="./data/CIFAR10", train=True, transform=train_transform, download=True)

    # resize_transform = transforms.Compose([
    #     transforms.Lambda(lambda x: Image.fromarray((x).astype(np.uint8)) if isinstance(x, np.ndarray) else x),
    #     transforms.Resize((224, 224)),  # Resize images to 224 x 224
    # ])

    # if dataset == 'cifar10-224':
    #     dataset_new = []
    #     for i in range(_train_dataset.data.shape[0]):
    #         dataset_new.append(resize_transform(_train_dataset.data[i]))
    #     dataset_new = np.stack(dataset_new)
    #     _train_dataset.data = dataset_new

    # val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    val_dataset = CIFAR10(root="./data/CIFAR10", train=True, transform=test_transform, download=True)

    train_dataset, _ = torch.utils.data.random_split(_train_dataset, [training_dataset_size, dataset_size-training_dataset_size], generator=torch.Generator().manual_seed(seed))
    train_dataset = CustomDataset(train_dataset, dataset_size)
    _, validation_dataset = torch.utils.data.random_split(val_dataset, [dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    # test_dataset = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)
    test_dataset = CIFAR10(root="./data/CIFAR10", train=False, transform=test_transform, download=True)

    if False:  # context_points == "imagenet":
        DATA_MEANS_CONTEXT = np.array([0.485, 0.456, 0.406])
        DATA_STD_CONTEXT = np.array([0.229, 0.224, 0.225])
    else:
        DATA_MEANS_CONTEXT = DATA_MEANS
        DATA_STD_CONTEXT = DATA_STD

    if context_transform:
        context_transform_list = [
            transforms.RandomCrop(input_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=(3,3)),
            # transforms.GaussianBlur(kernel_size=(3,3), sigma=0.1),
            transforms.RandomSolarize(threshold=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize(32),
        ]
        if dataset == 'cifar10-224':
            if context_points == "imagenet":
                context_transform_list.append(transforms.RandomResizedCrop(224))
            else:
                context_transform_list.append(transforms.Resize(224))
        context_transform_list.append(image_to_numpy_context)
        context_transform = transforms.Compose(context_transform_list)
    else:
        if context_points == "imagenet":
            context_transform_list = [
                transforms.RandomResizedCrop(224),
                # transforms.Resize(32),
                # transforms.Resize(224),
            ]
            context_transform_list.append(image_to_numpy_context)
            context_transform = transforms.Compose(context_transform_list)
        else:
            context_transform = test_transform

    if context_points == "train":
        context_dataset = CIFAR10(root="./data/CIFAR10", train=True, transform=context_transform, download=True)
    elif context_points == "cifar100":
        context_dataset = CIFAR100(root="./data/CIFAR100", train=True, transform=context_transform, download=True)
    elif context_points == "svhn":
        context_dataset = SVHN(root="./data/SVHN", split="train", download=True, transform=context_transform)
    elif context_points == "imagenet":
        context_dataset = ImageNet(root="./data/ImageNet", transform=context_transform)
    else:
        ValueError("Unknown context dataset")
    
    # if dataset == 'cifar10-224' and context_points != "imagenet":
    #     dataset_new = []
    #     for i in range(context_dataset.data.shape[0]):
    #         dataset_new.append(resize_transform(context_dataset.data[i]))
    #     dataset_new = np.stack(dataset_new)
    #     context_dataset.data = dataset_new

    full_context_dataset_size = len(context_dataset)
    context_set, _ = torch.utils.data.random_split(context_dataset, [context_dataset_size, full_context_dataset_size - context_dataset_size], generator=torch.Generator().manual_seed(seed))
    context_set = CustomDataset(context_set, training_dataset_size)

    if dataset == 'cifar10-224':
        ood_transform = transforms.Compose([
            transforms.Resize(224),
            image_to_numpy,
        ])
    else:
        ood_transform = test_transform

    if ood_points == "svhn":
        ood_dataset = SVHN(root="./data/SVHN", split="test", download=True, transform=ood_transform)
    elif ood_points == "cifar100":
        ood_dataset = CIFAR100(root="./data/CIFAR100", train=False, download=True, transform=ood_transform)
    else:
        ValueError("Unknown OOD dataset")

    ood_dataset = CustomDataset(ood_dataset, len(test_dataset))
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                #  collate_fn=numpy_collate,
                                 num_workers=num_workers_test,
                                 persistent_workers=persistent_workers_test
                                 )
    
    cifar101test_data = get_cifar10_test(root="./data/CIFAR10", seed=seed, v1=True, corr_config=None, batch_size=batch_size_test)

    cifar101test_loader  = data.DataLoader(cifar101test_data,
                                batch_size=batch_size_test,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers_test,
                                persistent_workers=persistent_workers_test
                                )

    try:
        if full_eval:
            corr_config_list = [
                "speckle_noise_1", "speckle_noise_2", "speckle_noise_3", "speckle_noise_4", "speckle_noise_5",
                "shot_noise_1", "shot_noise_2", "shot_noise_3", "shot_noise_4", "shot_noise_5",
                "pixelate_1", "pixelate_2", "pixelate_3", "pixelate_4", "pixelate_5",
                "gaussian_blur_1", "gaussian_blur_2", "gaussian_blur_3", "gaussian_blur_4", "gaussian_blur_5",
                ]
            ccifar10test_loader_list = []
            for corr_config in corr_config_list:
                ccifar10test_data = get_cifar10_test(root="./data/CIFAR10", seed=seed, v1=False, corr_config=corr_config, batch_size=batch_size_test)

                ccifar10test_loader  = data.DataLoader(ccifar10test_data,
                                            batch_size=batch_size_test,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=num_workers_test,
                                            persistent_workers=persistent_workers_test
                                            )
                ccifar10test_loader_list.append(ccifar10test_loader)
    except:
        print("Could not load corrupted CIFAR10 datasets.")
        assert full_eval == False, "Could not load corrupted CIFAR10 datasets. Please set full_eval to False or modify timm library to enabling loading datasets."

elif dataset == 'cifar100' or dataset == 'cifar100-224':
    _train_dataset = CIFAR100(root="./data/CIFAR100", train=True, download=True)
    input_dim = 32
    num_classes = 100
    dataset_size = 50000
    testset_size = 10000
    if training_dataset_size == 0:
        training_dataset_size = 50000
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size

    exmp_input = jnp.ones([1, input_dim, input_dim, 3])

    DATA_MEANS = (_train_dataset.data / 255.0).mean(axis=(0,1,2))
    DATA_STD = (_train_dataset.data / 255.0).std(axis=(0,1,2))
    # DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    # DATA_STD = np.array([0.2023, 0.1994, 0.2010])
    # DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    # DATA_STD = np.array([0.247, 0.243, 0.261])
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img

    test_transform_list = [
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
    ]
    train_transform_list = [
        transforms.RandomCrop(input_dim, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
    ]

    if dataset == 'cifar100-224':
        test_transform_list.append(transforms.Resize(224))
        train_transform_list.append(transforms.Resize(224))

    test_transform_list.append(image_to_numpy)
    train_transform_list.append(image_to_numpy)

    test_transform = transforms.Compose(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    _train_dataset = CIFAR100(root="./data/CIFAR100", train=True, transform=train_transform, download=True)
    # val_dataset = CIFAR100(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    val_dataset = CIFAR100(root="./data/CIFAR100", train=True, transform=test_transform, download=True)

    train_dataset, _ = torch.utils.data.random_split(_train_dataset, [training_dataset_size, dataset_size-training_dataset_size], generator=torch.Generator().manual_seed(seed))
    train_dataset = CustomDataset(train_dataset, dataset_size)
    _, validation_dataset = torch.utils.data.random_split(val_dataset, [dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    test_dataset = CIFAR100(root="./data/CIFAR100", train=False, transform=test_transform, download=True)

    if False: #  context_points == "imagenet":
        DATA_MEANS_CONTEXT = np.array([0.485, 0.456, 0.406])
        DATA_STD_CONTEXT = np.array([0.229, 0.224, 0.225])
    else:
        DATA_MEANS_CONTEXT = DATA_MEANS
        DATA_STD_CONTEXT = DATA_STD

    def image_to_numpy_context(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS_CONTEXT) / DATA_STD_CONTEXT
        return img
    
    if context_transform:
        context_transform_list = [
            transforms.RandomCrop(input_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.GaussianBlur(kernel_size=(3,3)),
            transforms.GaussianBlur(kernel_size=(3,3), sigma=0.1),
            transforms.RandomSolarize(threshold=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize(32),
        ]
        if dataset == 'cifar100-224':
            if context_points == "imagenet":
                context_transform_list.append(transforms.RandomResizedCrop(224))
            else:
                context_transform_list.append(transforms.Resize(224))
        context_transform_list.append(image_to_numpy_context)
        context_transform = transforms.Compose(context_transform_list)
    else:
        if context_points == "imagenet":
            context_transform_list = [
                transforms.RandomResizedCrop(224),
                # transforms.Resize(32),
                # transforms.Resize(224),
            ]
            context_transform_list.append(image_to_numpy_context)
            context_transform = transforms.Compose(context_transform_list)
        else:
            context_transform = test_transform

    if context_points == "train":
        context_dataset = CIFAR100(root="./data/CIFAR100", train=True, transform=context_transform, download=True)
    elif context_points == "cifar100":
        context_dataset = CIFAR10(root="./data/CIFAR10", train=True, transform=context_transform, download=True)
    elif context_points == "svhn":
        context_dataset = SVHN(root="./data/SVHN", split="train", download=True, transform=context_transform)
    elif context_points == "imagenet":
        context_dataset = ImageNet(root="./data/ImageNet", transform=context_transform)
    else:
        ValueError("Unknown context dataset")
    
    full_context_dataset_size = len(context_dataset)
    context_set, _ = torch.utils.data.random_split(context_dataset, [context_dataset_size, full_context_dataset_size - context_dataset_size], generator=torch.Generator().manual_seed(seed))
    context_set = CustomDataset(context_set, training_dataset_size)

    if ood_points == "svhn":
        ood_dataset = SVHN(root="./data/SVHN", split="test", download=True, transform=test_transform)
    elif ood_points == "cifar10":
        ood_dataset = CIFAR10(root="./data/CIFAR10", train=False, download=True, transform=test_transform)
    else:
        ValueError("Unknown OOD dataset")

    ood_dataset = CustomDataset(ood_dataset, len(test_dataset))
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                #  collate_fn=numpy_collate,
                                 num_workers=num_workers_test,
                                 persistent_workers=persistent_workers_test
                                 )

elif dataset == 'fmnist' or dataset == 'fmnist-224':
    _train_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, download=True)
    input_dim = 28
    num_classes = 10
    dataset_size = 60000
    testset_size = 10000
    if training_dataset_size == 0:
        training_dataset_size = 60000
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size

    exmp_input = jnp.ones([1, input_dim, input_dim, 1])

    DATA_MEANS = (_train_dataset.data / 255.0).mean(axis=(0,1,2)).numpy()
    DATA_STD = (_train_dataset.data / 255.0).std(axis=(0,1,2)).numpy()
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    if not ssm:
        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)[:,:,None]
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img
    else:
        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img.reshape((img.shape[-2]**2, 1))

    test_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])

    train_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])

    if dataset == 'fmnist-224':
        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img

        test_transform_list = [
            transforms.Resize(224),
            transforms.Grayscale(3),
        ]
        train_transform_list = [
            transforms.Resize(224),
            transforms.Grayscale(3),
        ]

        test_transform_list.append(image_to_numpy)
        train_transform_list.append(image_to_numpy)

    _train_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, transform=train_transform, download=True)
    val_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, transform=test_transform, download=True)

    train_dataset, _ = torch.utils.data.random_split(_train_dataset, [training_dataset_size, dataset_size-training_dataset_size], generator=torch.Generator().manual_seed(seed))
    train_dataset = CustomDataset(train_dataset, dataset_size)
    _, validation_dataset = torch.utils.data.random_split(val_dataset, [dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    test_dataset = FashionMNIST(root="./data/fashionMNIST", train=False, transform=test_transform, download=True)

    if context_transform:
        context_transform_list = [
            transforms.Grayscale(1),
            transforms.RandomCrop(input_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.GaussianBlur(kernel_size=(3,3)),
            transforms.GaussianBlur(kernel_size=(3,3), sigma=0.1),
            transforms.RandomSolarize(threshold=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize(28),
        ]
        if dataset == 'fmnist-224':
            context_transform_list.append(transforms.Resize(224))
        context_transform_list.append(image_to_numpy)
        context_transform = transforms.Compose(context_transform_list)
    else:
        context_transform = test_transform

    if context_points == "train":
        context_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, transform=context_transform, download=True)
    elif context_points == "kmnist":
        context_dataset = KMNIST(root="./data/", train=True, transform=context_transform, download=True)
    elif context_points == "mnist":
        context_dataset = MNIST("./data/", train=True, download=True, transform=context_transform)
    elif context_points == "imagenet":
        context_dataset = ImageNet(root="./data/ImageNet", train=True, transform=context_transform, download=True)
    else:
        ValueError("Unknown context dataset")
    
    context_set, _ = torch.utils.data.random_split(context_dataset, [context_dataset_size, 60000 - context_dataset_size], generator=torch.Generator().manual_seed(seed))
    context_set = CustomDataset(context_set, training_dataset_size)
        
    if ood_points == "mnist":
        ood_dataset = MNIST("./data/", train=False, download=True, transform=test_transform)
    elif ood_points == "kmnist":
        ood_dataset = KMNIST(root="./data/", train=False, transform=test_transform, download=True)
    else:
        ValueError("Unknown OOD dataset")

    ood_dataset = CustomDataset(ood_dataset, len(test_dataset))
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                #  collate_fn=numpy_collate,
                                 num_workers=num_workers_test,
                                 persistent_workers=persistent_workers_test
                                 )

elif dataset == 'mnist' or dataset == 'mnist-224':
    _train_dataset = FashionMNIST(root="./data/", train=True, download=True)
    input_dim = 28
    num_classes = 10
    dataset_size = 60000
    testset_size = 10000
    if training_dataset_size == 0:
        training_dataset_size = 60000
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size

    exmp_input = jnp.ones([1, input_dim, input_dim, 1])

    DATA_MEANS = (_train_dataset.data / 255.0).mean(axis=(0,1,2)).numpy()
    DATA_STD = (_train_dataset.data / 255.0).std(axis=(0,1,2)).numpy()
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    if not ssm:
        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)[:,:,None]
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img
    else:
        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img.reshape((img.shape[-2]**2, 1))

    test_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])

    train_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])

    if dataset == 'mnist-224':
        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img

        test_transform_list = [
            transforms.Resize(224),
            transforms.Grayscale(3),
        ]
        train_transform_list = [
            transforms.Resize(224),
            transforms.Grayscale(3),
        ]

        test_transform_list.append(image_to_numpy)
        train_transform_list.append(image_to_numpy)

    _train_dataset = MNIST(root="./data/", train=True, transform=train_transform, download=True)
    val_dataset = MNIST(root="./data/", train=True, transform=test_transform, download=True)

    train_dataset, _ = torch.utils.data.random_split(_train_dataset, [training_dataset_size, dataset_size-training_dataset_size], generator=torch.Generator().manual_seed(seed))
    train_dataset = CustomDataset(train_dataset, dataset_size)
    _, validation_dataset = torch.utils.data.random_split(val_dataset, [dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    test_dataset = MNIST(root="./data/", train=False, transform=test_transform, download=True)

    if context_transform:
        context_transform_list = [
            transforms.Grayscale(1),
            transforms.RandomCrop(input_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.GaussianBlur(kernel_size=(3,3)),
            transforms.GaussianBlur(kernel_size=(3,3), sigma=0.1),
            transforms.RandomSolarize(threshold=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize(28),
        ]
        if dataset == 'mnist-224':
            context_transform_list.append(transforms.Resize(224))
        context_transform_list.append(image_to_numpy)
        context_transform = transforms.Compose(context_transform_list)
    else:
        context_transform = test_transform

    if context_points == "train":
        context_dataset = MNIST("./data/", train=True, download=True, transform=context_transform)
    elif context_points == "kmnist":
        context_dataset = KMNIST(root="./data/", train=True, transform=context_transform, download=True)
    elif context_points == "fmnist":
        context_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, transform=context_transform, download=True)
    elif context_points == "imagenet":
        context_dataset = ImageNet(root="./data/ImageNet", train=True, transform=context_transform, download=True)
    else:
        ValueError("Unknown context dataset")
    
    context_set, _ = torch.utils.data.random_split(context_dataset, [context_dataset_size, 60000 - context_dataset_size], generator=torch.Generator().manual_seed(seed))
    context_set = CustomDataset(context_set, training_dataset_size)

    if ood_points == "fmnist":
        ood_dataset = FashionMNIST(root="./data/fashionMNIST", train=False, transform=test_transform, download=True)
    elif ood_points == "kmnist":
        ood_dataset = KMNIST(root="./data/", train=False, transform=test_transform, download=True)
    else:
        ValueError("Unknown OOD dataset")

    ood_dataset = CustomDataset(ood_dataset, len(test_dataset))
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                #  collate_fn=numpy_collate,
                                 num_workers=num_workers_test,
                                 persistent_workers=persistent_workers_test
                                 )

elif dataset == 'two-moons':
    Path.mkdir(Path('figures/two_moons'), parents=True, exist_ok=True)

    input_dim = 2
    num_classes = 2
    dataset_size = 10000
    training_dataset_size = 200
    batch_size_test = 32
    num_workers = 0
    persistent_workers = False

    x_train, y_train = sklearn_datasets.make_moons(
        n_samples=training_dataset_size, shuffle=True, noise=0.2, random_state=seed
    )
    y_train = np.array(y_train[:, None] == np.arange(2))

    context_set_size = 10000
    x_context_lim = float(re.search('x_context_lim_(.*)', context_points).group(1))

    x_context_min, x_context_max = (
        x_train[:, 0].min() - x_context_lim,
        x_train[:, 0].max() + x_context_lim,
    )
    y_context_min, y_context_max = (
        x_train[:, 1].min() - x_context_lim,
        x_train[:, 1].max() + x_context_lim,
    )

    h_context_x = (x_context_max - x_context_min) / context_set_size ** 0.5
    h_context_y = (y_context_max - y_context_min) / context_set_size ** 0.5

    xx_context, yy_context = np.meshgrid(
        np.arange(x_context_min, x_context_max, h_context_x), np.arange(y_context_min, y_context_max, h_context_y)
    )
    x_context = np.vstack((xx_context.reshape(-1), yy_context.reshape(-1))).T
    y_context = jnp.ones((x_context.shape[0], 2))

    # x_context = jnp.repeat(x_train, 100, axis=0)
    # y_context = jnp.repeat(y_train, 100, axis=0)

    h = 0.25
    test_lim = 1.5
    # test_lim = 4
    # x_min, x_max = x_train[:, 0].min() - test_lim, x_train[:, 0].max() + test_lim
    # y_min, y_max = x_train[:, 1].min() - test_lim, x_train[:, 1].max() + test_lim
    x_min, x_max = (
        x_train[:, 0].min() - test_lim,
        x_train[:, 0].max() + test_lim,
        )
    y_min, y_max = (
        x_train[:, 1].min() - test_lim,
        x_train[:, 1].max() + test_lim,
        )
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
        )
    _x_test = np.vstack((xx.reshape(-1), yy.reshape(-1))).T
    _y_test = jnp.ones((_x_test.shape[0], 2))

    permutation = np.random.permutation(_x_test.shape[0])
    x_test = _x_test[permutation]
    y_test = _y_test[permutation]

    permutation_inv = np.argsort(permutation)

    h_wide = 0.25
    test_lim_wide = 1.5
    # test_lim_wide = 4
    x_wide_min, x_wide_max = (
        x_train[:, 0].min() - test_lim_wide,
        x_train[:, 0].max() + test_lim_wide,
    )
    y_wide_min, y_wide_max = (
        x_train[:, 1].min() - test_lim_wide,
        x_train[:, 1].max() + test_lim_wide,
    )
    xx_wide, yy_wide = np.meshgrid(
        np.arange(x_wide_min, x_wide_max, h_wide), np.arange(y_wide_min, y_wide_max, h_wide)
    )
    x_test_wide = np.vstack((xx_wide.reshape(-1), yy_wide.reshape(-1))).T
    y_test_wide = jnp.ones((x_test_wide.shape[0], 2))

    testset_size = x_test.shape[0]

    exmp_input = x_test[0:1]

    train_dataset = TwoMoons(x_train.shape[0], x_train, y_train)
    train_dataset = CustomDataset(train_dataset, dataset_size)
    context_set = TwoMoons(x_context.shape[0], x_context, y_context)
    context_set = CustomDataset(context_set, dataset_size)
    validation_dataset = TwoMoons(x_train.shape[0], x_train, y_train)
    test_dataset = TwoMoons(x_test.shape[0], x_test, y_test)

    ood_dataset = context_set
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=context_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                #  collate_fn=numpy_collate,
                                 num_workers=num_workers_test,
                                 persistent_workers=persistent_workers_test
                                 )

elif dataset == 'snelson' or dataset == 'oat1d':
    input_dim = 1
    num_classes = 1
    n_context = 1000000
    dataset_size = 1000000
    n_test = 1000
    batch_size_test = 1000
    num_workers = 0
    persistent_workers = False


    if dataset == 'snelson':
        Path.mkdir(Path('figures/snelson'), parents=True, exist_ok=True)

        def _load_toydata(filename):
            try:
                with open(f"data/snelson/{filename}", "r") as f:
                    return np.array(
                        [float(i) for i in f.read().strip().split("\n")]
                    )
            except Exception as e:
                print(
                    f"Error: {e.args[0]}\n\nWorking directory needs to be set to repository root."
                )

        training_dataset_size = 148
        x_test_lim = 6
        # noise_std = 0.1

        x_train = _load_toydata("train_inputs")
        y_train = _load_toydata("train_outputs")
        
        mask = ((x_train < 1.5) | (x_train > 3)).flatten()
        x_train = x_train[mask]
        y_train = y_train[mask]
        
        idx = np.argsort(x_train)
        x_train = x_train[idx]
        y_train = y_train[idx]

        x_train = ((x_train - x_train.mean(0)) / x_train.std(0))[:, None]
        y_train = ((y_train - y_train.mean(0)) / y_train.std(0))[:, None]

        assert x_train.shape[0] == training_dataset_size

        x_test = np.linspace(-x_test_lim, x_test_lim, n_test)[:, None]
        y_test = np.zeros_like(x_test)

        x_context_lim = float(re.search('x_context_lim_(.*)', context_points).group(1))
        # x_context = np.linspace(-x_context_lim, x_context_lim, n_context)[:, None]
        x_context = np.array(jax.random.uniform(key=main_rng, shape=(n_context,), minval=-x_context_lim, maxval=x_context_lim, dtype=float)[:, None])
        y_context = np.zeros_like(x_context)

    elif dataset == 'oat1d':
        Path.mkdir(Path('figures/oat1d'), parents=True, exist_ok=True)

        training_dataset_size = 80
        x_test_lim = 6
        # noise_std = 0.001

        x_train1 = np.linspace(-7, -4, 40)
        y_train1 = np.sin(x_train1 * np.pi * 0.5 - 2.5) * 4.7 - 1.2

        x_train2 = np.linspace(3,8,40)
        y_train2 = np.sin(x_train2 * np.pi * 0.58 - 0.5) * 1.6 - 2.7

        x = np.concatenate([x_train1, x_train2], 0)[:, None]
        x_mean = x.mean()
        x_std = x.std()
        x = (x - x_mean) / x_std
        x_train = x

        assert x_train.shape[0] == training_dataset_size

        y = np.concatenate([y_train1, y_train2], 0)[:, None]
        y_mean = y.mean()
        y_std = y.std()
        y_train = (y - y_mean) / y_std

        x_test = np.linspace(-x_test_lim, x_test_lim, n_test)[:, None]
        y_test = np.zeros_like(x_test)

        x_context_lim = float(re.search('x_context_lim_(.*)', context_points).group(1))
        # x_context = np.linspace(-x_context_lim, x_context_lim, n_context)[:, None]
        x_context = np.array(jax.random.uniform(key=main_rng, shape=(n_context,), minval=-x_context_lim, maxval=x_context_lim, dtype=float)[:, None])
        y_context = np.zeros_like(x_context)

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    exmp_input = x_train[0:1]

    testset_size = x_test.shape[0]

    train_dataset = Regression(x_train.shape[0], x_train, y_train)
    train_dataset = CustomDataset(train_dataset, dataset_size)
    context_set = Regression(x_context.shape[0], x_context, y_context)
    context_set = CustomDataset(context_set, dataset_size)
    validation_dataset = Regression(x_train.shape[0], x_train, y_train)
    test_dataset = Regression(x_test.shape[0], x_test, y_test)

    ood_dataset = context_set
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=context_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                #  collate_fn=numpy_collate,
                                 num_workers=num_workers_test,
                                 persistent_workers=persistent_workers_test
                                 )

elif 'offline_rl' in dataset:
    import gym
    import d4rl

    # env_dict = gym.envs.registration.registry.env_specs.copy()
    # for env in env_dict:
    #     for type in ["door", "hammer", "pen", "relocate"]:
    #         for version in ["0", "1"]:
    #             try:
    #                 if f"{type}-v{version}" in env:
    #                     print("Remove {} from registry".format(env))
    #                     del gym.envs.registration.registry.env_specs[env]
    #                 if f"{type}-human-v{version}" in env:
    #                     print("Remove {} from registry".format(env))
    #                     del gym.envs.registration.registry.env_specs[env]
    #                 if f"{type}-human-v{version}" in env:
    #                     print("Remove {} from registry".format(env))
    #                     del gym.envs.registration.registry.env_specs[env]
    #                 if f"{type}-human-longhorizon-v{version}" in env:
    #                     print("Remove {} from registry".format(env))
    #                     del gym.envs.registration.registry.env_specs[env]
    #                 if f"{type}-expert-v{version}" in env:
    #                     print("Remove {} from registry".format(env))
    #                     del gym.envs.registration.registry.env_specs[env]
    #                 if f"{type}-cloned-v{version}" in env:
    #                     print("Remove {} from registry".format(env))
    #                     del gym.envs.registration.registry.env_specs[env]
    #             except:
    #                 print(f"Exception at {type}-{version}")

    env_name = re.search('offline_rl_(.*)', dataset).group(1)
    env = gym.make(env_name)
    rl_dataset = d4rl.qlearning_dataset(env)
    observations = rl_dataset['observations']  # An N x dim_observation Numpy array of observations
    actions = rl_dataset['actions']
    rewards = rl_dataset['rewards']
    next_observations = rl_dataset['next_observations']
    next_delta = next_observations - observations

    # concatenate for training
    x = np.concatenate((observations, actions), axis=1)
    y = np.concatenate((next_delta, rewards[:, None]), axis=1)

    input_dim = x.shape[-1]
    num_classes = y.shape[-1]
    frac_test = 0.2
    training_dataset_size = int(x.shape[0] * (1 - frac_test))
    # dataset_size = training_dataset_size
    dataset_size = 1000000
    n_context = 1000000
    n_test = int(x.shape[0] * frac_test)
    batch_size_test = 1000
    num_workers = 0
    persistent_workers = False
    exmp_input = x[0:1]

    ### technically, we should normalize using only the train split, but we want a global mean and std for online evaluation later on
    x_mean = x.mean(0)
    x_std = x.std(0)
    
    normalize = True  # TODO: remove hardcoding
    if normalize:
        x = (x - x_mean) / x_std

    y_mean = y.mean(0)
    y_std = y.std(0)

    # split the train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=frac_test, random_state=seed)

    assert x_train.shape[0] == training_dataset_size

    x_context_min = x_train.min(0)
    x_context_max = x_train.max(0)
    x_context = np.array(jax.random.uniform(key=main_rng, shape=(n_context,input_dim), minval=x_context_min, maxval=x_context_max, dtype=float))
    y_context = np.zeros_like(x_context)

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    testset_size = x_test.shape[0]

    train_dataset = Regression(x_train.shape[0], x_train, y_train)
    train_dataset = CustomDataset(train_dataset, dataset_size)
    context_set = Regression(x_context.shape[0], x_context, y_context)
    context_set = CustomDataset(context_set, dataset_size)
    validation_dataset = Regression(x_train.shape[0], x_train, y_train)
    test_dataset = Regression(x_test.shape[0], x_test, y_test)

    ood_dataset = context_set
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=context_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                #  collate_fn=numpy_collate,
                                 num_workers=num_workers_test,
                                 persistent_workers=persistent_workers_test
                                 )

# elif dataset in ['aptos', 'cassava', 'melanoma']:
#     # TR: Untested:
#     input_dim = 224
#     num_classes = 10
#     dataset_size = 50000
#     testset_size = 10000
#     if training_dataset_size == 0:
#         training_dataset_size = 50000
#     validation_dataset_size = dataset_size - training_dataset_size
#     batch_size_test = batch_size
#     exmp_input = jnp.array([1, input_dim, input_dim, 3])

#     from fspace.datasets import get_dataset

#     train_dataset, _, test_dataset = get_dataset(dataset, root=os.environ.get('DATADIR'), seed=seed)
#     validation_dataset = test_dataset

#     def image_to_numpy_context(img):
#         img = np.array(img, dtype=np.float32)
#         img = (img / 255. - DATA_MEANS_CONTEXT) / DATA_STD_CONTEXT
#         return img
    
#     if context_transform:
#         context_transform_list = [
#             transforms.RandomCrop(input_dim, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.GaussianBlur(kernel_size=(3,3)),
#             # transforms.GaussianBlur(kernel_size=(3,3), sigma=0.1),
#             transforms.RandomSolarize(threshold=0.5),
#             transforms.ColorJitter(brightness=0.5, contrast=0.5),
#             transforms.RandomResizedCrop(224),
#         ]

#         context_transform_list.append(image_to_numpy_context)
#         context_transform = transforms.Compose(context_transform_list)
#     else:
#         if context_points == "imagenet":
#             context_transform_list = [
#                 transforms.RandomResizedCrop(224),
#                 # transforms.Resize(32),
#                 # transforms.Resize(224),
#             ]
#             context_transform_list.append(image_to_numpy_context)
#             context_transform = transforms.Compose(context_transform_list)
#         else:
#             context_transform = None

#     if context_points == "train":
#         context_dataset = train_dataset
#     elif context_points == "imagenet":
#         context_dataset = ImageNet(root="./data/ImageNet", transform=context_transform)
#     else:
#         ValueError("Unknown context dataset")
    
#     full_context_dataset_size = len(context_dataset)
#     context_set, _ = torch.utils.data.random_split(context_dataset, [context_dataset_size, full_context_dataset_size - context_dataset_size], generator=torch.Generator().manual_seed(seed))
#     context_set = CustomDataset(context_set, training_dataset_size)

elif dataset == 'adult':
    if sensitive_attribute == 'sex':
        categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
        numerical_features = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        input_dim = 102 
        dataset_size = 45222 - 15060
        num_classes = 2
        if training_dataset_size == 0:
            training_dataset_size = 45222 - 15060
        testset_size = 15060 
        validation_dataset_size = dataset_size - training_dataset_size
        batch_size_test = batch_size

        df = pd.read_csv('data/adult/adult.csv', na_values='?')
        df.dropna(inplace=True)
        df.drop(['fnlwgt'], axis=1, inplace=True)

        y = df['income'].apply(lambda x: 1 if x == '>50K' else 0).values
        del df['income']
        df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
        s = df['gender'].values
        df[numerical_features] = StandardScaler().fit_transform(df[numerical_features])
        df = df[[c for c in df if c not in ['gender']] + ['gender']]
        df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')
        x = np.array(df.values, dtype=np.float32)

        # split into train and test 
        x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(x, y, s, test_size=testset_size, random_state=seed)
        if validation_dataset_size > 0:
            x_train, x_val, y_train, y_val, s_train, s_val = train_test_split(x_train, y_train, s_train, test_size=validation_dataset_size, random_state=seed)
        else:
            x_val, y_val, s_val = x_train, y_train, s_train

        train_dataset = TabularDataset(x_train, y_train, s_train)
        validation_dataset = TabularDataset(x_val, y_val, s_val)
        test_dataset = TabularDataset(x_test, y_test, s_test)
        if context_points == "fairtrain":
            context_set = create_sensitive_context_set(train_dataset)
        elif context_points == "train":
            context_set = train_dataset
        else:
            raise ValueError("context_points must be either fairtrain or train")
        
        ood_dataset = context_set
        ood_loader = data.DataLoader(ood_dataset,
                                    batch_size=context_batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    collate_fn=numpy_collate,
                                    num_workers=num_workers_test,
                                    persistent_workers=persistent_workers_test
                                    )

        '''
        meta = json.load(open("./data/adult/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], names=column_names, skipinitialspace=True,
                            na_values=meta["na_values"])
        test = pd.read_csv(meta["test_path"], header=0, names=column_names, skipinitialspace=True,
                           na_values=meta["na_values"])

        train.dropna(inplace=True)
        test.dropna(inplace=True)

        train = pd.concat([train, test], axis=0)

        train['income'] = train['income'].apply(lambda x: 0 if x=='<=50K' else 1) # target label
        train.drop(['fnlwgt'], axis=1, inplace=True) # useless feature
        train['sex'] = train['sex'].apply(lambda x: 0 if x=='Male' else 1) # sensitive attribute

        #for col in train: # normalize non-categorical features to zero mean and unit variance
        #    if col not in meta['categorical_feat']:
        #        train[col] = (train[col] - train[col].mean()) / train[col].std()

        for col in meta['categorical_feat']: # one-hot encoding
            train = pd.concat([train, pd.get_dummies(train[col], prefix=col)],axis=1)
            train.drop([col],axis=1, inplace=True)

        # convert bool to int
        for col in train.columns:
            if train[col].dtype == bool:
                train[col] = train[col].astype(int)

        # move sensitive attribute to the last column to work with create_sensitive_context_set
        train = train[[c for c in train if c not in ['income', 'sex']] + ['income', 'sex']]

        X_train = np.array(train.drop(['income'], axis=1).values, dtype=np.float32)
        y_train = train['income'].values
        s_train = train['sex'].values

        train_dataset = TabularDataset(X_train, y_train, s_train)
        val_dataset = TabularDataset(X_train, y_train, s_train)
        _, validation_dataset = data.random_split(val_dataset, [dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))
        train_dataset, test_dataset = data.random_split(train_dataset, [training_dataset_size, testset_size], generator=torch.Generator().manual_seed(seed))
        train_dataset = TabularDataset(train_dataset.dataset.X[train_dataset.indices], train_dataset.dataset.Y[train_dataset.indices], train_dataset.dataset.S[train_dataset.indices])
        test_dataset = TabularDataset(test_dataset.dataset.X[test_dataset.indices], test_dataset.dataset.Y[test_dataset.indices], test_dataset.dataset.S[test_dataset.indices])

        if context_points == "fairtrain":
            context_set = create_sensitive_context_set(train_dataset)
        elif context_points == "train":
            context_set = train_dataset
        else:
            raise ValueError("context_points must be either fairtrain or train")

        ood_dataset = context_set
        ood_loader = data.DataLoader(ood_dataset,
                                    batch_size=context_batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    collate_fn=numpy_collate,
                                    num_workers=num_workers_test,
                                    persistent_workers=persistent_workers_test
                                    )
        '''

elif dataset == 'compas':
    categorical_features = ["age_cat","c_charge_degree","c_charge_desc"]
    features_to_keep = ["sex","age","age_cat","race","juv_fel_count","juv_misd_count","juv_other_count","priors_count","c_charge_degree","c_charge_desc","two_year_recid"]
    numerical_features = ["age","juv_fel_count","juv_misd_count","juv_other_count","priors_count"]
    input_dim = 401
    num_classes = 2
    dataset_size = 6172 - 1233
    if training_dataset_size == 0:
        training_dataset_size = 6172 - 1233
    testset_size = 1233 # NOT USED
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size


    df = pd.read_csv('data/compas/compas-scores-two-years.csv')

    df['race'] = df['race'].apply(lambda race: 1 if race == 'Caucasian' else 0)
    df['sex'] = df['sex'].apply(lambda sex: 1 if sex == 'Male' else 0)
    df = df[
            (df['days_b_screening_arrest'] <= 30) &
            (df['days_b_screening_arrest'] >= -30) &
            (df['is_recid'] != -1) &
            (df['c_charge_degree'] != 'O') &
            (df['score_text'] != 'N/A')
        ].copy()
    df = df[features_to_keep]
    y = df['two_year_recid'].values
    del df['two_year_recid']
    s = df['race'].values

    df[numerical_features] = StandardScaler().fit_transform(df[numerical_features])
    df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')

    x = np.array(df.values, dtype=np.float32)

    x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(x, y, s, test_size=testset_size, random_state=seed)
    if validation_dataset_size > 0:
        x_train, x_val, y_train, y_val, s_train, s_val = train_test_split(x_train, y_train, s_train, test_size=validation_dataset_size, random_state=seed)
    else:
        x_val, y_val, s_val = x_train, y_train, s_train

    train_dataset = TabularDataset(x_train, y_train, s_train)
    validation_dataset = TabularDataset(x_val, y_val, s_val)
    test_dataset = TabularDataset(x_test, y_test, s_test)
    
    if context_points == "fairtrain":
        context_set = create_sensitive_context_set(train_dataset)
    elif context_points == "train":
        context_set = train_dataset
    else:
        raise ValueError("context_points must be either fairtrain or train")
    
    ood_dataset = context_set
    ood_loader = data.DataLoader(ood_dataset,
                                batch_size=context_batch_size,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=numpy_collate,
                                num_workers=num_workers_test,
                                persistent_workers=persistent_workers_test
                                )


    '''
    meta = json.load(open("./data/compas/meta.json"))
    meta["categorical_feat"] = meta["categorical_feat"].split(",")

    raw_data = pd.read_csv(meta["train_path"])

    df = raw_data[
        (raw_data['days_b_screening_arrest'] <= 30) &
        (raw_data['days_b_screening_arrest'] >= -30) &
        (raw_data['is_recid'] != -1) &
        (raw_data['c_charge_degree'] != 'O') &
        (raw_data['score_text'] != 'N/A')
    ].copy()

    df['race'] = df['race'].apply(lambda race: 1 if race == 'Caucasian' else 0)
    df['two_year_recid'] = df['two_year_recid'].apply(lambda x: 1 if x == 0 else 0) # label 
    df = df[meta["features_to_keep"].split(",")]
    for col in meta['categorical_feat']:
        if col == 'race':
            continue
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)],axis=1)
        df.drop([col],axis=1, inplace=True)

    df = df[[c for c in df if c not in ['two_year_recid', 'race']] + ['two_year_recid', 'race']] # move sensitive attribute to the last column to work with create_sensitive_context_set

    # turn 'True' to 1 and 'False' to 0 for df 
    df = df.replace(True, 1)
    df = df.replace(False, 0)

    X = np.array(df.drop(columns=['two_year_recid']).values, dtype=np.float32)
    Y = df['two_year_recid'].values
    S = df['race'].values

    dataset = TabularDataset(X, Y, S)
    val_dataset = TabularDataset(X, Y, S)
    _, validation_dataset = data.random_split(val_dataset, [dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))
    train_dataset, test_dataset = data.random_split(dataset, [training_dataset_size, dataset_size-training_dataset_size], generator=torch.Generator().manual_seed(0))
    train_dataset = TabularDataset(train_dataset.dataset.X[train_dataset.indices], train_dataset.dataset.Y[train_dataset.indices], train_dataset.dataset.S[train_dataset.indices])
    test_dataset = TabularDataset(test_dataset.dataset.X[test_dataset.indices], test_dataset.dataset.Y[test_dataset.indices], test_dataset.dataset.S[test_dataset.indices])

    if context_points == "fairtrain":
        context_set = create_sensitive_context_set(train_dataset)
    elif context_points == "train":
        context_set = train_dataset
    else:
        raise ValueError("context_points must be either fairtrain or train")

    ood_dataset = context_set
    ood_loader = data.DataLoader(ood_dataset,
                                batch_size=context_batch_size,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=numpy_collate,
                                num_workers=num_workers_test,
                                persistent_workers=persistent_workers_test
                                )
    '''

elif dataset == 'celeba': # hair color as target and gender as sensitive attribute
    input_dim = 224
    num_classes = 2
    num_sensitive = 2
    if not validation_training:
        dataset_size = 162770
    else:
        dataset_size = 19867
    testset_size = 19962
    gender_idx = 20
    hair_color_idx = 9
    if training_dataset_size == 0:
        training_dataset_size = dataset_size
    validation_dataset_size = 19867
    train_validation_dataset_size = int(val_train_frac * validation_dataset_size)
    validation_validation_dataset_size = validation_dataset_size - train_validation_dataset_size
    batch_size_test = batch_size

    DATA_MEANS = np.array([0.485, 0.456, 0.406]) # ImageNet means
    DATA_STD = np.array([0.229, 0.224, 0.225]) # ImageNet stds
    #DATA_MEANS = [0.5063, 0.4258, 0.3832] # Actual means
    #DATA_STD = [0.3107, 0.2904, 0.2897] # Actual stds
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32).transpose(1,2,0)
        return img
    def image_to_numpy_context(img):
        img = np.array(img, dtype=np.float32).transpose(1,2,0)
        return img

    ## OPTION 1 ##
    #train_transform = transforms.Compose([
    #    transforms.Resize((256,256)),
    #    transforms.RandomCrop(224),
    #    transforms.RandomHorizontalFlip(),
    #    image_to_numpy
    #])
    #test_transform = transforms.Compose([
    #    transforms.Resize((224,224)),
    #    #transforms.ToTensor(),
    #    #transforms.Normalize(DATA_MEANS, DATA_STD),
    #    image_to_numpy
    #])

    ## OPTION 2 ##
    #train_transform = transforms.Compose([
    #    transforms.Resize((224,224)),
    #    transforms.RandomCrop(224),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #    transforms.Normalize(DATA_MEANS, DATA_STD),
    #    to_numpy
    #])
    #test_transform = transforms.Compose([
    #    transforms.Resize((224,224)),
    #    transforms.ToTensor(),
    #    transforms.Normalize(DATA_MEANS, DATA_STD),
    #    to_numpy
    #])

    test_transform_list = [
        #transforms.CenterCrop(178),
        #transforms.ToTensor(),
        #transforms.Normalize(mean=DATA_MEANS, std=DATA_STD),
        #transforms.Resize(input_dim)

        transforms.CenterCrop(178),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    train_transform_list = [
        #transforms.CenterCrop(178),
        #transforms.ToTensor(),
        #transforms.Normalize(mean=DATA_MEANS, std=DATA_STD),
        #transforms.Resize(input_dim)

        transforms.RandomResizedCrop(
            (224,224),
            scale=(0.7, 1.0),
            ratio=(1.0, 1.3333333333333333),
            interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    test_transform_list.append(image_to_numpy)
    train_transform_list.append(image_to_numpy)

    test_transform = transforms.Compose(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    data_dir = './data'

    if not validation_training:
        train_dataset = CelebADataset(root=data_dir, split='train', target_type='attr', transform=train_transform, target_transform=None, download=False, want_sensitive=(context_points == "fairtrain"))
        validation_dataset = CelebADataset(root=data_dir, split='train', target_type='attr', transform=train_transform, target_transform=None, download=False, want_sensitive=True)
    else:
        _train_dataset = CelebADataset(root=data_dir, split='valid', target_type='attr', transform=train_transform, target_transform=None, download=False, want_sensitive=(context_points == "fairtrain"))
        train_dataset, _ = torch.utils.data.random_split(_train_dataset, [train_validation_dataset_size, validation_dataset_size-train_validation_dataset_size], generator=torch.Generator().manual_seed(seed))

        _validation_dataset = CelebADataset(root=data_dir, split='valid', target_type='attr', transform=test_transform, target_transform=None, download=False, want_sensitive=(context_points == "fairtrain"))
        if val_train_frac == 1:
            validation_dataset, _ = torch.utils.data.random_split(_validation_dataset, [train_validation_dataset_size,  validation_dataset_size-train_validation_dataset_size], generator=torch.Generator().manual_seed(seed))
        else:
            _, validation_dataset = torch.utils.data.random_split(_validation_dataset, [train_validation_dataset_size,  validation_dataset_size-train_validation_dataset_size], generator=torch.Generator().manual_seed(seed))
        # val_dataset = CelebADataset(root=data_dir, split='valid', target_type='attr', transform=train_transform, target_transform=None, download=False, want_sensitive=True)
    test_dataset = CelebADataset(root=data_dir, split='test', target_type='attr', transform=test_transform, target_transform=None, download=False, want_sensitive=True)

    #train_dataset, _ = torch.utils.data.random_split(_train_dataset, [training_dataset_size, dataset_size-training_dataset_size], generator=torch.Generator().manual_seed(seed))
    # _, validation_dataset = torch.utils.data.random_split(val_dataset, [dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    exmp_input = train_dataset[0][0][None, :]

    DATA_MEANS_CONTEXT = DATA_MEANS
    DATA_STD_CONTEXT = DATA_STD

    if context_transform:
        context_transform_list = [ 
        ]
        context_transform_list.append(image_to_numpy_context)
        context_transform = transforms.Compose(context_transform_list)
    else:
        context_transform = test_transform

    if context_points == "fairtrain":
        context_set = create_sensitive_context_set(train_dataset)
    else:
        context_set = train_dataset
    
    ood_dataset = train_dataset
    ood_loader = data.DataLoader(ood_dataset,
                                batch_size=128,
                                shuffle=False,
                                drop_last=False,
                                # collate_fn=numpy_collate,
                                num_workers=num_workers_test,
                                persistent_workers=persistent_workers_test
                                )

elif dataset == 'waterbirds':
    input_dim = 224
    num_classes = 2
    num_sensitive = 2
    if not validation_training:
        dataset_size = 4795
    else:
        dataset_size = 1199
    testset_size = 5794

    if training_dataset_size == 0:
        training_dataset_size = 4795
    validation_dataset_size = 1199
    train_validation_dataset_size = int(val_train_frac * validation_dataset_size)
    validation_validation_dataset_size = validation_dataset_size - train_validation_dataset_size
    batch_size_test = batch_size

    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32).transpose(1,2,0)
        return img
    def image_to_numpy_context(img):
        img = np.array(img, dtype=np.float32).transpose(1,2,0)
        return img

    scale = 256.0/224.0

    test_transform_list = [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(input_dim),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    train_transform_list = [
        transforms.RandomResizedCrop(
            (input_dim,input_dim),
            scale = (0.7, 1.0),
            ratio = (0.75, 1.3333333333333333),
            interpolation = 2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    test_transform_list.append(image_to_numpy)
    train_transform_list.append(image_to_numpy)

    test_transform = transforms.Compose(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    data_dir = './data/waterbirds'

    if not validation_training:
        train_dataset = Waterbirds(root=data_dir, split='train', transform=train_transform, download=False, want_sensitive=(context_points == "fairtrain"))
        validation_dataset = Waterbirds(root=data_dir, split='train', transform=test_transform, download=False, want_sensitive=True)
    else:
        idx = random_np.permutation(validation_dataset_size)
        train_dataset = Waterbirds(root=data_dir, split='val', transform=train_transform, download=False, want_sensitive=(context_points == "fairtrain"), val_split='val_train', val_train_frac=val_train_frac, idx=idx)
        if val_train_frac == 1:
            validation_dataset = Waterbirds(root=data_dir, split='val', transform=test_transform, download=False, want_sensitive=(context_points == "fairtrain"), val_split='val_train', val_train_frac=val_train_frac, idx=idx)
        else:
            validation_dataset = Waterbirds(root=data_dir, split='val', transform=test_transform, download=False, want_sensitive=(context_points == "fairtrain"), val_split='val_val', val_train_frac=val_train_frac, idx=idx)
        # validation_dataset = Waterbirds(root=data_dir, split='val', transform=test_transform, download=False, want_sensitive=True)
    test_dataset = Waterbirds(root=data_dir, split='test', transform=test_transform, download=False, want_sensitive=True)

    exmp_input = train_dataset[0][0][None, :]

    if context_transform:
        context_transform_list = [ 
        ]
        context_transform_list.append(image_to_numpy_context)
        context_transform = transforms.Compose(context_transform_list)
    else:
        context_transform = test_transform

    if context_points == "fairtrain":
        context_set = create_sensitive_context_set(train_dataset)
    else:
        context_set = train_dataset
    
    ood_dataset = train_dataset
    ood_loader = data.DataLoader(ood_dataset,
                                batch_size=128,
                                shuffle=False,
                                drop_last=False,
                                # collate_fn=numpy_collate,
                                num_workers=num_workers_test,
                                persistent_workers=persistent_workers_test
                                )  

elif dataset == 'mnli':
    task = "mnli"
    input_dim = 128*3
    num_classes = 3
    num_sensitive = 2
    batch_size_test = batch_size 
    if not validation_training:
        dataset_size = 412349
    else:
        dataset_size = 82462
    testset_size = 123712
    training_dataset_size = 206175
    validation_dataset_size = 82462

    data_dir = './data'
    if not validation_training:
        train_dataset = MultiNLI(root=data_dir, split='train', transform=None, want_sensitive=(context_points == "fairtrain"))
        validation_dataset = MultiNLI(root=data_dir, split='train', transform=None, want_sensitive=True)
    else:
        train_dataset = MultiNLI(root=data_dir, split='val', transform=None, want_sensitive=(context_points == "fairtrain"))
        validation_dataset = MultiNLI(root=data_dir, split='val', transform=None, want_sensitive=True)
    test_dataset = MultiNLI(root=data_dir, split='test', transform=None, want_sensitive=True)

    exmp_input = train_dataset[0][0][None, :]

    if context_points == "fairtrain":
        context_set = create_sensitive_context_set(train_dataset)
    else:
        context_set = train_dataset

    ood_dataset = train_dataset
    ood_loader = data.DataLoader(ood_dataset,
                                batch_size=128,
                                shuffle=False,
                                drop_last=False,
                                # collate_fn=numpy_collate,
                                num_workers=num_workers_test,
                                persistent_workers=persistent_workers_test
                                )  

elif dataset == 'civilcomments':
    raise ValueError("CivilComments dataset not supported yet.")

elif dataset == 'utkface': # discretized age as target, race/sex as sensitive attribute
    input_dim = 3*200*200
    num_classes = 3
    dataset_size = 22013 if sensitive_attribute == 'race' else 23705
    testset_size = 8675 if sensitive_attribute == 'race' else 9778
    if training_dataset_size == 0:
        training_dataset_size = dataset_size
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size

    #DATA_MEANS = np.array([0.485, 0.456, 0.406]) # ImageNet means
    #DATA_STD = np.array([0.229, 0.224, 0.225]) # ImageNet stds
    DATA_MEANS = np.array([0.5960, 0.4573, 0.3921]) # Actual means
    DATA_STD = np.array([0.2586, 0.2314, 0.2275]) # Actual stds
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img
    
    test_transform = transforms.Compose([
        #transforms.Resize(64), 
        image_to_numpy
    ])
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        #transforms.Resize(64), 
        image_to_numpy
    ])

    train_dataset = UTKFace(root='./data/utkface', split='train', target='age', sensitive=sensitive_attribute, transform=train_transform, want_sensitive=(context_points == "fairtrain"))
    test_dataset = UTKFace(root='./data/utkface', split='test', target='age', sensitive=sensitive_attribute, transform=test_transform, want_sensitive=True)
    _, validation_dataset = data.random_split(train_dataset, [dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    def image_to_numpy_context(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img

    ood_dataset = UTKFace(root='./data/utkface', split='test', target='age', sensitive=sensitive_attribute, transform=test_transform)
    ood_loader = data.DataLoader(ood_dataset,
                                batch_size=128,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=numpy_collate,
                                num_workers=num_workers_test,
                                persistent_workers=persistent_workers_test
                                )
    
    if context_points == "fairtrain":
        context_set = create_sensitive_context_set(train_dataset)
    elif context_points == "train":
        context_set = train_dataset
    else:
        raise ValueError("context_points must be either fairtrain or train")

else:
    raise ValueError("Dataset not found.")
    
if dataset == "two-moons" or dataset == "snelson" or dataset == "oat1d" or "offline_rl" in dataset:
    num_workers_train = 0
    num_workers_test = 0
    persistent_workers_train = False
    persistent_workers_test = False
    pin_memory = False
elif dataset == "adult" or dataset == "compas":
    num_workers_train = 1 
    num_workers_test = 1
    persistent_workers_train = True
    persistent_workers_test = True
    pin_memory = False
else:
    num_workers_train = 0
    persistent_workers_train = False
    pin_memory = False

if context_points == "imagenet" or dataset == "two-moons" or dataset == "snelson" or dataset == "oat1d" or "offline_rl" in dataset:
    num_workers_context = 0
    persistent_workers_context = False
    pin_memory = False
elif dataset == "adult" or dataset == "compas":
    num_workers_context = 1 
    persistent_workers_context = False
    pin_memory = False
elif dataset == "celeba" or dataset == "waterbirds":
    num_workers_context = 0
    persistent_workers_context = False
    pin_memory = False
else:
    num_workers_context = 8
    persistent_workers_context = True
    pin_memory = False

print(f"num_workers_train: {num_workers_train}")
print(f"persistent_workers_context: {persistent_workers_context}")

if group_dro or dataset == "celeba" or dataset == "waterbirds" or dataset == "mnli":

    if dataset == "waterbirds":
        gdro_weights = train_dataset.group_proportions
        num_groups = train_dataset.num_groups

        train_dataset_group_proportions = train_dataset.group_proportions
        validation_dataset_group_proportions = validation_dataset.group_proportions

        if data_balancing:
            init_group_weights = train_dataset.group_weights[np.array(train_dataset.group, dtype=int)]

            # init_group_weights = train_dataset.group_weights[train_dataset.sensitive]

            gdro_sampler = data.WeightedRandomSampler(init_group_weights, len(train_dataset), replacement=True, generator=torch.Generator().manual_seed(seed))
            train_loader = data.DataLoader(train_dataset, 
                                        batch_size=batch_size,
                                        sampler=gdro_sampler,
                                        drop_last=True,
                                        #    collate_fn=numpy_collate,
                                        num_workers=num_workers_train,
                                        pin_memory=pin_memory,
                                        persistent_workers=persistent_workers_train,
                                        )
        else:
            train_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        #    collate_fn=numpy_collate,
                                        num_workers=num_workers_train,
                                        pin_memory=pin_memory,
                                        persistent_workers=persistent_workers_train,
                                        )

    elif dataset == "celeba":
        num_groups = 4

        if not validation_training:
            gdro_weights = train_dataset.group_proportions
            train_dataset_group_proportions = train_dataset.group_proportions
            validation_dataset_group_proportions = validation_dataset.group_proportions
        else:
            gdro_weights = train_dataset.dataset.group_proportions
            train_dataset_group_proportions = train_dataset.dataset.group_proportions
            validation_dataset_group_proportions = validation_dataset.dataset.group_proportions

        if data_balancing:
            if not validation_training:
                init_group_weights = train_dataset.group_weights[np.array(train_dataset.group[train_dataset.indices], dtype=int)]
                # init_group_weights = train_dataset.dataset.group_weights[np.array(train_dataset.dataset.group, dtype=int)]
            else:
                init_group_weights = train_dataset.dataset.group_weights[np.array(train_dataset.dataset.group[train_dataset.indices], dtype=int)]

            # init_group_weights = train_dataset.dataset.group_weights[np.array(train_dataset.dataset.sensitive[train_dataset.indices], dtype=int)]
            # # init_group_weights = train_dataset.dataset.group_weights[np.array(train_dataset.dataset.sensitive, dtype=int)]

            gdro_sampler = data.WeightedRandomSampler(init_group_weights, len(train_dataset), replacement=True, generator=torch.Generator().manual_seed(seed))
            train_loader = data.DataLoader(train_dataset, 
                                        batch_size=batch_size,
                                        sampler=gdro_sampler,
                                        drop_last=True,
                                        #    collate_fn=numpy_collate,
                                        num_workers=num_workers_train,
                                        pin_memory=pin_memory,
                                        persistent_workers=persistent_workers_train,
                                        )
        else:
            train_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        #    collate_fn=numpy_collate,
                                        num_workers=num_workers_train,
                                        pin_memory=pin_memory,
                                        persistent_workers=persistent_workers_train,
                                        )
    elif dataset == 'mnli':
        if data_balancing:
            init_group_weights = train_dataset.group_weights[np.array(train_dataset.group, dtype=int)]

            gdro_weights = train_dataset.group_proportions
            num_groups = train_dataset.num_groups

            train_dataset_group_proportions = train_dataset.group_proportions
            validation_dataset_group_proportions = validation_dataset.group_proportions

            gdro_sampler = data.WeightedRandomSampler(init_group_weights, len(train_dataset), replacement=True, generator=torch.Generator().manual_seed(seed))
            train_loader = data.DataLoader(train_dataset, 
                                        batch_size=batch_size,
                                        sampler=gdro_sampler,
                                        drop_last=True,
                                        #    collate_fn=numpy_collate,
                                        num_workers=num_workers_train,
                                        pin_memory=pin_memory,
                                        persistent_workers=persistent_workers_train,
                                        )
        else:
            gdro_weights = train_dataset.weights
            num_groups = train_dataset.num_groups

            train_dataset_group_proportions = train_dataset.group_proportions
            validation_dataset_group_proportions = validation_dataset.group_proportions
            
            train_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        #    collate_fn=numpy_collate,
                                        num_workers=num_workers_train,
                                        pin_memory=pin_memory,
                                        persistent_workers=persistent_workers_train,
                                        )
else:
    train_loader = data.DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True,
                                #    collate_fn=numpy_collate,
                                num_workers=num_workers_train,
                                pin_memory=pin_memory,
                                persistent_workers=persistent_workers_train,
                                )

if "context" in reg_type:
    # if data_balancing:
    #     init_group_weights = context_set.group_weights[np.array(train_dataset.group, dtype=int)] ** dataset_group_scale
    # else:
    #     init_group_weights = context_set.group_weights[train_dataset.sensitive]
    if dataset == "waterbirds":
        init_group_weights = context_set.group_weights[np.array(train_dataset.group, dtype=int)] ** dataset_group_scale
    elif dataset == "mnli":
        init_group_weights = context_set.group_weights[np.array(train_dataset.group, dtype=int)] ** dataset_group_scale
    elif dataset == "celeba":
        init_group_weights = context_set.dataset.group_weights[np.array(context_set.dataset.group[context_set.indices], dtype=int)] ** dataset_group_scale

    context_sampler = data.WeightedRandomSampler(init_group_weights, len(context_set), replacement=True, generator=torch.Generator().manual_seed(seed))
    shuffle = False
else:
    context_sampler = None
    shuffle = True

context_loader  = data.DataLoader(context_set,
                            batch_size=context_batch_size,
                            sampler=context_sampler,
                            shuffle=shuffle,
                            drop_last=True,
                            #    collate_fn=numpy_collate,
                            num_workers=num_workers_context,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers_context,
                            )
val_loader   = data.DataLoader(validation_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=False,
                            #    collate_fn=numpy_collate,
                               num_workers=num_workers_test,
                               pin_memory=pin_memory,
                               persistent_workers=persistent_workers_test
                               )
test_loader  = data.DataLoader(test_dataset,
                               batch_size=batch_size_test,
                               shuffle=False,
                               drop_last=False,
                            #    collate_fn=numpy_collate,
                               num_workers=num_workers_test,
                               pin_memory=pin_memory,
                               persistent_workers=persistent_workers_test
                               )

# for j in range(10):
#     # data_loader = tqdm(enumerate(zip(train_loader, context_loader)), leave=False)
#     data_loader = tqdm(enumerate(train_loader))
#     t0 = time.time()

#     for i, batch in data_loader:
#     # for i, (batch, batch_context) in data_loader:
#         x = np.array(batch[0], dtype=np.float32)
#         y = np.array(batch[1], dtype=np.float32)
#         z = np.array(batch[2], dtype=np.float32)
#         # pass

#     T = time.time()

#     print(f"{T-t0}s")
# print('done')

class TrainState(train_state.TrainState):
    batch_stats: Any
    # params_logvar: Any


class TrainerModule:
    def __init__(self,
                 model_name : str,
                 model_class : nn.Module,
                 optimizer_name : str,
                 model_hparams : dict,
                 optimizer_hparams : dict,
                 objective_hparams : dict,
                 ssm_hparams: dict,
                 other_hparams: dict,
                 exmp_inputs : Any,
                 ):
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.optimizer_name = optimizer_name

        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.objective_hparams = objective_hparams
        self.ssm_hparams = ssm_hparams
        self.other_hparams = other_hparams
        self.ssm_optimizer_hparams = ssm_hparams["ssm_optimizer_hparams"]
        
        self.seed = other_hparams["seed"]
        self.num_epochs = other_hparams["num_epochs"]
        self.evaluate = other_hparams["evaluate"]
        self.linearize = other_hparams['linearize']
        self.restore_checkpoint = other_hparams["restore_checkpoint"]
        self.checkpoint_dir = other_hparams["checkpoint_dir"]
        self.final_layer_random_init = other_hparams["final_layer_random_init"]
        self.batch_stats_init_epochs = other_hparams["batch_stats_init_epochs"]
        self.mc_samples_llk = objective_hparams["mc_samples_llk"]
        self.mc_samples_reg = objective_hparams["mc_samples_reg"]
        self.mc_samples_eval = other_hparams["mc_samples_eval"]
        self.dataset = other_hparams["dataset"]
        self.training_dataset_size = objective_hparams["training_dataset_size"]
        self.batch_size = objective_hparams["batch_size"]
        self.n_batches_train = self.training_dataset_size / self.batch_size
        self.num_classes = self.model_hparams["num_classes"]
        self.stochastic = other_hparams["stochastic"]
        self.learning_rate = self.optimizer_hparams["learning_rate"]
        self.learning_rate_scale_logvar = self.optimizer_hparams["learning_rate_scale_logvar"]
        self.alpha = self.optimizer_hparams["alpha"]
        self.weight_decay = self.optimizer_hparams["weight_decay"]
        self.momentum = self.optimizer_hparams['momentum']

        self.prior_mean = objective_hparams["prior_mean"]
        self.lr_schedule_name = objective_hparams["lr_schedule_name"]
        self.prior_var = objective_hparams["prior_var"]
        self.prior_likelihood_scale = objective_hparams["prior_likelihood_scale"]
        self.prior_likelihood_f_scale = objective_hparams["prior_likelihood_f_scale"]
        self.prior_likelihood_cov_scale = objective_hparams["prior_likelihood_cov_scale"]
        self.prior_likelihood_cov_diag = objective_hparams["prior_likelihood_cov_diag"]
        self.prior_likelihood_mean = objective_hparams["prior_likelihood_mean"]
        self.prior_likelihood_normalize_feature = objective_hparams["prior_likelihood_normalize_feature"]
        self.likelihood_scale = objective_hparams["likelihood_scale"]
        self.reg_scale = objective_hparams["reg_scale"]
        self.empirical_fairness_prior_scale = objective_hparams["empirical_fairness_prior_scale"]
        self.llm_dropout = objective_hparams["llm_dropout"]
        self.rho_sam = objective_hparams["rho_sam"]
        self.rho_adversarial = objective_hparams["rho_adversarial"]
        self.dropout_rate_sam = objective_hparams["dropout_rate_sam"]
        self.method = self.objective_hparams["method"]
        self.reg_type = self.objective_hparams["reg_type"]
        self.init_logvar = objective_hparams["init_logvar"]
        self.init_final_layer_weights_logvar = objective_hparams["init_final_layer_weights_logvar"]
        self.init_final_layer_bias_logvar = objective_hparams["init_final_layer_bias_logvar"]
        self.prior_feature_logvar = objective_hparams["prior_feature_logvar"]
        self.pretrained_prior = objective_hparams["pretrained_prior"]
        self.output_var = other_hparams["output_var"]
        
        self.ssm = ssm_hparams['ssm']
        self.primary_type = ssm_hparams['primary_type']
        self.secondary_type = ssm_hparams['secondary_type']
        self.tertiary_type = ssm_hparams['tertiary_type']

        self.optimizer_name_ssm = self.ssm_optimizer_hparams['optimizer_name']
        self.learning_rate_ssm = self.ssm_optimizer_hparams['learning_rate']
        self.alpha_ssm = self.ssm_optimizer_hparams['alpha']
        self.weight_decay_ssm = self.ssm_optimizer_hparams['weight_decay']
        self.momentum_ssm = self.ssm_optimizer_hparams['momentum']
        
        self.prediction_type = other_hparams["prediction_type"]
        self.debug_print = other_hparams["debug_print"]
        self.debug_print_updated = other_hparams["debug_print"]
        self.log_frequency = other_hparams["log_frequency"]
        self.log_frequency_steps = other_hparams["log_frequency_steps"]
        self.full_eval = other_hparams["full_eval"]
        self.sensitive_attribute = other_hparams["sensitive_attribute"]
        self.fairness_eval = other_hparams["fairness_eval"]
        self.fairness_train = other_hparams["fairness_train"]
        self.validation_training = other_hparams["validation_training"]
        self.final_layer_retraining = other_hparams["final_layer_retraining"]
        self.group_dro = other_hparams["group_dro"]
        self.gdro_step_size = other_hparams["gdro_step_size"]
        # self.gdro_weights = other_hparams["gdro_weights"]
        self.quick_eval = other_hparams["quick_eval"]
        self.val_train_frac = other_hparams["val_train_frac"]
        self.dataset_group_scale = other_hparams["dataset_group_scale"]
        self.data_balancing = other_hparams["data_balancing"]
        self.save_to_wandb = other_hparams["save_to_wandb"]
        self.wandb_project = other_hparams["wandb_project"]
        self.wandb_account = other_hparams["wandb_account"]

        self.n_batches_eval = 100
        self.n_batches_eval_context = 10
        self.n_batches_eval_final = 100

        self.params_prior_mean = None
        self.params_prior_logvar = None
        self.batch_stats_prior = None
        self.pred_fn = None

        self.gdro_weights = None

        if self.ssm:
            self.split_params = split_params_ssm
        else:
            self.split_params = split_params

        self.run_name = f"{method}_{reg_type}_{prior_var}_{prior_mean}_{reg_scale}_{learning_rate}_{alpha}_{num_epochs}_{context_points}_{model_name}_{dataset}_{seed}"
        if self.fairness_eval:
            self.run_name = f"{method}_{reg_type}_{prior_var}_{prior_mean}_{reg_scale}_{fairness_train}_{group_dro}_{gdro_step_size}_{empirical_fairness_prior_scale}_{learning_rate}_{alpha}_{num_epochs}_{context_points}_{model_name}_{dataset}_{seed}"

        self.log_dir = os.path.join(CHECKPOINT_PATH, self.run_name)

        self.logger = {
            "epoch": [],
            "loss_train": [],
            "acc_train": [],
            "acc_test": [],
            "acc_test_best": [],
            "acc_sel_test": [],
            "acc_sel_test_ood": [],
            "nll_test": [],
            "ece_test": [],
            "ood_auroc_entropy": [],
            "ood_auroc_aleatoric": [],
            "ood_auroc_epistemic": [],
            "predictive_entropy_test": [],
            "aleatoric_uncertainty_test": [],
            "epistemic_uncertainty_test": [],
            "predictive_entropy_context": [],
            "aleatoric_uncertainty_context": [],
            "epistemic_uncertainty_context": [],
            "predictive_entropy_ood": [],
            "aleatoric_uncertainty_ood": [],
            "epistemic_uncertainty_ood": [],
        }
        
        if "cifar10" in self.dataset and "cifar100" not in self.dataset:
            self.logger["acc_test_cifar101"] = []
            self.logger["acc_sel_test_cifar101"] = []
            self.logger["nll_test_cifar101"] = []
            self.logger["ece_test_cifar101"] = []
        if self.full_eval:
            if "cifar10" in self.dataset and "cifar100" not in self.dataset:
                for corr_config in corr_config_list:
                    self.logger[f"acc_test_ccifar10_{corr_config}"] = []
                    self.logger[f"acc_test_ccifar10_{corr_config}"] = []
                    self.logger[f"acc_sel_test_ccifar10_{corr_config}"] = []
        if self.fairness_eval:
            self.logger["diff_in_eop"] = []
            self.logger["diff_in_eoo"] = []
            self.logger["diff_in_prp"] = []
            self.logger["worst_group_acc_train"] = []
            self.logger["worst_group_acc_test"] = []
            self.logger["group_weighted_acc_test"] = []

            if validation_training:
                self.logger["acc_val"] = []
                self.logger["worst_group_acc_val"] = []
                self.logger["group_weighted_acc_val"] = []

        self.wandb_logger = []

        self.create_functions()
        
        if self.ssm:
            # TODO: changes
            self.lr = self.ssm_hparams['ssm_optimizer_hparams']["lr"]
            self.alpha = self.ssm_hparams['ssm_optimizer_hparams']["alpha"]
            self.weight_decay = self.ssm_hparams['ssm_optimizer_hparams']["weight_decay"]
            self.lr_schedule = True

            # layer_cls = S4DLayer
            # self.learning_rate_layer = getattr(layer_cls, "lr", None)
            # for key in self.learning_rate_layer.keys():
            #     self.learning_rate_layer[key] = self.learning_rate_ssm
            
            # self.model_init = partial(
            #     BatchStackedModel,
            #     layer_cls=layer_cls,
            #     d_output=self.num_classes,
            #     classification=True,
            #     **self.ssm_hparams['ssm_model_hparams'],
            # )
            self.init_model(exmp_inputs)
        else:
            if model_name == 'ResNet18-Pretrained':
                self.model = ResNet18(output='logits', pretrained='imagenet', \
                                      num_classes=self.num_classes, dtype='float32')
            elif model_name == 'ResNet50-Pretrained':
                self.model = ResNet50(output='logits', pretrained='imagenet', \
                                      num_classes=self.num_classes, dtype='float32')
            # elif model_name == 'ResNet18':
            #     self.model = ResNet18(output='logits', pretrained=None, normalize=False, \
            #                           num_classes=self.num_classes, dtype='float32')
            # elif model_name == 'ResNet50':
            #     self.model = ResNet50(output='logits', pretrained=None, normalize=False, \
            #                           num_classes=self.num_classes, dtype='float32')
            elif 'bert' in model_name:
                print("Use BERT with flax")
                bert_config = BertConfig.from_pretrained(
                    model_name,
                    num_labels=3,
                    finetuning_task='mnli'
                )
                bert_config.attention_probs_dropout_prob = self.llm_dropout
                bert_config.hidden_dropout_prob = self.llm_dropout

                self.model = FlaxBertForSequenceClassification.from_pretrained(
                    model_name,
                    config=bert_config,
                    seed=seed
                )
                self.model.apply = self.model.__call__
            else:
                self.model = self.model_class(**self.model_hparams)        
            self.init_model(exmp_inputs)
        # print(self.model.tabulate(random.PRNGKey(0), x=exmp_inputs[0]))

        assert self.mc_samples_llk == 1 if not self.stochastic else True
        assert self.mc_samples_eval == 1 if not self.stochastic else True
        assert self.mc_samples_reg == 1
        # assert self.objective_hparams["reg_points"] == "train" if self.objective_hparams["method"] == "psmap" else True

    def create_train_state(
        self,
        rng,
        model_cls,
        trainloader,
        learning_rate_layer=None,
        total_steps=-1,
    ):
        def map_nested_fn(fn):
            """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

            def map_fn(nested_dict):
                ret = {
                    k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
                    for k, v in nested_dict.items()
                }
                return ret

            return map_fn
        
        self.model = model_cls(training=True)
        init_rng, dropout_rng = jax.random.split(rng, num=2)

        params = self.model.init(
            {"params": init_rng, "dropout": dropout_rng},
            jnp.array(next(iter(trainloader))[0]),
        )
        # Note: Added immediate `unfreeze()` to play well w/ Optax. See below!
        params = params["params"].unfreeze()

        if learning_rate_layer is None:
            learning_rate_layer = {}

        num_steps_per_epoch = len(trainloader)

        if self.optimizer_name_ssm.lower() == 'adam':
            self.ssm_optimizer_hparams.pop('momentum')
            opt_class_ssm = optax.adamw
        elif self.optimizer_name_ssm.lower() == 'sgd':
            opt_class_ssm = optax.sgd
        else:
            assert False, f'Unknown optimizer "{self.optimizer_name_ssm}"'

        if self.alpha_ssm != 1:
            learning_rate_schedule = optax.cosine_decay_schedule(
                init_value=self.learning_rate_ssm,
                decay_steps=num_steps_per_epoch*num_epochs,
                alpha=self.alpha_ssm,
            )
        else:
            learning_rate_schedule = lambda lr: self.learning_rate_ssm
            # learning_rate_schedule = optax.piecewise_constant_schedule(
            #     init_value=self.learning_rate_ssm
            # )

        transf_ssm = []
        # transf_ssm = [optax.clip(1.0)]
        # transf_ssm = [optax.clip_by_global_norm(1.0)]

        if opt_class_ssm == optax.sgd and 'weight_decay' in self.ssm_optimizer_hparams:  # wd is integrated in adamw
            transf_ssm.append(optax.add_decayed_weights(self.weight_decay_ssm))
            self.ssm_optimizer_hparams.pop('weight_decay')

        optimizers = {
                k: optax.chain(
                *transf_ssm,
                opt_class_ssm(learning_rate_schedule, **self.ssm_optimizer_hparams)
            )
            for k, v in learning_rate_layer.items()
        }

        if self.optimizer_name.lower() == 'adam':
            self.optimizer_hparams.pop('momentum')
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{self.optimizer_name}"'
        
        if self.alpha != 1:
            if self.lr_schedule_name == 'linear':
                learning_rate_schedule = optax.linear_schedule(
                    init_value=self.learning_rate,
                    end_value=self.alpha,
                    transition_steps=num_steps_per_epoch*num_epochs,
                )
            else:
                learning_rate_schedule = optax.cosine_decay_schedule(
                    init_value=self.learning_rate,
                    decay_steps=num_steps_per_epoch*num_epochs,
                    alpha=self.alpha,
                )
        else:
            learning_rate_schedule = lambda lr: self.learning_rate
            # learning_rate_schedule = optax.piecewise_constant_schedule(
            #     init_value=self.learning_rate
            # )

        transf = []
        # transf = [optax.clip(1.0)]
        # transf = [optax.clip_by_global_norm(1.0)]

        if opt_class == optax.sgd and 'weight_decay':  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(self.weight_decay))
            self.optimizer_hparams.pop('weight_decay')

        optimizers["__default__"] = optax.chain(
            *transf,
            opt_class(learning_rate_schedule, **self.optimizer_hparams)
        )

        name_map = map_nested_fn(lambda k, _: k if k in learning_rate_layer else "__default__")
        tx = optax.multi_transform(optimizers, name_map)

        # Check that all special parameter names are actually parameters
        extra_keys = set(learning_rate_layer.keys()) - set(jax.tree_util.tree_leaves(name_map(params)))
        assert (len(extra_keys) == 0), f"Special params {extra_keys} do not correspond to actual params"

        _is_complex = lambda x: x.dtype in [jnp.complex64, jnp.complex128]
        param_sizes = map_nested_fn(
            lambda k, param: param.size * (2 if _is_complex(param) else 1)
            if learning_rate_layer.get(k, self.learning_rate) > 0.0
            else 0
        )(params)
        
        print(f"[*] Trainable Parameters: {sum(jax.tree_util.tree_leaves(param_sizes))}")
        print(f"[*] Total training steps: {total_steps}")

        _, params_subset = self.split_params(params, primary_type=self.primary_type, secondary_type=self.secondary_type, tertiary_type=self.tertiary_type)
        params_subset_logvar = jax.tree_map(lambda x: x * 0 + jnp.log(self.prior_var), params_subset)

        params_combined = {
            "params": params,
            "params_logvar": params_subset_logvar,
        }

        if 'bert' in self.model_name:
            return TrainState.create(apply_fn=self.model.__call__, params=params_combined, tx=tx, batch_stats=None)
        else:
            return TrainState.create(apply_fn=self.model.apply, params=params_combined, tx=tx, batch_stats=None) 
    
    def create_functions(self):
        def breakpoint_if_nonpositive(x, y, z):
            is_positive = jnp.greater(x, 0)
            def true_fn(x):
                pass
            def false_fn(x):
                jax.debug.print("{}", x[0])
                jax.debug.print("{}", x[2])
                jax.debug.print("{}", jnp.linalg.eigh(jax.lax.stop_gradient(x[1]))[0])
                jax.debug.breakpoint()

            jax.lax.cond(is_positive, true_fn, false_fn, [x,y, z])

        def calculate_cov(jac, logvar):
            var = jnp.exp(logvar)
            # jac has shape (batch_dim, output_dim, params_dims...)
            # jac_2D has shape (batch_dim * output_dim, nb_params)
            batch_dim, output_dim = jac.shape[:2]
            jac_2D = jnp.reshape(jac, (batch_dim * output_dim, -1))
            # sigma_flatten has shape (nb_params,) and will be broadcasted to the same shape as jac_2D
            sigma_flatten = jnp.reshape(var, (-1,))
            # jac_sigma_product has the same shape as jac_2D
            jac_sigma_product = jnp.multiply(jac_2D, sigma_flatten)
            cov = jnp.matmul(jac_sigma_product, jac_2D.T)
            cov = jnp.reshape(cov, (batch_dim, output_dim, batch_dim, output_dim))
            return cov

        def calculate_moments(params_mean, params_logvar, inputs, batch_stats, rng_key):
            ### Split both mean and logvar parameters
            params_feature_mean, params_final_layer_mean = self.split_params(params_mean, "dense")
            params_feature_logvar, params_final_layer_logvar = self.split_params(params_logvar, "dense")

            ### sample feature parameters and merge with final-layer mean parameters
            params_feature_sample = sample_parameters(params_feature_mean, params_feature_logvar, self.stochastic, rng_key)
            params_partial_sample = merge_params(params_feature_sample, params_final_layer_mean)

            ### feature covariance (mostly the same as Jacobian covariance (does not include bias term), up to numerical errors)
            if 'bert' in self.model_name:
                _out = self.model.__call__(
                    input_ids=inputs[:,0],
                    token_type_ids=inputs[:,1],
                    attention_mask=inputs[:,2],
                    params=params_partial_sample,
                    train=True,
                    dropout_rng=self.dropout_rng,
                )
            else:
                _out = self.model.apply({'params': params_partial_sample, 'batch_stats': batch_stats},
                                        inputs,
                                        train=True,
                                        feature=True,
                                        mutable=['batch_stats'])
            out, _ = _out
            preds_f_sample, feature_sample = out[0], out[1]
            
            n_samples = preds_f_sample.shape[0]
            feature_dim = params_final_layer_mean[self.final_layer_key]["kernel"].shape[0]
            final_layer_var_weights = jnp.exp(params_final_layer_logvar[self.final_layer_key]["kernel"])
            final_layer_var_bias = jnp.exp(params_final_layer_logvar[self.final_layer_key]["bias"])

            feature_times_var = (jnp.repeat(final_layer_var_weights, n_samples).reshape(n_samples, feature_dim, self.num_classes) * feature_sample[:, :, None]).transpose(2, 0, 1)
            preds_f_cov = jnp.matmul(feature_times_var, feature_sample.T).transpose(1, 2, 0)
            preds_f_cov += preds_f_cov + final_layer_var_bias[None, None, :]

            ### alternative, less memory efficient way to compute covariance
            # diag_mat_weights = jnp.diagflat(final_layer_var_weights).reshape(feature_dim, self.num_classes, feature_dim, self.num_classes).transpose(1, 3, 0, 2)
            # diag_mat_bias = jnp.tile(jnp.diagflat(final_layer_var_bias), (n_samples, n_samples)).reshape(n_samples, self.num_classes, n_samples, self.num_classes)
            # preds_f_cov = jnp.matmul(jnp.matmul(feature_sample, diag_mat_weights), feature_sample.T).transpose(2, 0, 3, 1)
            # preds_f_cov += diag_mat_bias

            # ### Compute single-sample MC estimate of mean of preds_f
            # _out = self.model.apply(
            #     {'params': params_partial_sample, 'batch_stats': batch_stats},
            #     inputs,
            #     train=True,
            #     mutable=['batch_stats']
            #     )
            # out, _ = _out
            # preds_f_sample = out
            # 
            # ### Compute single-sample MC estimate of covariance of preds_f
            # pred_fn = lambda final_layer_params: self.model.apply({'params': merge_params(params_feature_sample, final_layer_params), 'batch_stats': batch_stats}, inputs, train=True, mutable=['batch_stats'])
            # jacobian = jax.jacobian(pred_fn)(params_final_layer_mean)[0]
            # preds_f_cov = tree.map_structure(calculate_cov, jacobian, params_final_layer_logvar)
            # preds_f_cov = jnp.stack(tree.flatten(preds_f_cov), axis=0).sum(axis=0)[:, 0, :, :]

            return preds_f_sample, preds_f_cov

        def calculate_function_kl(params_variational_mean, params_variational_logvar, inputs, batch_stats, rng_key):
            ### set prior batch stats
            if self.batch_stats_init_epochs == 0:
                batch_stats_prior = jax.lax.stop_gradient(batch_stats)
            else:
                batch_stats_prior = self.batch_stats_prior
                
            ### set prior mean parameters
            if self.params_prior_mean is not None:
                params_prior_mean = jax.lax.stop_gradient(self.params_prior_mean)
            else:
                # params_prior_mean = jax.lax.stop_gradient(self.model.init(rng_key, inputs[0:1], train=True)["params"])
                params_prior_mean = jax.lax.stop_gradient(self.model.init(jax.random.PRNGKey(self.seed), inputs[0:1], train=True)["params"])
                # params_prior_mean = jax.tree_map(lambda x, y: x + y, jax.lax.stop_gradient(params), jax.lax.stop_gradient(self.model.init(rng_key, inputs[0:1], train=True)["params"]))

            ### set parameter prior variance
            feature_prior_logvar = self.prior_feature_logvar
            final_layer_prior_logvar = jnp.log(self.prior_var)

            ### initialize and split prior logvar parameters into feature and final-layer parameters
            params_prior_logvar_init = jax.tree_map(lambda x: x * 0, params_prior_mean)  # initialize logvar parameters with zeros
            params_feature_prior_logvar_init, params_final_layer_prior_logvar_init = self.split_params(params_prior_logvar_init, "dense")

            ### set feature and final-layer logvar parameters separately
            params_feature_prior_logvar = jax.tree_map(lambda x: x * 0 + feature_prior_logvar, params_feature_prior_logvar_init)
            params_final_layer_prior_logvar = jax.tree_map(lambda x: x * 0 + final_layer_prior_logvar, params_final_layer_prior_logvar_init)

            ### merge logvar parameters
            params_prior_logvar = merge_params(params_feature_prior_logvar, params_final_layer_prior_logvar)

            preds_f_prior_mean, preds_f_prior_cov = calculate_moments(params_prior_mean, params_prior_logvar, inputs, batch_stats_prior, rng_key)
            preds_f_variational_mean, preds_f_variational_cov = calculate_moments(params_variational_mean, params_variational_logvar, inputs, batch_stats, rng_key)

            if self.debug_print_updated:
                jax.debug.print("\ncov prior:\n{}", preds_f_prior_cov[0:2, 0, 0:2, 0])
                jax.debug.print("cov variational:\n{}\n", preds_f_variational_cov[0:2, 0, 0:2, 0])
                jax.debug.print("cov prior inv:\n{}", jnp.linalg.inv(preds_f_prior_cov)[0:2, 0, 0:2, 0])
                jax.debug.print("cov variational inv:\n{}\n", jnp.linalg.inv(preds_f_variational_cov)[0:2, 0, 0:2, 0])

            kl = 0
            n_samples = preds_f_variational_mean.shape[0]
            cov_jitter = 1e-6
            for j in range(self.num_classes):
                _preds_f_prior_mean = preds_f_prior_mean[:, j].transpose()
                _preds_f_prior_cov = preds_f_prior_cov[:, :, j] + jnp.eye(n_samples) * cov_jitter

                _preds_f_variational_mean = preds_f_variational_mean[:, j].transpose()
                _preds_f_variational_cov = preds_f_variational_cov[:, :, j] + jnp.eye(n_samples) * cov_jitter

                # _preds_f_prior_mean = preds_f_prior_mean[:, j].transpose()
                # _preds_f_prior_cov = preds_f_prior_cov[:, j, :, j] + jnp.eye(n_samples) * cov_jitter
                # _preds_f_prior_mean = jnp.ones(n_samples) * 0
                # _preds_f_prior_cov = jnp.eye(n_samples) * final_layer_prior_var

                q = tfd.MultivariateNormalFullCovariance(
                    loc=_preds_f_variational_mean,
                    covariance_matrix=_preds_f_variational_cov,
                    validate_args=False,
                    allow_nan_stats=True,
                )
                p = tfd.MultivariateNormalFullCovariance(
                    loc=_preds_f_prior_mean,
                    covariance_matrix=_preds_f_prior_cov,
                    validate_args=False,
                    allow_nan_stats=True,
                )
                kl += tfd.kl_divergence(q, p, allow_nan_stats=False)

            return kl

        def calculate_function_prior_density(preds_f, params, inputs, batch_stats, rng_key, prior_var):
            ### set prior batch stats
            if self.batch_stats_init_epochs == 0:
                batch_stats_prior = jax.lax.stop_gradient(batch_stats)
            else:
                batch_stats_prior = self.batch_stats_prior

            ### set parameter prior mean
            if self.params_prior_mean is not None:
                params_prior_mean = jax.lax.stop_gradient(self.params_prior_mean)
            else:
                # params_prior_mean = jax.lax.stop_gradient(self.model.init(rng_key, inputs[0:1], train=True)["params"])
                params_prior_mean = jax.lax.stop_gradient(self.model.init(jax.random.PRNGKey(self.seed), inputs[0:1], train=True)["params"])
                # params_prior_mean = jax.lax.stop_gradient(params)
            params_feature_prior_mean, params_final_layer_prior_mean = self.split_params(params_prior_mean, "dense")

            ### initialize and split prior logvar parameters into feature and final-layer parameters
            params_prior_logvar_init = jax.tree_map(lambda x: x * 0, params_prior_mean)  # initialize logvar parameters with zeros
            params_feature_prior_logvar_init, params_final_layer_prior_logvar_init = self.split_params(params_prior_logvar_init, "dense")

            ### set feature parameter logvar and final-layer parameter variance
            feature_prior_logvar = self.prior_feature_logvar
            final_layer_prior_logvar = jnp.log(prior_var)
            params_feature_prior_logvar = jax.tree_map(lambda x: x * 0 + feature_prior_logvar, params_feature_prior_logvar_init)
            # params_final_layer_prior_logvar = jax.tree_map(lambda x: x * 0 + final_layer_prior_logvar, params_final_layer_prior_logvar_init)

            params_feature_prior_sample = sample_parameters(params_feature_prior_mean, params_feature_prior_logvar, self.stochastic, rng_key)
            # params_feature_prior_sample = params_feature_prior_mean
            # params_feature_prior_sample = sample_parameters(jax.tree_map(lambda x: x * 0, params_feature_prior_mean), params_feature_prior_logvar, self.stochastic, rng_key)  # use for non-init distribution feature variance

            params_prior_sample = merge_params(params_feature_prior_sample, params_final_layer_prior_mean)

            ### feature covariance (mostly the same as Jacobian covariance, up to numerical errors)
            if 'bert' in self.model_name:
                _out = self.model.__call__(
                    input_ids=inputs[:,0],
                    token_type_ids=inputs[:,1],
                    attention_mask=inputs[:,2],
                    params=params_prior_sample,
                    train=True,
                    dropout_rng=self.dropout_rng,
                )
            else:
                _out = self.model.apply({'params': params_prior_sample, 'batch_stats': batch_stats_prior},
                                        inputs,
                                        train=True,
                                        feature=True,
                                        mutable=['batch_stats'])
            out, _ = _out
            # out = self.model.apply({'params': params_prior_sample, 'batch_stats': batch_stats_prior},
            #                         inputs,
            #                         train=False,
            #                         feature=True,
            #                         mutable=False)
            # preds_f_prior_mean, feature_prior = jax.lax.stop_gradient(out[0]), jax.lax.stop_gradient(out[1])
            preds_f_prior_mean = jnp.zeros_like(jax.lax.stop_gradient(out[0]))
            feature_prior = jax.lax.stop_gradient(out[1])

            preds_f_prior_cov = prior_var * jnp.matmul(feature_prior, feature_prior.T)  # assumes the prior is identical across output dimensions
            preds_f_prior_cov += jnp.ones_like(preds_f_prior_cov) * prior_var  # add bias variance
            preds_f_prior_cov += jnp.eye(preds_f_prior_cov.shape[0]) * prior_var  # jnp.max(jnp.array([eps_cov * prior_var, eps_cov]))  # add small constant to the diagonal to ensure positive definiteness

            #### alternative feature covariance implementation
            # feature_dim = params_prior_sample[self.final_layer_key]["kernel"].shape[0]
            # final_layer_prior_var = jax.tree_map(lambda x: x * 0 + prior_var, jax.lax.stop_gradient(params_prior_sample[self.final_layer_key]["kernel"]))
            # diag_mat = jnp.diagflat(final_layer_prior_var).reshape(feature_dim,self.num_classes,feature_dim,self.num_classes).transpose(1, 3, 0, 2)
            # preds_f_prior_cov = jnp.matmul(jnp.matmul(feature_prior, diag_mat), feature_prior.T).transpose(2, 0, 3, 1)
            # preds_f_prior_cov += jnp.ones_like(preds_f_prior_cov) * prior_var  # add bias variance

            ### Jacobian covariance (mostly the same as feature covariance (does include bias term), up to numerical errors)
            # _out = self.model.apply(
            #     {'params': params_prior_sample, 'batch_stats': batch_stats_prior},
            #     inputs,
            #     train=True,
            #     mutable=['batch_stats']
            #     )
            # out, _ = _out
            # preds_f_prior_mean = jax.lax.stop_gradient(out)

            # pred_fn = lambda final_layer_params: self.model.apply(
            #     {'params': merge_params(params_feature_prior_sample, final_layer_params), 'batch_stats': batch_stats_prior},
            #     inputs,
            #     train=True,
            #     mutable=['batch_stats']
            #     )
            # jacobian = jax.jacobian(pred_fn)(params_final_layer_prior_mean)[0]
            # preds_f_prior_cov = tree.map_structure(calculate_cov, jacobian, params_final_layer_prior_logvar)
            # preds_f_prior_cov = jnp.stack(tree.flatten(preds_f_prior_cov), axis=0).sum(axis=0)

            p = tfd.MultivariateNormalFullCovariance(
                loc=preds_f_prior_mean[:, 0],  # assumes the prior is identical across output dimensions
                covariance_matrix=preds_f_prior_cov,
                validate_args=False,
                allow_nan_stats=True,
            )
            log_density = jnp.sum(p.log_prob(preds_f[0].T))

            reg = -log_density

            # log_density = 0
            # for j in range(self.num_classes):
            #     p = tfd.MultivariateNormalFullCovariance(
            #         loc=preds_f_prior_mean[:, j],
            #         covariance_matrix=preds_f_prior_cov[:, j, :, j],
            #         validate_args=False,
            #         allow_nan_stats=True,
            #     )
            #     log_density += p.log_prob(preds_f[:, j])
            #     reg = -log_density

            # ### L2 regularization on non-final-layer parameters
            # params_model, _ = self.split_params(params, "dense")
            # reg += 1 / (2 * (1/0.025)) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.square(x), params_model))[0])  # 1/2 * ||params - params_prior_mean||^2

            # ### L2 regularization on model parameters
            # params_model, _ = self.split_params(params, "batch_norm")
            # reg += 1 / (2 * (1/0.025)) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.square(x), params_model))[0])  # 1/2 * ||params - params_prior_mean||^2

            # ### L2 regularization on BN parameters
            # _, params_batchnorm = self.split_params(params, "batch_norm")
            # reg += 1 / (2 * (1/0.007)) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.square(x), params_batchnorm))[0])  # 1/2 * ||params - params_prior_mean||^2

            if self.debug_print_updated:
                jax.debug.print("\nf - mean: {}", jnp.mean(preds_f[:, 0] - preds_f_prior_mean[:, 0]))
                jax.debug.print("log_density: {}\n", p.log_prob(preds_f.T))
                jax.debug.print("cholesky: {}\n", jnp.linalg.cholesky(preds_f_prior_cov[:, 0, :, 0]))

            return reg

        def calculate_empirical_gaussian_prior_density(preds_f, params, inputs, batch_stats, prior_likelihood_cov_scale, prior_likelihood_cov_diag, rng_key):
            ### set prior batch stats
            if self.batch_stats_init_epochs == 0:
                batch_stats_prior = jax.lax.stop_gradient(batch_stats)
            else:
                batch_stats_prior = self.batch_stats_prior

            ### set parameter prior mean
            if self.params_prior_mean is not None:
                params_prior_mean = jax.lax.stop_gradient(self.params_prior_mean)
            else:
                # params_prior_mean = jax.lax.stop_gradient(self.model.init(rng_key, inputs[0:1], train=True)["params"])
                params_prior_mean = jax.lax.stop_gradient(self.model.init(jax.random.PRNGKey(self.seed), inputs[0:1], train=True)["params"])
                # params_prior_mean = jax.lax.stop_gradient(params)
            params_feature_prior_mean, params_final_layer_prior_mean = self.split_params(params_prior_mean, "dense")

            ### initialize and split prior logvar parameters into feature and final-layer parameters
            params_prior_logvar_init = jax.tree_map(lambda x: x * 0, params_prior_mean)  # initialize logvar parameters with zeros
            params_feature_prior_logvar_init, params_final_layer_prior_logvar_init = self.split_params(params_prior_logvar_init, "dense")

            ### set feature parameter logvar and final-layer parameter variance
            feature_prior_logvar = self.prior_feature_logvar
            final_layer_prior_logvar = jnp.log(self.prior_var)
            params_feature_prior_logvar = jax.tree_map(lambda x: x * 0 + feature_prior_logvar, params_feature_prior_logvar_init)
            params_final_layer_prior_logvar = jax.tree_map(lambda x: x * 0 + final_layer_prior_logvar, params_final_layer_prior_logvar_init)

            params_feature_prior_sample = sample_parameters(params_feature_prior_mean, params_feature_prior_logvar, self.stochastic, rng_key)
            # params_feature_prior_sample = params_feature_prior_mean
            # params_feature_prior_sample = sample_parameters(jax.tree_map(lambda x: x * 0, params_feature_prior_mean), params_feature_prior_logvar, self.stochastic, rng_key)  # use for non-init distribution feature variance

            params_prior_sample = merge_params(params_feature_prior_sample, params_final_layer_prior_mean)

            ### feature covariance (mostly the same as Jacobian covariance, up to numerical errors)
            if 'bert' in self.model_name:
                _out = self.model.__call__(
                    input_ids=inputs[:,0],
                    token_type_ids=inputs[:,1],
                    attention_mask=inputs[:,2],
                    params=params_prior_sample,
                    train=True,
                    dropout_rng=self.dropout_rng,
                )
            else:
                _out = self.model.apply({'params': params_prior_sample, 'batch_stats': batch_stats_prior},
                                        inputs,
                                        train=True,
                                        feature=True,
                                        mutable=['batch_stats'])
            out, _ = _out
            preds_f_prior_mean = jnp.zeros_like(jax.lax.stop_gradient(out[0]))
            feature_prior = jax.lax.stop_gradient(out[1])
            # preds_f_prior_mean, feature_prior = jax.lax.stop_gradient(out[0]), jax.lax.stop_gradient(out[1])

            preds_f_prior_cov = prior_likelihood_cov_scale * jnp.matmul(feature_prior, feature_prior.T)  # assumes the prior is identical across output dimensions
            preds_f_prior_cov += jnp.ones_like(preds_f_prior_cov) * prior_likelihood_cov_scale  # add bias variance
            preds_f_prior_cov += jnp.eye(preds_f_prior_cov.shape[0]) * prior_likelihood_cov_diag  # jnp.max(jnp.array([eps_cov * prior_var, eps_cov]))  # add small constant to the diagonal to ensure positive definiteness

            p = tfd.MultivariateNormalFullCovariance(
                loc=preds_f_prior_mean[:, 0],  # assumes the prior is identical across output dimensions
                covariance_matrix=preds_f_prior_cov,
                validate_args=False,
                allow_nan_stats=True,
            )
            log_density = jnp.sum(p.log_prob(preds_f[0].T))

            reg = -log_density
            
            ### L2 regularization on non-final-layer parameters
            reg += 1 / (2 * (self.prior_var)) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.square(x), params))[0])  # 1/2 * ||params - params_prior_mean||^2

            if self.debug_print_updated:
                jax.debug.print("\nf - mean: {}", jnp.mean(preds_f[:, 0] - preds_f_prior_mean[:, 0]))
                jax.debug.print("log_density: {}\n", p.log_prob(preds_f.T))
                jax.debug.print("cholesky: {}\n", jnp.linalg.cholesky(preds_f_prior_cov[:, 0, :, 0]))

            return reg

        def calculate_empirical_gaussian_prior_kl(params_samples, params_variational_mean, params_variational_logvar, preds_f, inputs, batch_stats, prior_likelihood_scale, prior_likelihood_f_scale, prior_likelihood_cov_scale, prior_likelihood_cov_diag, rng_key):
            ### set prior batch stats
            if self.batch_stats_init_epochs == 0:
                batch_stats_prior = jax.lax.stop_gradient(batch_stats)
            else:
                batch_stats_prior = self.batch_stats_prior

            ### set parameter prior mean
            if self.params_prior_mean is not None:
                params_prior_mean = jax.lax.stop_gradient(self.params_prior_mean)
            else:
                params_prior_mean = jax.tree_map(lambda x: x * prior_likelihood_scale, jax.lax.stop_gradient(self.model.init(rng_key, inputs[0:1], train=True)["params"]))
                # params_prior_mean = jax.lax.stop_gradient(self.model.init(jax.random.PRNGKey(self.seed), inputs[0:1], train=True)["params"])
                # params_prior_mean = jax.lax.stop_gradient(params)

            if not self.ssm:
                if 'bert' in self.model_name:
                    _out = self.model.__call__(
                        input_ids=inputs[:,0],
                        token_type_ids=inputs[:,1],
                        attention_mask=inputs[:,2],
                        params=params_prior_mean,
                        train=True,
                        dropout_rng=self.dropout_rng,
                    )
                else:
                    _out = self.model.apply({'params': params_prior_mean, 'batch_stats': batch_stats_prior},
                                            inputs,
                                            train=True,
                                            feature=True,
                                            mutable=['batch_stats'])
                out, _ = _out
            else:
                out, mod_vars = self.pred_fn(
                    {"params": params_prior_mean},
                    inputs,
                    rngs={"dropout": rng_key},
                    mutable=["intermediates"],
                )
            preds_f_prior_mean, feature_prior = jax.lax.stop_gradient(out[0]), jax.lax.stop_gradient(out[1])

            cross_entropy = 0
            for i, params_sample in enumerate(params_samples):
                if not self.ssm:
                    if 'bert' in self.model_name:
                        _out = self.model.__call__(
                            input_ids=inputs[:,0],
                            token_type_ids=inputs[:,1],
                            attention_mask=inputs[:,2],
                            params=params_sample,
                            train=True,
                            dropout_rng=self.dropout_rng,
                        )
                    else:
                        _out = self.model.apply({'params': params_sample, 'batch_stats': batch_stats},
                                                inputs,
                                                train=True,
                                                feature=True,
                                                mutable=['batch_stats'])
                    out, _ = _out
                else:
                    out, mod_vars = self.pred_fn(
                        {"params": params_sample},
                        inputs,
                        rngs={"dropout": rng_key},
                        mutable=["intermediates"],
                    )
                feature_variational_sample = out[1]

                if self.prior_likelihood_normalize_feature:
                    feature_cov = (feature_variational_sample - feature_variational_sample.mean(0)) / feature_variational_sample.std(0)
                else:
                    feature_cov = feature_variational_sample
                    # feature_cov = feature_prior

                n_inputs = feature_cov.shape[0]
                feature_dim = params_variational_logvar[self.final_layer_key]["kernel"].shape[0]
                final_layer_var_weights = jnp.exp(params_variational_logvar[self.final_layer_key]["kernel"])
                final_layer_var_bias = jnp.exp(params_variational_logvar[self.final_layer_key]["bias"])

                feature_times_var = (jnp.repeat(final_layer_var_weights, n_inputs).reshape(n_inputs, feature_dim, self.num_classes) * feature_cov[:, :, None]).transpose(2, 0, 1)
                preds_f_variational_cov = jnp.matmul(feature_times_var, feature_cov.T).transpose(1, 2, 0)
                preds_f_variational_cov += final_layer_var_bias[None, None, :]
                preds_f_variational_cov += prior_likelihood_cov_diag * jnp.eye(preds_f_variational_cov.shape[0])[:, :, None]  # add jitter for numerical stability
                preds_f_variational_cov *= prior_likelihood_cov_scale

                _cross_entropy = 0
                for j in range(self.num_classes):
                    # breakpoint_if_nonpositive(jnp.linalg.eigh(jax.lax.stop_gradient(preds_f_variational_cov[:, :, j]))[0].min(), preds_f_variational_cov[:, :, j], feature_cov)
                    q = tfd.MultivariateNormalFullCovariance(
                        loc=preds_f[i, :, j],
                        covariance_matrix=preds_f_variational_cov[:, :, j],
                        validate_args=False,
                        allow_nan_stats=True,
                    )

                    _cross_entropy += -(1 / self.num_classes) * jnp.sum(q.log_prob(prior_likelihood_f_scale * preds_f_prior_mean[:, j].T))

                cross_entropy += (1 / len(params_samples)) * _cross_entropy
                # cross_entropy += 1000 * jnp.sum(jnp.square(final_layer_var_weights.flatten() - prior_likelihood_cov_scale)) + jnp.sum(jnp.square(final_layer_var_bias.flatten() - prior_likelihood_cov_scale))
                
            kl_params = calculate_parameter_kl(params_variational_mean, params_variational_logvar)

            kl = kl_params + cross_entropy

            if self.debug_print_updated:
                jax.debug.print("kl_params: {}", kl_params)
                jax.debug.print("kl_prior: {}", cross_entropy)

            return kl

        def calculate_empirical_categorical_prior(preds_f, prediction_type):
            if prediction_type == "classification":
                log_preds = jax.nn.log_softmax((1 / self.prior_likelihood_scale) * preds_f, -1)
                cross_entropy = jnp.sum(-(1 / self.num_classes) * log_preds, axis=-1)  # sum over output dimensions
            elif prediction_type == "regression":
                preds_f_std = jnp.std(preds_f, axis=0)
                cross_entropy = jnp.sum(jnp.log(preds_f_std * jnp.sqrt(2 * jnp.pi * jnp.e)), axis=-1)  # sum over output dimensions

            reg = jnp.mean(jnp.sum(cross_entropy, axis=-1), axis=0)  # sum over input points and contract samples dimension

            return reg
        
        def calculate_empirical_categorical_prior_kl(params_samples, params_variational_logvar, preds_f, prior_var, prediction_type):
            if prediction_type == "classification":
                log_preds = jax.nn.log_softmax((1 / self.prior_likelihood_scale) * preds_f, -1)
                cross_entropy = jnp.sum(-(1 / self.num_classes) * log_preds, axis=-1)  # sum over output dimensions
                cross_entropy = jnp.mean(jnp.sum(cross_entropy, axis=-1), axis=0)  # sum over input points and take mean over MC samples
            elif prediction_type == "regression":
                # NotImplementedError
                preds_f_std = jnp.std(preds_f, axis=0)  # computed over MC samples
                cross_entropy = jnp.sum(jnp.log((preds_f_std + self.prior_likelihood_scale) * jnp.sqrt(2 * jnp.pi * jnp.e)), axis=-1)  # sum over output dimensions
                cross_entropy = jnp.sum(cross_entropy, axis=-1)  # sum over input points

            for params_sample in params_samples:
                cross_entropy += (1 / self.mc_samples_llk) * calculate_parameter_norm(params_sample, prior_var)
            
            ### remove batchnorm parameters from the KL calculation
            # params_variational_logvar, _ = self.split_params(params_variational_logvar, "batch_norm")

            params_variational_var = jax.tree_map(lambda x: jnp.exp(x), params_variational_logvar)

            neg_entropy = jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(
                lambda x: -jnp.log((x ** 0.5) * jnp.sqrt(2 * jnp.pi * jnp.e)), params_variational_var))[0]
                )

            kl = neg_entropy + cross_entropy

            return kl

        def calculate_entropic_prior(preds_f):
            preds = jax.nn.softmax(preds_f, -1)
            entropy = jnp.sum(preds * jnp.log(preds + 1e-10), axis=-1)
            reg = jnp.mean(self.prior_likelihood_scale * entropy)

            return reg

        def calculate_function_norm(preds_f_reg, inputs, batch_stats, prior_var):
            if self.params_prior_mean is None:
                preds_f_prior_mean = jnp.zeros_like(preds_f_reg)
            else:
                if 'bert' in self.model_name:
                    _out = self.model.__call__(
                        input_ids=inputs[:,0],
                        token_type_ids=inputs[:,1],
                        attention_mask=inputs[:,2],
                        params=self.params_prior_mean,
                        train=True,
                        dropout_rng=self.dropout_rng,
                    )
                else:
                    _out = self.model.apply({'params': self.params_prior_mean, 'batch_stats': batch_stats},
                                            inputs,
                                            train=True,
                                            mutable=['batch_stats'])
                out, _ = _out
                preds_f_prior_mean = jax.lax.stop_gradient(out) if 'bert' not in self.model_name else jax.lax.stop_gradient(out[0])
            reg = 1 / (2 * prior_var) * jnp.sum(jnp.square(preds_f_reg - preds_f_prior_mean))  # 1/(2 * function_var) * ||f(inputs, params) - f(inputs, params_prior_mean)||^2

            return reg
        
        def calculate_empirical_fairness(params, preds_f_llk, preds_f_reg, targets, prior_var):
            # reg = self.empirical_fairness_prior_scale * jnp.sum(jnp.square(preds_f_llk - preds_f_reg))  # 1/(2 * function_var) * ||f(context_x, params) - f(context_x', params)||^2

            nll_llk = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f_llk, -1), targets).mean(0)  # likelihood_scale = temperature
            nll_reg = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f_reg, -1), targets).mean(0)  # likelihood_scale = temperature

            reg = self.empirical_fairness_prior_scale * jnp.sum(jnp.abs((nll_reg - nll_llk) - prior_likelihood_mean))
            reg += calculate_parameter_norm(params, prior_var)

            return reg

        def calculate_group_dro_prior(params, preds_f_llk, preds_f_reg, batch_stats, inputs, targets, group_attr, num_groups, gdro_weights, likelihood_scale, train=True):
            if "exponential_sharpness_penalty":
                if 'bert' in self.model_name:
                    def loss_fn(_params):
                        out = self.pred_fn(
                            input_ids=inputs[:,0],
                            token_type_ids=inputs[:,1],
                            attention_mask=inputs[:,2],
                            params=_params,
                            train=train,
                            dropout_rng=self.dropout_rng,
                        )
                        preds_f, new_model_state = out[0], None

                        per_sample_nll = categorical_nll_with_softmax(jax.nn.softmax((1 / likelihood_scale) * preds_f, -1), targets)
                        # loss, _ = calculate_group_loss(per_sample_nll, group_attr, num_groups, gdro_weights)
                        loss = per_sample_nll.mean(0)

                        return loss, (preds_f, new_model_state)
                else:
                    def loss_fn(_params):
                        out = self.pred_fn(
                            {'params': _params, 'batch_stats': batch_stats},
                            inputs,
                            train=train,
                            mutable=['batch_stats'] if train else False
                            )
                        preds_f, new_model_state = out if train else (out, None)

                        per_sample_nll = categorical_nll_with_softmax(jax.nn.softmax((1 / likelihood_scale) * preds_f, -1), targets)
                        # loss, _ = calculate_group_loss(per_sample_nll, group_attr, num_groups, gdro_weights)
                        loss = per_sample_nll.mean(0)

                        return loss, (preds_f, new_model_state)

                _, grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(params)

                nll = categorical_nll_with_softmax(jax.nn.softmax((1 / likelihood_scale) * preds_f_llk, -1), targets).mean()

                _epsilon = grads
                _epsilon = jax.lax.stop_gradient(_epsilon)
                epsilon = dual_vector(_epsilon)

                params_perturbed = jax.tree_map(lambda a, b: a + b * self.rho_sam, params, epsilon)

                if 'bert' in self.model_name:
                    out = self.pred_fn(
                        input_ids=inputs[:,0],
                        token_type_ids=inputs[:,1],
                        attention_mask=inputs[:,2],
                        params=params_perturbed,
                        train=train,
                        dropout_rng=self.dropout_rng,
                    )
                    preds_f_reg, new_model_state_perturbed = out[0], None
                else:
                    out = self.pred_fn(
                        {'params': params_perturbed, 'batch_stats': batch_stats},
                        inputs,
                        train=train,
                        mutable=['batch_stats'] if train else False
                        )
                    preds_f_reg, new_model_state_perturbed = out if train else (out, None)
                    
                per_sample_nll_perturbed = categorical_nll_with_softmax(jax.nn.softmax((1 / likelihood_scale) * preds_f_reg, -1), targets)
                # nll_group, _ = calculate_group_loss(per_sample_nll_perturbed, group_attr, num_groups, gdro_weights)
                nll_group = per_sample_nll_perturbed.mean(0)

                unfairness_penalty = nll_group
            else:
                per_sample_nll = categorical_nll_with_softmax(jax.nn.softmax((1 / likelihood_scale) * preds_f_llk, -1), targets).mean(0)
                # nll_group, gdro_weights = calculate_group_loss(per_sample_nll, group_attr, num_groups, gdro_weights)
                nll_group = per_sample_nll.mean(0)

                if "exponential_laplace" in self.reg_type:
                    unfairness_penalty = nll_group + jnp.mean(jnp.abs(nll_group - per_sample_nll)) / self.n_batches_train
                elif "exponential_fraction_squared" in self.reg_type:
                    nll_average = per_sample_nll.mean(0)
                    unfairness_penalty = nll_group / jax.lax.stop_gradient(nll_average) ** 2
                elif "exponential_fraction" in self.reg_type:
                    nll_average = per_sample_nll.mean(0)
                    unfairness_penalty = nll_group / jax.lax.stop_gradient(nll_average)
                elif "exponential" in self.reg_type:
                    unfairness_penalty = nll_group

            reg = self.empirical_fairness_prior_scale * self.n_batches_train * unfairness_penalty

            return reg, gdro_weights, new_model_state_perturbed

        def calculate_group_loss(per_sample_loss, group_idx, num_groups, prev_weights):
            group_map = (group_idx == jnp.arange(num_groups).reshape(-1, 1)).astype(jnp.float32)
            group_count = group_map.sum(1)
            group_denom = group_count + (group_count==0).astype(jnp.float32)  # avoid nans

            if "group_dro_prior" in self.reg_type and not self.data_balancing:
                group_loss = (group_map @ per_sample_loss) / (group_denom ** 2)
            else:
                group_loss = (group_map @ per_sample_loss) / group_denom

            if "fixed_weights" in self.reg_type:
                weights = jnp.exp(self.gdro_step_size * group_loss)  # step size in GDRO repo is 0.01
            else:
                weights = prev_weights * jnp.exp(self.gdro_step_size * group_loss)  # step size in GDRO repo is 0.01
            weights = weights / weights.sum()

            if "argmax_weight" in self.reg_type:
                argmax_weight_idx = jnp.argmax(weights)
                # argmax_weight = weights[argmax_weight_idx]
                weights = jnp.zeros_like(weights)
                weights = weights.at[argmax_weight_idx].set(1)

            robust_loss = group_loss @ weights
            return robust_loss, weights

        def calculate_parameter_norm(params, prior_var):
            params_model = params
            # params_model, params_batchnorm = self.split_params(params, "batch_norm")

            if "feature_parameter_norm" not in self.objective_hparams["reg_type"]:
                params_reg = params_model
                if self.params_prior_mean is None:
                    params_reg_prior_mean = jax.tree_map(lambda x: x * 0, jax.lax.stop_gradient(params_model))
                else:
                    params_reg_prior_mean = self.params_prior_mean
                reg = 1 / (2 * prior_var) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x, y: jnp.square(x - y), params_reg, params_reg_prior_mean))[0])  # 1/2 * ||params - params_prior_mean||^2
            else:
                params_reg, _ = self.split_params(params_model, "dense")
                if self.pretrained_prior:
                    params_reg_prior_mean, _ = self.split_params(self.params_prior_mean, "dense")
                else:
                    params_reg_prior_mean = jax.tree_map(lambda x: x * self.prior_mean, jax.lax.stop_gradient(params_reg))
                reg = 1 / (2 * prior_var) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x, y: jnp.square(x - y), params_reg, params_reg_prior_mean))[0])  # 1/2 * ||params_feature - params_feature_prior_mean||^2

            return reg  # this scaling makes prior_precision consistent with the MAP objective scaling but inconsistent with the weight decay coefficient

        def calculate_parameter_l1_norm(params, prior_var):
            params_model = params
            # params_model, params_batchnorm = self.split_params(params, "batch_norm")

            if self.objective_hparams["reg_type"] != "feature_parameter_norm":
                params_reg = params_model
                if self.params_prior_mean is None:
                    params_reg_prior_mean = jax.tree_map(lambda x: x * 0, jax.lax.stop_gradient(params_model))
                else:
                    params_reg_prior_mean = self.params_prior_mean
                reg = 1 / (2 * prior_var) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x, y: jnp.square(x - y), params_reg, params_reg_prior_mean))[0])  # 1/2 * ||params - params_prior_mean||^2
            else:
                params_reg, _ = self.split_params(params_model, "dense")
                params_reg_prior_mean, _ = self.split_params(self.params_prior_mean, "dense")
                reg = 1 / (2 * prior_var) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x, y: jnp.abs(x - y), params_reg, params_reg_prior_mean))[0])  # 1/2 * ||params_feature - params_feature_prior_mean||^2

            return reg  # this scaling makes prior_precision consistent with the MAP objective scaling but inconsistent with the weight decay coefficient


        def kl_univariate_gaussians(mean_q, var_q, mean_p, var_p):
            logstd_jitter = 0
            kl_1 = jnp.log((var_p + logstd_jitter) ** 0.5) - jnp.log((var_q + logstd_jitter) ** 0.5)
            kl_2 = ((var_q + logstd_jitter) + (mean_q - mean_p) ** 2) / (var_p + logstd_jitter)
            kl_3 = -1
            kl = 0.5 * (kl_1 + kl_2 + kl_3)

            return kl

        def calculate_parameter_kl(params_variational_mean, params_variational_logvar):
            if self.ssm:
                _, params_variational_mean = self.split_params(params_variational_mean, primary_type=self.primary_type, secondary_type=self.secondary_type, tertiary_type=self.tertiary_type)
                _, params_variational_logvar = self.split_params(params_variational_logvar, primary_type=self.primary_type, secondary_type=self.secondary_type, tertiary_type=self.tertiary_type)

            if self.params_prior_mean is not None:
                params_prior_mean = self.params_prior_mean
                params_prior_logvar = self.params_prior_logvar
            else:
                params_prior_mean = jax.tree_map(lambda x: x * 0 + self.prior_mean, jax.lax.stop_gradient(params_variational_mean))
                params_prior_logvar = jax.tree_map(lambda x: x * 0 + jnp.log(self.prior_var), jax.lax.stop_gradient(params_variational_logvar))

            ### remove batchnorm parameters from the KL calculation
            # params_prior_mean, _ = self.split_params(params_prior_mean, "batch_norm")
            # params_prior_logvar, _ = self.split_params(params_prior_logvar, "batch_norm")
            # params_variational_mean, _ = self.split_params(params_variational_mean, "batch_norm")
            # params_variational_logvar, _ = self.split_params(params_variational_logvar, "batch_norm")

            params_prior_var = jax.tree_map(lambda x: jnp.exp(x), params_prior_logvar)
            params_variational_var = jax.tree_map(lambda x: jnp.exp(x), params_variational_logvar)

            kl = jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(
                lambda a, b, c, d: kl_univariate_gaussians(a, b, c, d),
                params_variational_mean, params_variational_var, params_prior_mean, params_prior_var
                ))[0])
            
            return kl
        
        def calculate_sharpness_prior(params, params_epsilon, preds_f_llk, prior_likelihood_mean, rng_key, batch_stats, inputs, targets, train=True):            
            if "steepest_ascent" in self.reg_type:
                if "learned_epsilon" in self.reg_type:
                    if 'bert' in self.model_name:
                        def loss_fn(_params):
                            out = self.pred_fn(
                                input_ids=inputs[:,0],
                                token_type_ids=inputs[:,1],
                                attention_mask=inputs[:,2],
                                params=_params,
                                train=train,
                                dropout_rng=self.dropout_rng,
                            )
                            preds_f, new_model_state = out[0], None

                            loss = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f, -1), targets).mean(0).sum()

                            return loss, (preds_f, new_model_state)
                    else:
                        def loss_fn(_params):
                            out = self.pred_fn(
                                {'params': _params, 'batch_stats': batch_stats},
                                inputs,
                                train=train,
                                mutable=['batch_stats'] if train else False
                                )
                            preds_f, new_model_state = out if train else (out, None)

                            loss = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f, -1), targets).mean(0).sum()

                            return loss, (preds_f, new_model_state)
                    
                    _, grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(params_epsilon)
                    _epsilon = grads
                else:
                    if "permuted_targets" in self.reg_type:
                        if 'bert' in self.model_name:
                            def loss_fn(_params):
                                out = self.pred_fn(
                                    input_ids=inputs[:,0],
                                    token_type_ids=inputs[:,1],
                                    attention_mask=inputs[:,2],
                                    params=_params,
                                    train=train,
                                    dropout_rng=self.dropout_rng,
                                )
                                preds_f, new_model_state = out[0], None

                                targets_permuted = jnp.roll(targets, 1, axis=-1)
                                loss = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f, -1), targets_permuted).mean(0).sum()

                                return loss, (preds_f, new_model_state)
                        else:
                            def loss_fn(_params):
                                out = self.pred_fn(
                                    {'params': _params, 'batch_stats': batch_stats},
                                    inputs,
                                    train=train,
                                    mutable=['batch_stats'] if train else False
                                    )
                                preds_f, new_model_state = out if train else (out, None)

                                targets_permuted = jnp.roll(targets, 1, axis=-1)
                                loss = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f, -1), targets_permuted).mean(0).sum()

                                return loss, (preds_f, new_model_state)

                        _, grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(params)
                    else:
                        if 'bert' in self.model_name:
                            def loss_fn(_params):
                                out = self.pred_fn(
                                    input_ids=inputs[:,0],
                                    token_type_ids=inputs[:,1],
                                    attention_mask=inputs[:,2],
                                    params=_params,
                                    train=train,
                                    dropout_rng=self.dropout_rng,
                                )
                                preds_f, new_model_state = out[0], None

                                loss = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f, -1), targets).mean(0).sum()

                                return loss, (preds_f, new_model_state)
                        else:
                            def loss_fn(_params):
                                out = self.pred_fn(
                                    {'params': _params, 'batch_stats': batch_stats},
                                    inputs,
                                    train=train,
                                    mutable=['batch_stats'] if train else False
                                    )
                                preds_f, new_model_state = out if train else (out, None)

                                loss = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f, -1), targets).mean(0).sum()

                                return loss, (preds_f, new_model_state)

                        _, grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(params)

                    _epsilon = grads

                if "steepest_ascent_nondiff" in self.reg_type:
                    _epsilon = jax.lax.stop_gradient(_epsilon)

            elif "learned_epsilon" in self.reg_type:
                _epsilon = params_epsilon

            elif "random_normal" in self.reg_type:
                _epsilon = jax.tree_map(lambda x: random.normal(rng_key, x.shape), params)

            elif "random_init" in self.reg_type:
                _epsilon = self.model.init(rng_key, inputs[0:1], train=True)["params"]

            if "dropout" in self.reg_type and "steepest_ascent" not in self.reg_type and "random_normal" not in self.reg_type and "random_init" not in self.reg_type:
                params_mask = jax.tree_map(lambda x: random.bernoulli(rng_key, p=1-self.dropout_rate_sam, shape=x.shape), params)
                params_perturbed = jax.tree_map(lambda x, y: x * y, params, params_mask)
            else:
                epsilon = dual_vector(_epsilon)  # TODO: decide whether to normalize before or after dropout

                if "dropout" in self.reg_type:
                    mask = jax.tree_map(lambda x: random.bernoulli(rng_key, p=1-self.dropout_rate_sam, shape=x.shape), epsilon)
                    epsilon = jax.tree_map(lambda x, y: x * y, epsilon, mask)

                params_perturbed = jax.tree_map(lambda a, b: a + b * self.rho_sam, params, epsilon)

            if 'bert' in self.model_name:
                out = self.pred_fn(
                    input_ids=inputs[:,0],
                    token_type_ids=inputs[:,1],
                    attention_mask=inputs[:,2],
                    params=params_perturbed,
                    train=train,
                    dropout_rng=self.dropout_rng,
                )
                preds_f_reg, new_model_state_perturbed = out[0], None
            else:
                out = self.pred_fn(
                    {'params': params_perturbed, 'batch_stats': batch_stats},
                    inputs,
                    train=train,
                    mutable=['batch_stats'] if train else False
                    )
                preds_f_reg, new_model_state_perturbed = out if train else (out, None)
                
            nll_llk = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f_llk, -1), targets).mean(0)  # likelihood_scale = temperature
            nll_perturbed = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f_reg, -1), targets)  # likelihood_scale = temperature

            prior_likelihood_variance = 1 / self.n_batches_train

            if "gaussian" in self.reg_type:
                sharpness_penalty = jnp.sum(jnp.square((nll_perturbed - nll_llk) - prior_likelihood_mean) / prior_likelihood_variance)
            elif "laplace" in self.reg_type:
                sharpness_penalty = jnp.sum(jnp.abs((nll_perturbed - nll_llk) - prior_likelihood_mean) / prior_likelihood_variance)

            parameter_norm = calculate_parameter_norm(params, self.prior_var)

            reg = parameter_norm + sharpness_penalty

            if "steepest_ascent_learned_epsilon" in self.reg_type:
                reg += jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x, y: jnp.abs(x - y), params_epsilon, params))[0])

            return reg, new_model_state_perturbed

        def calculate_adversarial_prior(params, preds_f_llk, grads, prior_likelihood_mean, rng_key, batch_stats, inputs, targets, train=True):
            if 'bert' in self.model_name:
                def loss_fn(_inputs):
                    out = self.pred_fn(
                        input_ids=_inputs[:,0],
                        token_type_ids=_inputs[:,1],
                        attention_mask=_inputs[:,2],
                        params=params,
                        train=train,
                        dropout_rng=self.dropout_rng,
                    )
                    preds_f_llk, new_model_state = out[0], None

                    loss_vector = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f_llk, -1), targets).mean(0)
                    loss = loss_vector.sum()

                    return loss, (loss_vector, preds_f_llk, new_model_state)
            else:
                def loss_fn(_inputs):
                    out = self.pred_fn(
                        {'params': params, 'batch_stats': batch_stats},
                        _inputs,
                        train=train,
                        mutable=['batch_stats'] if train else False
                        )
                    preds_f_llk, new_model_state = out if train else (out, None)

                    loss_vector = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f_llk, -1), targets).mean(0)
                    loss = loss_vector.sum()

                    return loss, (loss_vector, preds_f_llk, new_model_state)

            ret, grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(inputs)
            nll_llk = ret[1][0]

            if "steepest_ascent" in self.reg_type:
                _epsilon = grads

                if "steepest_ascent_nondiff" in self.reg_type:
                    _epsilon = jax.lax.stop_gradient(_epsilon)

            if "random_normal" in self.reg_type:
                _epsilon = random.normal(rng_key, inputs.shape)

            epsilon = _epsilon / jnp.sqrt(jnp.sum(jnp.square(_epsilon)))
            inputs_perturbed = inputs + self.rho_adversarial * epsilon

            ret_perturbed = loss_fn(inputs_perturbed)[1]
            nll_perturbed, new_model_state_perturbed = ret_perturbed[0], ret_perturbed[2]

            prior_likelihood_variance = 1 / self.n_batches_train

            if "gaussian" in self.reg_type:
                sharpness_penalty = jnp.sum(jnp.square((nll_llk - nll_perturbed) - prior_likelihood_mean) / prior_likelihood_variance)
            elif "laplace" in self.reg_type:
                sharpness_penalty = jnp.sum(jnp.abs((nll_llk - nll_perturbed) - prior_likelihood_mean) / prior_likelihood_variance)

            parameter_norm = calculate_parameter_norm(params, self.prior_var)

            reg = parameter_norm + sharpness_penalty

            return reg, new_model_state_perturbed

        def f_lin(params_dict, inputs, train, mutable):
            params = params_dict["params"]
            batch_stats = params_dict["batch_stats"]

            out = self.model.apply(
                {'params': jax.lax.stop_gradient(params), 'batch_stats': batch_stats},
                inputs,
                train=train,
                mutable=['batch_stats'] if train else False
                )
            _, new_model_state = out if train else (out, None)

            if train:
                _pred_fn = lambda params: self.model.apply(
                    {'params': params, 'batch_stats': batch_stats},
                    inputs,
                    train=True,
                    mutable=['batch_stats']
                    )[0]
        
                eps = jax.tree_map(lambda x: random.normal(rng_key, x.shape), jax.lax.stop_gradient(params))
                eps_scaled = jax.tree_map(lambda x, y: x * jnp.abs(y) * self.prior_var ** 0.5, eps, params)

                pred_f, pred_jvp = jax.jvp(_pred_fn, (params,), (jax.tree_map(lambda x: x, eps_scaled),))
                preds_f = pred_f + pred_jvp
            else:
                _pred_fn = lambda params: self.model.apply(
                    {'params': params, 'batch_stats': batch_stats},
                    inputs,
                    train=False,
                    mutable=False
                    )
                pred_f = _pred_fn(params)
                preds_f = pred_f

            if train:
                return preds_f, new_model_state
            else:
                return preds_f

        def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
            """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.
            Args:
                y: A pytree of numpy ndarray, vector y in the equation above.
            """
            vector_norm = jnp.sqrt(sum(
                [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
            normalized_vector = jax.tree_map(lambda x: x / vector_norm, y)
            return normalized_vector

        def sample_parameters(params, params_logvar, stochastic, rng_key, primary_type="", secondary_type="", tertiary_type=""):
            if self.ssm:
                params_det_mean, params_random_mean = self.split_params(params, primary_type, secondary_type, tertiary_type)
                _, params_random_logvar = self.split_params(params_logvar, primary_type, secondary_type, tertiary_type)
            else:
                params_random_mean = params
                params_random_logvar = params_logvar
                
            if stochastic:
                eps = jax.tree_map(lambda x: random.normal(rng_key, x.shape), params_random_logvar)
                params_std_sample = jax.tree_map(lambda x, y: x * jnp.exp(y) ** 0.5, eps, params_random_logvar)
                params_sample = jax.tree_map(lambda x, y: x + y, params_random_mean, params_std_sample)
            else:
                params_sample = params

            if self.ssm:
                params_sample = merge_params(params_det_mean, params_sample)

            return params_sample

        def calculate_forward_pass(params, params_logvar, rng_key, batch_stats, inputs, _inputs_context, train):
            if self.linearize and self.pred_fn is None:
                self.pred_fn = f_lin
            else:
                if self.ssm:
                    model = self.model_init(training=True)
                    self.pred_fn = model.apply
                elif 'bert' in self.model_name:
                    self.pred_fn = self.model.__call__
                    assert (
                        self.objective_hparams["forward_points"] == "train" and self.objective_hparams["reg_points"] == "train"
                        or
                        self.objective_hparams["forward_points"] == "joint" and self.objective_hparams["reg_points"] == "context"
                        )
                else:
                    self.pred_fn = self.model.apply

            preds_f_llk_list = []
            preds_f_reg_list = []
            params_samples = []
            for _ in range(self.mc_samples_llk):
                rng_key, _ = jax.random.split(rng_key)

                if self.objective_hparams["stochastic"]:
                    params = sample_parameters(params, params_logvar, self.stochastic, rng_key, primary_type=self.primary_type, secondary_type=self.secondary_type, tertiary_type=self.tertiary_type)
                    params_samples.append(params)

                if (
                    self.objective_hparams["forward_points"] == "train" and
                    self.objective_hparams["reg_points"] == "train"
                    ):
                    inputs_forward = inputs  # inputs used to update batch stats
                    inputs_reg = inputs_forward

                    if not self.ssm:
                        if 'bert' in self.model_name:
                            out = self.pred_fn(
                                input_ids=inputs_forward[:, 0],
                                token_type_ids=inputs_forward[:, 1],
                                attention_mask=inputs_forward[:, 2],
                                params=params,
                                train=train,
                                dropout_rng= self.dropout_rng,
                            )
                            preds_f_llk, new_model_state = out[0], None
                        else:    
                            # a forward pass on the training points for the log-likelihood and regularization terms (batch stats are updated)
                            out = self.pred_fn(
                                {'params': params, 'batch_stats': batch_stats},
                                inputs_forward,
                                train=train,
                                mutable=['batch_stats'] if train else False
                                )
                            preds_f_llk, new_model_state = out if train else (out, None)
                    else:
                        out, mod_vars = self.pred_fn(
                            {"params": params},
                            inputs_forward,
                            rngs={"dropout": rng_key},
                            mutable=["intermediates"],
                        )
                        preds_f_llk = out[0]
                        new_model_state = None

                    preds_f_reg = preds_f_llk

                elif (
                    self.objective_hparams["forward_points"] == "train" and
                    self.objective_hparams["reg_points"] != "train"
                    ):
                    inputs_forward = inputs  # inputs used to update batch stats

                    if not self.ssm:
                        # a forward pass on the training points for the log-likelihood term (batch stats are updated)
                        out = self.pred_fn(
                            {'params': params, 'batch_stats': batch_stats},
                            inputs_forward,
                            train=train,
                            mutable=['batch_stats'] if train else False
                            )
                        preds_f_llk, new_model_state = out if train else (out, None)
                    else:
                        out, mod_vars = self.pred_fn(
                            {"params": params},
                            inputs_forward,
                            rngs={"dropout": rng_key},
                            mutable=["intermediates"],
                        )
                        preds_f_llk = out[0]
                        new_model_state = None

                    if self.objective_hparams["reg_points"] == "context":
                        inputs_reg = _inputs_context

                        if not self.ssm:
                            # a forward pass on the context points for the regularization term (batch stats are not updated)
                            out = self.pred_fn(
                                {'params': params, 'batch_stats': batch_stats},
                                inputs_reg,
                                train=train,
                                mutable=['batch_stats'] if train else False
                                )
                            _preds_f_reg, _ = out if train else (out, None)
                        else:
                            out[0], mod_vars = self.pred_fn(
                                {"params": params},
                                inputs_reg,
                                rngs={"dropout": rng_key},
                                mutable=["intermediates"],
                            )
                            _preds_f_reg = out[0]
                            new_model_state = None

                        preds_f_reg = jnp.concatenate([preds_f_llk, _preds_f_reg], axis=0)

                    elif self.objective_hparams["reg_points"] == "joint":
                        inputs_reg = jnp.concatenate([inputs, _inputs_context], axis=0)

                        if not self.ssm:
                            # a forward pass on the joint points (context + train) for the regularization term (batch stats are not updated)
                            out = self.pred_fn(
                                {'params': params, 'batch_stats': batch_stats},
                                inputs_reg,
                                train=train,
                                mutable=['batch_stats'] if train else False
                                )
                            preds_f_reg, _ = out if train else (out, None)
                        else:
                            out, mod_vars = self.pred_fn(
                                {"params": params},
                                inputs_reg,
                                rngs={"dropout": rng_key},
                                mutable=["intermediates"],
                            )
                            preds_f_reg = out[0]
                            new_model_state = None

                    else:
                        raise ValueError("Unknown forward_points/reg_points/context_points combination.")

                elif self.objective_hparams["forward_points"] == "joint":
                    inputs_forward = jnp.concatenate([inputs, _inputs_context], axis=0)  # inputs used to update batch stats

                    if not self.ssm:
                        # a forward pass on both training and context points (batch stats are updated)
                        if 'bert' in self.model_name:
                            out = self.pred_fn(
                                input_ids=inputs_forward[:, 0],
                                token_type_ids=inputs_forward[:, 1],
                                attention_mask=inputs_forward[:, 2],
                                params=params,
                                train=train,
                                dropout_rng= self.dropout_rng,
                            )
                            preds_f_joint, new_model_state = out[0], None
                        else:
                            out = self.pred_fn(
                                {'params': params, 'batch_stats': batch_stats},
                                inputs_forward,
                                train=train,
                                mutable=['batch_stats'] if train else False
                                )
                            preds_f_joint, new_model_state = out if train else (out, None)
                    else:
                        out, mod_vars = self.pred_fn(
                            {"params": params},
                            inputs_forward,
                            rngs={"dropout": rng_key},
                            mutable=["intermediates"],
                        )
                        preds_f_joint = out[0]
                        new_model_state = None

                    preds_f_llk = preds_f_joint[:inputs.shape[0]]

                    if self.objective_hparams["reg_points"] == "context":
                        inputs_reg = _inputs_context
                        preds_f_reg = preds_f_joint[-inputs_reg.shape[0]:]

                    elif self.objective_hparams["reg_points"] == "joint":
                        inputs_reg = jnp.concatenate([inputs, _inputs_context], axis=0)
                        preds_f_reg = preds_f_joint

                    else:
                        raise ValueError("Unknown forward_points/reg_points/context_points combination.")

                else:
                    raise ValueError("Unknown forward_points/reg_points/context_points combination.")
                    
                preds_f_llk_list.append(preds_f_llk)
                preds_f_reg_list.append(preds_f_reg)

            preds_f_llk = jnp.stack(preds_f_llk_list, axis=0)
            preds_f_reg = jnp.stack(preds_f_reg_list, axis=0)

            return preds_f_llk, preds_f_reg, params_samples, new_model_state, inputs_reg

        def calculate_loss(params, params_logvar, params_epsilon, epsilon, rng_key, batch_stats, batch, batch_context, train, group_attr=None, group_attr_context=None, gdro_weights=None):
            inputs, targets = batch
            _inputs_context, targets_reg = batch_context

            if self.rho_sam > 0 and ("sharpness_prior" not in self.reg_type and "group_dro_prior" not in self.reg_type):
                params = jax.tree_map(lambda a, b: a + b * self.rho_sam, params, epsilon)

            preds_f_llk, preds_f_reg, params_samples, new_model_state, inputs_reg = calculate_forward_pass(params, params_logvar, rng_key, batch_stats, inputs, _inputs_context, train)

            if self.reg_type == "parameter_kl":
                assert self.objective_hparams["stochastic"] == True
                reg = calculate_parameter_kl(params, params_logvar)
            elif self.reg_type == "function_kl":
                reg = calculate_function_kl(params, params_logvar, inputs_reg, batch_stats, rng_key)
            elif self.reg_type == "function_prior":
                reg = calculate_function_prior_density(preds_f_reg, params, inputs_reg, batch_stats, rng_key, self.prior_var)
            elif self.reg_type == "entropic_prior":
                reg = calculate_entropic_prior(preds_f_reg)
            elif self.reg_type == "doubly_entropic_prior":
                reg = calculate_entropic_prior(preds_f_reg)
                reg += calculate_function_prior_density(preds_f_reg, params, inputs_reg, batch_stats, rng_key)
            elif self.reg_type == "empirical_gaussian_prior":
                reg = calculate_function_prior_density(preds_f_reg, params, inputs_reg, batch_stats, rng_key, self.prior_likelihood_scale)
                reg += calculate_parameter_norm(params, self.prior_var)
            elif self.reg_type == "empirical_categorical_prior":
                reg = calculate_empirical_categorical_prior(preds_f_reg, self.prediction_type)
                reg += calculate_parameter_norm(params, self.prior_var)
            elif self.reg_type == "empirical_gaussian_prior_density":
                reg = calculate_empirical_gaussian_prior_density(preds_f_reg, params, inputs_reg, batch_stats, self.prior_likelihood_cov_scale, self.prior_likelihood_cov_diag, rng_key)
            elif self.reg_type == "empirical_gaussian_prior_kl":
                reg = calculate_empirical_gaussian_prior_kl(params_samples, params, params_logvar, preds_f_reg, inputs_reg, batch_stats, self.prior_likelihood_scale, self.prior_likelihood_f_scale, self.prior_likelihood_cov_scale, self.prior_likelihood_cov_diag, rng_key)
            elif self.reg_type == "empirical_categorical_prior_kl":
                reg = calculate_empirical_categorical_prior_kl(params_samples, params_logvar, preds_f_reg, self.prior_var, self.prediction_type)
            elif self.reg_type == "empirical_function_parameter_norm":
                reg = calculate_function_norm(preds_f_reg, inputs_reg, batch_stats, self.prior_var)
                reg += calculate_parameter_norm(params, self.prior_var)
            elif self.reg_type == "empirical_fairness_prior":
                reg = calculate_empirical_fairness(params, preds_f_llk, preds_f_reg, targets, self.prior_var)
            elif "group_dro_prior" in self.reg_type:
                if "context" in self.reg_type:
                    reg, gdro_weights, new_model_state_perturbed = calculate_group_dro_prior(params, preds_f_reg, preds_f_reg, batch_stats, inputs_reg, targets_reg, group_attr_context, num_groups, gdro_weights, self.prior_likelihood_scale)
                else:
                    reg, gdro_weights, new_model_state_perturbed = calculate_group_dro_prior(params, preds_f_llk, preds_f_reg, batch_stats, inputs, targets, group_attr, num_groups, gdro_weights, self.prior_likelihood_scale)
                if self.method == 'psmap':
                    reg += calculate_parameter_norm(params, self.prior_var)
                elif self.method == 'psvi':
                    reg += calculate_parameter_kl(params, params_logvar)
            elif self.reg_type == "function_norm":
                reg = calculate_function_norm(preds_f_reg, inputs_reg, batch_stats, self.prior_var)
            elif self.reg_type == "parameter_norm":
                reg = calculate_parameter_norm(params, self.prior_var)
            elif "sharpness_prior" in self.reg_type:
                reg, new_model_state_perturbed = calculate_sharpness_prior(params, params_epsilon, preds_f_reg, self.prior_likelihood_mean, rng_key, batch_stats, inputs_reg, targets_reg, train)
            elif "adversarial_prior" in self.reg_type:
                reg, new_model_state_perturbed = calculate_adversarial_prior(params, preds_f_llk, self.prior_likelihood_mean, rng_key, batch_stats, inputs, targets, train)
            else:
                raise ValueError("Unknown regularization type.")
            # reg = 0

            scale = 1 / self.n_batches_train  # 1 / (number of mini-batches)
            reg = scale * reg

            ### linearized loss SAM (does not currently work)
            # if self.rho_sam > 0:
            #     assert self.likelihood_scale == 1
            #     pred_fn = lambda params: self.model.apply({'params': params, 'batch_stats': batch_stats},
            #                             inputs,
            #                             train=True,
            #                             mutable=['batch_stats'])[0]

            #     _, jvp_times_epsilon = jax.jvp(pred_fn, (params,), (jax.tree_map(lambda x: x * self.rho_sam, epsilon),))
            #     jvp_times_epsilon = jax.lax.stop_gradient(jvp_times_epsilon)

            #     softmax_scaling = jax.lax.stop_gradient(jnp.prod(jnp.mean(jax.nn.softmax(preds_f_llk, -1), axis=0), axis=-1)[:, None])
            #     vjp_fn = jax.vjp(pred_fn, jax.lax.stop_gradient(params))[1]
            #     jac_inner_times_epsilon = vjp_fn(jvp_times_epsilon * softmax_scaling)[0]  # constant wrt params
            #     reg_times_params = jax.tree_map(lambda x, y: x * y, jac_inner_times_epsilon, params)  # linear in params
            #     reg = jnp.sum(jax.flatten_util.ravel_pytree(reg_times_params)[0])

            if self.output_var:
                likelihood_scale = preds_f_llk[:, :, self.num_classes:]
                preds_f_llk = preds_f_llk[:, :, :self.num_classes]
            else:
                likelihood_scale = self.likelihood_scale

            if self.group_dro and "group_dro_prior" not in self.reg_type:
                per_sample_nll = categorical_nll_with_softmax(jax.nn.softmax((1 / likelihood_scale) * preds_f_llk, -1), targets).mean(0)
                nll, gdro_weights = calculate_group_loss(per_sample_nll, group_attr, num_groups, gdro_weights)
            elif self.prediction_type == "classification":
                nll = categorical_nll_with_softmax(jax.nn.softmax((1 / likelihood_scale) * preds_f_llk, -1), targets).mean(0).sum()  # likelihood_scale = temperature
            elif self.prediction_type == "regression":
                nll = gaussian_nll(preds_f_llk, targets, likelihood_scale).mean(0).sum()  # likelihood_scale = likelihood variance
            else:
                raise ValueError("Unknown prediction type.")

            loss = (nll + self.reg_scale * reg) / self.batch_size  # per data point loss
            
            if self.prediction_type == "classification":
                acc = 100 * (preds_f_llk.argmax(axis=-1) == targets).mean()
            elif self.prediction_type == "regression":
                acc = jnp.mean(jnp.mean(jnp.square(jnp.mean(preds_f_llk, axis=0) - targets), axis=-1), axis=-1)
            else:
                raise ValueError("Unknown prediction type.")

            if self.debug_print_updated:
                jax.debug.print("nll: {}", nll)
                jax.debug.print("reg: {}", reg)
                jax.debug.print("loss: {}", loss)
                jax.debug.print("acc: {}", acc)

            if self.group_dro:
                return loss, (acc, new_model_state, gdro_weights)
            else:
                return loss, (acc, new_model_state)

        @partial(jit, static_argnums=(5,))
        def pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key, feature=False):
            params = sample_parameters(params, params_logvar, self.stochastic, rng_key, primary_type=self.primary_type, secondary_type=self.secondary_type, tertiary_type=self.tertiary_type)

            if not self.ssm:
                if 'bert' in self.model_name:
                    out = self.pred_fn(
                        input_ids=inputs[:,0],
                        token_type_ids=inputs[:,1],
                        attention_mask=inputs[:,2],
                        params=params,
                        train=False,
                        dropout_rng=self.dropout_rng,
                    )

                    preds_f, feature = out[0], None
                    if feature:
                        return preds_f, feature
                    else:
                        return preds_f
                elif feature:
                    preds_f, feature = self.pred_fn(
                        {'params': params, 'batch_stats': batch_stats},
                        inputs,
                        train=False,
                        mutable=False,
                        feature=feature
                        )
                    return preds_f, feature        
                else:
                    preds_f = self.pred_fn(
                        {'params': params, 'batch_stats': batch_stats},
                        inputs,
                        train=False,
                        mutable=False
                        )
                    return preds_f
            else:
                out, mod_vars = self.pred_fn(
                    {"params": params},
                    inputs,
                    rngs={"dropout": rng_key},
                    mutable=["intermediates"],
                    )
                if feature:
                    preds_f, feature = out
                    return preds_f, feature        
                else:
                    preds_f = out[0]
                    return preds_f

        def evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, type):
            if type == "test":
                _logits_test = []
                _targets_test = []

                for i, batch in enumerate(test_loader):
                    inputs_test = np.array(batch[0], dtype=np.float32)
                    _targets = np.array(batch[1], dtype=np.float32)
                    inputs = inputs_test
                    _logits_test_list = []
                    for _ in range(self.mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        _logits_test_list.append(pred)
                    _logits_test.append(jnp.stack(_logits_test_list, axis=0))
                    _targets_test.append(_targets)
                    if i == n_batches_eval:
                        break
                logits_test = jnp.concatenate(_logits_test, axis=1)[:, :testset_size, :]
                targets_test = jnp.concatenate(_targets_test, axis=0)

                ret = [logits_test, targets_test]

            elif type == "train":
                _logits_train = []
                _targets_train = []
                _sens_train = []

                for i, batch in enumerate(train_loader):
                    inputs_train = np.array(batch[0], dtype=np.float32)
                    _targets = np.array(batch[1], dtype=np.float32)
                    sens = np.array(batch[2], dtype=np.float32)
                    inputs = inputs_train
                    _logits_train_list = []
                    for _ in range(self.mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        _logits_train_list.append(pred)
                    _logits_train.append(jnp.stack(_logits_train_list, axis=0))
                    _targets_train.append(_targets)
                    _sens_train.append(sens)
                    if i == n_batches_eval:
                        break
                logits_train = jnp.concatenate(_logits_train, axis=1)[:, :training_dataset_size, :]
                targets_train = jnp.concatenate(_targets_train, axis=0)
                sens_train = jnp.concatenate(_sens_train, axis=0)

                ret = [logits_train, targets_train, sens_train]

            elif type == "fairtest":
                # doesn't use n_batches_eval because we might 
                # miss out on some extreme minority sensitive groups
                _logits_test = []
                _targets_test = []
                _sens_test = []

                for i, batch in enumerate(test_loader):
                    inputs_test = np.array(batch[0], dtype=np.float32)
                    _targets = np.array(batch[1], dtype=np.float32)
                    sens = np.array(batch[2], dtype=np.float32)
                    inputs = inputs_test
                    _logits_test_list = []
                    for _ in range(self.mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        _logits_test_list.append(pred)
                    _logits_test.append(jnp.stack(_logits_test_list, axis=0))
                    _targets_test.append(_targets)
                    _sens_test.append(sens)
                logits_test = jnp.concatenate(_logits_test, axis=1)[:, :testset_size, :]
                targets_test = jnp.concatenate(_targets_test, axis=0)
                sens_test = jnp.concatenate(_sens_test, axis=0)

                ret = [logits_test, targets_test, sens_test]

            elif type == "validation":
                # doesn't use n_batches_eval because we might 
                # miss out on some extreme minority sensitive groups
                _logits_val = []
                _targets_val = []
                _sens_val = []

                for i, batch in enumerate(val_loader):
                    inputs_val = np.array(batch[0], dtype=np.float32)
                    _targets = np.array(batch[1], dtype=np.float32)
                    sens = np.array(batch[2], dtype=np.float32)
                    inputs = inputs_val
                    _logits_val_list = []
                    for _ in range(self.mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        _logits_val_list.append(pred)
                    _logits_val.append(jnp.stack(_logits_val_list, axis=0))
                    _targets_val.append(_targets)
                    _sens_val.append(sens)
                logits_val = jnp.concatenate(_logits_val, axis=1)
                targets_val = jnp.concatenate(_targets_val, axis=0)
                sens_val = jnp.concatenate(_sens_val, axis=0)

                ret = [logits_val, targets_val, sens_val]

            elif type == "context":
                _logits_context = []
                for i, batch in enumerate(context_loader):
                    inputs_context = np.array(batch[0], dtype=np.float32)
                    n_context_points = inputs_context.shape[0]
                    inputs = jnp.concatenate([inputs_context, inputs_context], axis=0)
                    _logits_context_list = []
                    for _ in range(self.mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        _pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        pred = _pred[:_pred.shape[0] - n_context_points]
                        _logits_context_list.append(pred)
                    _logits_context.append(jnp.stack(_logits_context_list, axis=0))
                    if i == self.n_batches_eval_context:
                        break
                logits_context = jnp.concatenate(_logits_context, axis=1)

                ret = [logits_context, None]

            elif type == "ood":
                _logits_ood = []
                for i, batch in enumerate(ood_loader):
                    inputs_ood = np.array(batch[0], dtype=np.float32)
                    inputs = inputs_ood
                    _logits_ood_list = []
                    for _ in range(self.mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        _logits_ood_list.append(pred)
                    _logits_ood.append(jnp.stack(_logits_ood_list, axis=0))
                    if i == n_batches_eval:
                        break
                logits_ood = jnp.concatenate(_logits_ood, axis=1)

                ret = [logits_ood, None]

            if type == "cifar101":
                _logits_test = []
                _targets_test = []

                for i, batch in enumerate(cifar101test_loader):
                    inputs_test = np.array(batch[0], dtype=np.float32)
                    _targets = np.array(batch[1], dtype=np.float32)
                    inputs = inputs_test
                    _logits_test_list = []
                    for _ in range(self.mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        _logits_test_list.append(pred)
                    _logits_test.append(jnp.stack(_logits_test_list, axis=0))
                    _targets_test.append(_targets)
                    if i == n_batches_eval:
                        break
                logits_test = jnp.concatenate(_logits_test, axis=1)[:, :testset_size, :]
                targets_test = jnp.concatenate(_targets_test, axis=0)

                ret = [logits_test, targets_test]

            if type == "corruptedcifar10":
                _logits_test_list_full = []
                _targets_test_list_full = []

                for ccifar10test_loader in ccifar10test_loader_list:
                    _logits_test = []
                    _targets_test = []
                    for i, batch in enumerate(ccifar10test_loader):
                        inputs_test = np.array(batch[0], dtype=np.float32)
                        _targets = np.array(batch[1], dtype=np.float32)
                        inputs = inputs_test
                        _logits_test_list = []
                        for _ in range(self.mc_samples_eval):
                            rng_key, _ = jax.random.split(rng_key)
                            pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                            _logits_test_list.append(pred)
                        _logits_test.append(jnp.stack(_logits_test_list, axis=0))
                        _targets_test.append(_targets)
                        if i == n_batches_eval:
                            break
                    logits_test = jnp.concatenate(_logits_test, axis=1)[:, :testset_size, :]
                    targets_test = jnp.concatenate(_targets_test, axis=0)[:testset_size]
                    _logits_test_list_full.append(logits_test)
                    _targets_test_list_full.append(targets_test)

                ret = [_logits_test_list_full, _targets_test_list_full]

            return ret

        def calculate_metrics(params, params_logvar, rng_key, batch_stats, n_batches_eval, full_eval):
            if not self.quick_eval:
                if fairness_eval:
                    logits_test, targets_test, sens_test = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "fairtest")
                    logits_train, targets_train, sens_train = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "train")
                else:
                    logits_test, targets_test = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "test")
                assert not np.any(np.isnan(logits_test)), f"logits_test contains NaNs."

                if validation_training:
                    logits_val, targets_val, sens_val = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "validation")

                if not self.fairness_eval:
                    logits_context, _ = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "context")
                    assert not np.any(np.isnan(logits_context)), f"logits_context contains NaNs."

                    logits_ood, _ = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "ood")
                    assert not np.any(np.isnan(logits_ood)), f"logits_ood contains NaNs."

                if "cifar10" in self.dataset and "cifar100" not in self.dataset:
                    logits_cifar101, targets_cifar101 = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "cifar101")

                    acc_test_cifar101 = 100 * np.array(np.mean(jax.nn.softmax(logits_cifar101, axis=-1).mean(0).argmax(axis=-1) == targets_cifar101))
                    acc_sel_test_cifar101 = selective_accuracy(jax.nn.softmax(logits_cifar101, axis=-1).mean(0), targets_cifar101)
                    nll_test_cifar101 = float(categorical_nll_with_softmax(jax.nn.softmax(logits_cifar101, -1).mean(0), targets_cifar101).mean())
                    ece_test_cifar101 = 100 * calibration(jax.nn.one_hot(targets_cifar101, self.num_classes), jax.nn.softmax(logits_cifar101, axis=-1).mean(0))[0]

                    self.logger["acc_test_cifar101"].append(acc_test_cifar101)
                    self.logger["acc_sel_test_cifar101"].append(acc_sel_test_cifar101)
                    self.logger["nll_test_cifar101"].append(nll_test_cifar101)
                    self.logger["ece_test_cifar101"].append(ece_test_cifar101)

                if full_eval:
                    if "cifar10" in self.dataset and "cifar100" not in self.dataset:
                        logits_ccifar10_list, targets_ccifar10_list = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "corruptedcifar10")

                        for i, corr_config in enumerate(corr_config_list):
                            acc_test_ccifar10 = 100 * np.array(np.mean(jax.nn.softmax(logits_ccifar10_list[i], axis=-1).mean(0).argmax(axis=-1) == targets_ccifar10_list[i]))
                            acc_sel_test_ccifar10 = selective_accuracy(jax.nn.softmax(logits_ccifar10_list[i], axis=-1).mean(0), targets_ccifar10_list[i])
                            self.logger[f"acc_test_ccifar10_{corr_config}"].append(acc_test_ccifar10)
                            self.logger[f"acc_sel_test_ccifar10_{corr_config}"].append(acc_sel_test_ccifar10)

                acc_test = 100 * np.array(np.mean(jax.nn.softmax(logits_test, axis=-1).mean(0).argmax(axis=-1) == targets_test))
                acc_sel_test = selective_accuracy(jax.nn.softmax(logits_test, axis=-1).mean(0), targets_test)
                nll_test = float(categorical_nll_with_softmax(jax.nn.softmax(logits_test, -1).mean(0), targets_test).mean())
                ece_test = 100 * calibration(jax.nn.one_hot(targets_test, self.num_classes), jax.nn.softmax(logits_test, axis=-1).mean(0))[0]

                if validation_training:
                    acc_val = 100 * np.array(np.mean(jax.nn.softmax(logits_val, axis=-1).mean(0).argmax(axis=-1) == targets_val))

                predictive_entropy_test = float(categorical_entropy(jax.nn.softmax(logits_test, -1).mean(0)).mean(0))
                aleatoric_uncertainty_test = float(categorical_entropy(jax.nn.softmax(logits_test, -1)).mean(0).mean(0))
                epistemic_uncertainty_test = predictive_entropy_test - aleatoric_uncertainty_test

                if not fairness_eval:
                    acc_sel_test_ood = selective_accuracy_test_ood(
                        jax.nn.softmax(logits_test, axis=-1).mean(0),
                        jax.nn.softmax(logits_ood, axis=-1).mean(0),
                        targets_test
                        )

                    ood_auroc_entropy = 100 * auroc_logits(logits_test, logits_ood, score="entropy", rng_key=rng_key)
                    ood_auroc_aleatoric = 100 * auroc_logits(logits_test, logits_ood, score="expected entropy", rng_key=rng_key)
                    ood_auroc_epistemic = 100 * auroc_logits(logits_test, logits_ood, score="mutual information", rng_key=rng_key)

                    predictive_entropy_context = float(categorical_entropy(jax.nn.softmax(logits_context, -1).mean(0)).mean(0))
                    predictive_entropy_ood = float(categorical_entropy(jax.nn.softmax(logits_ood, -1).mean(0)).mean(0))
                    aleatoric_uncertainty_context = float(categorical_entropy(jax.nn.softmax(logits_context, -1)).mean(0).mean(0))
                    aleatoric_uncertainty_ood = float(categorical_entropy(jax.nn.softmax(logits_ood, -1)).mean(0).mean(0))
                    epistemic_uncertainty_context = predictive_entropy_context - aleatoric_uncertainty_context
                    epistemic_uncertainty_ood = predictive_entropy_ood - aleatoric_uncertainty_ood
                else:
                    acc_sel_test_ood = 0
                    ood_auroc_entropy = 0
                    ood_auroc_aleatoric = 0
                    ood_auroc_epistemic = 0
                    predictive_entropy_context = 0
                    predictive_entropy_ood = 0
                    aleatoric_uncertainty_context = 0
                    aleatoric_uncertainty_ood = 0
                    epistemic_uncertainty_context = 0
                    epistemic_uncertainty_ood = 0

                if validation_training:
                    self.logger["acc_val"].append(acc_val)

                if fairness_eval:
                    weighted_acc_test = compute_weighted_test_accuracy(targets_test, jax.nn.softmax(logits_test, axis=-1).mean(0).argmax(axis=-1), sens_test, train_dataset_group_proportions, num_labels=num_classes, num_sensitive=num_sensitive)
                    worst_group_acc_train = compute_worst_group_accuracy(targets_train, jax.nn.softmax(logits_train, axis=-1).mean(0).argmax(axis=-1), sens_train, num_labels=num_classes, num_sensitive=num_sensitive)
                    worst_group_acc_test = compute_worst_group_accuracy(targets_test, jax.nn.softmax(logits_test, axis=-1).mean(0).argmax(axis=-1), sens_test, num_labels=num_classes, num_sensitive=num_sensitive)
                    diff_in_eop = compute_equality_of_opportunity(targets_test, jax.nn.softmax(logits_test, axis=-1).mean(0).argmax(axis=-1), sens_test, num_labels=num_classes, num_sensitive=num_sensitive)
                    diff_in_eoo = compute_equality_of_odds(targets_test, jax.nn.softmax(logits_test, axis=-1).mean(0).argmax(axis=-1), sens_test, num_labels=num_classes, num_sensitive=num_sensitive)
                    diff_in_prp = compute_predictive_rate_parity(targets_test, jax.nn.softmax(logits_test, axis=-1).mean(0).argmax(axis=-1), sens_test, num_labels=num_classes, num_sensitive=num_sensitive)

                    self.logger["worst_group_acc_train"].append(100*worst_group_acc_train)
                    self.logger["worst_group_acc_test"].append(100*worst_group_acc_test)
                    self.logger["group_weighted_acc_test"].append(100*weighted_acc_test)
                    self.logger["diff_in_eop"].append(100*diff_in_eop)
                    self.logger["diff_in_eoo"].append(100*diff_in_eoo)
                    self.logger["diff_in_prp"].append(100*diff_in_prp)

                    if validation_training:
                        weighted_acc_val = compute_weighted_test_accuracy(targets_val, jax.nn.softmax(logits_val, axis=-1).mean(0).argmax(axis=-1), sens_val, validation_dataset_group_proportions, num_labels=num_classes, num_sensitive=num_sensitive)
                        worst_group_acc_val = compute_worst_group_accuracy(targets_val, jax.nn.softmax(logits_val, axis=-1).mean(0).argmax(axis=-1), sens_val, num_labels=num_classes, num_sensitive=num_sensitive)

                        self.logger["worst_group_acc_val"].append(100*worst_group_acc_val)
                        self.logger["group_weighted_acc_val"].append(100*weighted_acc_val)                

                self.logger["acc_test"].append(acc_test)
                self.logger["acc_sel_test"].append(acc_sel_test)
                self.logger["acc_sel_test_ood"].append(acc_sel_test_ood)
                self.logger["nll_test"].append(nll_test)
                self.logger["ece_test"].append(ece_test)
                self.logger["ood_auroc_entropy"].append(ood_auroc_entropy)
                self.logger["ood_auroc_aleatoric"].append(ood_auroc_aleatoric)
                self.logger["ood_auroc_epistemic"].append(ood_auroc_epistemic)
                self.logger["predictive_entropy_test"].append(predictive_entropy_test)
                self.logger["predictive_entropy_context"].append(predictive_entropy_context)
                self.logger["predictive_entropy_ood"].append(predictive_entropy_ood)
                self.logger["aleatoric_uncertainty_test"].append(aleatoric_uncertainty_test)
                self.logger["aleatoric_uncertainty_context"].append(aleatoric_uncertainty_context)
                self.logger["aleatoric_uncertainty_ood"].append(aleatoric_uncertainty_ood)
                self.logger["epistemic_uncertainty_test"].append(epistemic_uncertainty_test)
                self.logger["epistemic_uncertainty_context"].append(epistemic_uncertainty_context)
                self.logger["epistemic_uncertainty_ood"].append(epistemic_uncertainty_ood)
            else:
                logits_test, targets_test, sens_test = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "fairtest")
                acc_test = 100 * np.array(np.mean(jax.nn.softmax(logits_test, axis=-1).mean(0).argmax(axis=-1) == targets_test))
                nll_test = float(categorical_nll_with_softmax(jax.nn.softmax(logits_test, -1).mean(0), targets_test).mean())
                ece_test = 100 * calibration(jax.nn.one_hot(targets_test, self.num_classes), jax.nn.softmax(logits_test, axis=-1).mean(0))[0]
                worst_group_acc_test = compute_worst_group_accuracy(targets_test, jax.nn.softmax(logits_test, axis=-1).mean(0).argmax(axis=-1), sens_test, num_labels=num_classes, num_sensitive=num_sensitive)

                for key in self.logger.keys():
                    self.logger[key].append(0)

                self.logger["acc_test"][-1] = acc_test
                self.logger["nll_test"][-1] = nll_test
                self.logger["ece_test"][-1] = ece_test
                self.logger["worst_group_acc_test"][-1] = worst_group_acc_test

        @partial(jit, static_argnums=(4,))
        def train_step(state, batch, batch_context, rng_key, debug_print, group_attr=None, group_attr_context=None, gdro_weights=None):
            self.debug_print_updated = debug_print

            params = state.params["params"]
            params_logvar = state.params["params_logvar"]
            params_epsilon = state.params["params_epsilon"]

            if self.rho_sam != 0 and ("sharpness_prior" not in self.reg_type and "group_dro_prior" not in self.reg_type):
                assert self.prediction_type == "classification" and self.reg_type == "parameter_norm"

                inputs, targets = batch
                _inputs_context, _ = batch_context
                if 'bert' in self.model_name:
                    def loss_fn(_params):
                        out = self.pred_fn(
                            input_ids=inputs[:,0],
                            token_type_ids=inputs[:,1],
                            attention_mask=inputs[:,2],
                            params=_params,
                            train=True,
                            dropout_rng=self.dropout_rng,
                        )
                        preds_f, new_model_state = out

                        loss = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f, -1), targets).mean(0).sum()

                        return loss, (preds_f, new_model_state)
                else:
                    def loss_fn(_params):
                        out = self.pred_fn(
                            {'params': _params, 'batch_stats': state.batch_stats},
                            inputs,
                            train=True,
                            mutable=['batch_stats']
                            )
                        preds_f, new_model_state = out

                        loss = categorical_nll_with_softmax(jax.nn.softmax((1 / self.likelihood_scale) * preds_f, -1), targets).mean(0).sum()

                        return loss, (preds_f, new_model_state)

                _, grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(params)

                epsilon = dual_vector(grads)
            else:
                epsilon = 0
            
            loss_fn = lambda params, params_logvar, params_epsilon: calculate_loss(params, params_logvar, params_epsilon, epsilon, rng_key, state.batch_stats, batch, batch_context, train=True, group_attr=group_attr, group_attr_context=group_attr_context, gdro_weights=gdro_weights)
            # Get loss, gradients for loss, and other outputs of loss function
            if self.group_dro:
                ret, _grads = jax.value_and_grad(loss_fn, argnums=(0,1,2,), has_aux=True)(params, params_logvar, params_epsilon)
                gdro_weights = ret[1][2]
                ret = (ret[0], ret[1][:-1])
            else:
                ret, _grads = jax.value_and_grad(loss_fn, argnums=(0,1,2,), has_aux=True)(params, params_logvar, params_epsilon)
            grads, grads_logvar, grads_epsilon = _grads[0], jax.tree_map(lambda x: self.learning_rate_scale_logvar * x, _grads[1]), _grads[2]

            if self.final_layer_retraining:
                if "bert" in self.model_name:
                    grads['bert'] = jax.tree_map(lambda x: x * 0, grads['bert'])            

            loss, acc, new_model_state = ret[0], *ret[1] 
            # Update parameters and batch statistics
            if 'bert' in self.model_name:
                state = state.apply_gradients(grads={"params": grads, "params_logvar": grads_logvar, "params_epsilon": grads_epsilon})
            elif not self.ssm:
                state = state.apply_gradients(grads=freeze({"params": grads, "params_logvar": grads_logvar, "params_epsilon": grads_epsilon}), batch_stats=new_model_state['batch_stats'])
            else:
                state = state.apply_gradients(grads={"params": grads, "params_logvar": grads_logvar})
            if self.group_dro:
                return state, loss, acc, gdro_weights
            else:
                return state, loss, acc

        def eval_step(state, rng_key, n_batches_eval, full_eval):
            calculate_metrics(state.params["params"], state.params["params_logvar"], rng_key, state.batch_stats, n_batches_eval, full_eval)
    
        self.train_step = train_step
        self.evaluation_predictions = evaluation_predictions
        self.eval_step = eval_step
        self.pred_fn_jit = pred_fn_jit

    def init_model(self, exmp_inputs):
        init_rng = jax.random.PRNGKey(self.seed)
        init_rng_logvar, _ = random.split(init_rng)

        self.optimizer_hparams.pop("learning_rate")
        self.optimizer_hparams.pop("learning_rate_scale_logvar")
        self.optimizer_hparams.pop("alpha")
        self.ssm_optimizer_hparams.pop('optimizer_name')
        self.ssm_optimizer_hparams.pop('learning_rate')
        self.ssm_optimizer_hparams.pop('alpha')

        if self.ssm:
            self.state = self.create_train_state(
                init_rng,
                self.model_init, 
                train_loader,
                learning_rate_layer=self.learning_rate_layer,
                total_steps=len(train_loader) * self.num_epochs,
            )
            self.final_layer_key = "decoder"
        elif 'bert' in model_name:
            print('Using BERT init.')
            self.init_rng, self.dropout_rng = jax.random.split(init_rng, num=2)

            self.init_params = {"params": self.model.params, "params_logvar": None, "params_epsilon": self.model.params}
            self.init_batch_stats = None
            self.state = None
        else:
            variables = self.model.init(init_rng, exmp_inputs, train=True)
            variables_logvar = self.model.init(init_rng_logvar, exmp_inputs, train=True)

            init_params = variables['params']

            if self.stochastic:
                init_params_logvar = jax.tree_map(lambda x: x + self.init_logvar, variables_logvar['params'])
                init_params_feature_logvar, init_params_final_layer_logvar = self.split_params(init_params_logvar, "dense")
                init_params_final_layer_logvar = jax.tree_map(lambda x: x * 0 + self.init_logvar, init_params_final_layer_logvar)
                self.final_layer_key = [key for key in init_params_final_layer_logvar.keys()][-1]

                minval_weights = self.init_final_layer_weights_logvar
                maxval_weights = self.init_final_layer_weights_logvar + 0.1
                init_params_final_layer_logvar[self.final_layer_key]["kernel"] = jnp.array(jax.random.uniform(key=init_rng, shape=init_params_final_layer_logvar[self.final_layer_key]["kernel"].shape, minval=minval_weights, maxval=maxval_weights, dtype=float))

                minval_bias = self.init_final_layer_bias_logvar
                maxval_bias = self.init_final_layer_bias_logvar + 0.1
                init_params_final_layer_logvar[self.final_layer_key]["bias"] = jnp.array(jax.random.uniform(key=init_rng, shape=init_params_final_layer_logvar[self.final_layer_key]["bias"].shape, minval=minval_bias, maxval=maxval_bias, dtype=float))

                # init_params_final_layer_logvar = jax.tree_map(lambda x: x * 0 - 15, init_params_final_layer_logvar)
                # init_params_final_layer_logvar = jax.tree_map(lambda x: x * 0 - 50, init_params_final_layer_logvar)
                init_params_logvar = merge_params(init_params_feature_logvar, init_params_final_layer_logvar)
                init_params_epsilon = None

            else:
                init_params_logvar = None
                init_params_epsilon = None

                # init_params_logvar = jax.tree_map(lambda x: x + self.init_logvar, variables_logvar['params'])
                # init_params_feature_logvar, init_params_final_layer_logvar = self.split_params(init_params_logvar, "dense")
                # init_params_final_layer_logvar = jax.tree_map(lambda x: x * 0 + self.init_logvar, init_params_final_layer_logvar)
                # self.final_layer_key = [key for key in init_params_final_layer_logvar.keys()][-1]

                # minval_weights = self.init_final_layer_weights_logvar
                # maxval_weights = self.init_final_layer_weights_logvar + 0.1
                # init_params_final_layer_logvar[self.final_layer_key]["kernel"] = jnp.array(jax.random.uniform(key=init_rng, shape=init_params_final_layer_logvar[self.final_layer_key]["kernel"].shape, minval=minval_weights, maxval=maxval_weights, dtype=float))

                # minval_bias = self.init_final_layer_bias_logvar
                # maxval_bias = self.init_final_layer_bias_logvar + 0.1
                # init_params_final_layer_logvar[self.final_layer_key]["bias"] = jnp.array(jax.random.uniform(key=init_rng, shape=init_params_final_layer_logvar[self.final_layer_key]["bias"].shape, minval=minval_bias, maxval=maxval_bias, dtype=float))

                # init_params_epsilon = merge_params(init_params_feature_logvar, init_params_final_layer_logvar)

                # init_params_epsilon = init_params

            # self.init_params = freeze({"params": init_params, "params_logvar": copy(init_params)})
            self.init_params = freeze({"params": init_params, "params_logvar": init_params_logvar, "params_epsilon": init_params_epsilon})
            self.init_batch_stats = variables['batch_stats']

            self.linearization_params = jax.tree_map(lambda x: x * 1.00001, jax.lax.stop_gradient(self.init_params))
            self.linearization_batch_stats = jax.tree_map(lambda x: x * 1.00001, jax.lax.stop_gradient(self.init_batch_stats))

            self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        if 'bert' in model_name:
            print("Use BERT init optimizer")
            if self.optimizer_name.lower() == 'sgd':
                from flax import traverse_util
                opt_class = optax.sgd

                # learning_rate schedule
                if self.alpha != 1:
                    learning_rate_schedule = optax.cosine_decay_schedule(
                        init_value=self.learning_rate,
                        decay_steps=num_steps_per_epoch * num_epochs,
                        alpha=self.alpha,
                    )
                else:
                    learning_rate_schedule = optax.piecewise_constant_schedule(
                        init_value=self.learning_rate
                    )

                def decay_mask_fn(params):
                    flat_params = traverse_util.flatten_dict(params)
                    flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in
                                 flat_params}
                    return traverse_util.unflatten_dict(flat_mask)

                transf = []
                # transf = [optax.clip(1.0)]
                if (
                        opt_class == optax.sgd or opt_class == optax.adamw) and 'weight_decay' in self.optimizer_hparams:  # wd is integrated in adamw
                    transf.append(optax.add_decayed_weights(self.weight_decay))
                    self.optimizer_hparams.pop('weight_decay')

                optimizer = optax.chain(
                    *transf,
                    opt_class(learning_rate_schedule, **self.optimizer_hparams)
                )

                self.state = TrainState.create(
                    apply_fn=self.model.__call__,
                    params=self.init_params if self.state is None else self.state.params,  # use init no freeze
                    batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
                    tx=optimizer
                )
            elif self.optimizer_name.lower() == 'adamw':
                from flax import traverse_util

                # learning_rate schedule
                if self.alpha != 1:
                    if self.lr_schedule_name == 'linear':
                        learning_rate_schedule = optax.linear_schedule(
                            init_value=self.learning_rate,
                            end_value=self.alpha,
                            transition_steps=num_steps_per_epoch*num_epochs,
                        )
                    else:
                        learning_rate_schedule = optax.cosine_decay_schedule(
                            init_value=self.learning_rate,
                            decay_steps=num_steps_per_epoch*num_epochs,
                            alpha=self.alpha,
                        )
                else:
                    learning_rate_schedule = lambda lr: self.learning_rate
                    # learning_rate_schedule = optax.piecewise_constant_schedule(
                    #     init_value=self.learning_rate
                    # )

                def decay_mask_fn(params):
                    flat_params = traverse_util.flatten_dict(params)
                    flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
                    return traverse_util.unflatten_dict(flat_mask)

                def adamw(weight_decay):
                    return optax.adamw(learning_rate=learning_rate_schedule, b1=0.9, b2=0.999, eps=1e-6,
                                       weight_decay=weight_decay, mask=decay_mask_fn)

                self.state = TrainState.create(
                    apply_fn=self.model.__call__,
                    params=self.init_params if self.state is None else self.state.params, # use init no freeze
                    batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
                    tx=adamw(weight_decay=weight_decay), # 0.01, adjust weight_decay
                )
        elif not self.ssm:
            if self.optimizer_name.lower() == 'adam':
                opt_class = optax.adam
                self.optimizer_hparams.pop('momentum')
                self.optimizer_hparams.pop('weight_decay')
            elif self.optimizer_name.lower() == 'adamw':
                self.optimizer_hparams.pop('momentum')
                opt_class = optax.adamw
            elif self.optimizer_name.lower() == 'sgd':
                opt_class = optax.sgd
            else:
                assert False, f'Unknown optimizer "{self.optimizer_name}"'

            if self.alpha != 1:
                learning_rate_schedule = optax.cosine_decay_schedule(
                    init_value=self.learning_rate,
                    decay_steps=num_steps_per_epoch*num_epochs,
                    alpha=self.alpha,
                )
            else:
                learning_rate_schedule = optax.piecewise_constant_schedule(
                    init_value=self.learning_rate
                )
            transf = []
            # transf = [optax.clip(1.0)]
            if (opt_class == optax.sgd or opt_class == optax.adamw) and 'weight_decay' in self.optimizer_hparams:  # wd is integrated in adamw
                transf.append(optax.add_decayed_weights(self.weight_decay))
                self.optimizer_hparams.pop('weight_decay')
            optimizer = optax.chain(
                *transf,
                opt_class(learning_rate_schedule, **self.optimizer_hparams)
            )

            self.state = TrainState.create(
                apply_fn=self.model.apply if 'bert' not in self.model_name else self.model.__call__,
                params=self.init_params if self.state is None else self.state.params,
                batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
                tx=optimizer
                )

    def train_model(self, train_loader, context_loader, val_loader, rng_key, num_epochs=200):
        print(f"\nTraining model for {num_epochs} epochs:\n")
        self.init_optimizer(num_epochs, len(train_loader))
        best_eval = 0.0

        if self.batch_stats_prior is None and self.batch_stats_init_epochs != 0:
            print(f"Calibrating batch normalization statistics for {self.batch_stats_init_epochs} epochs:\n")
            self.batch_stats_prior = self.state.batch_stats
            self.state_pretrain = self.state
            for _ in tqdm(range(self.batch_stats_init_epochs)):
                self.pretrain(train_loader, context_loader, rng_key=rng_key)
            self.state = self.state.replace(batch_stats=self.state_pretrain.batch_stats)
            self.batch_stats_prior = self.state_pretrain.batch_stats

        if self.fairness_train:
            train_sampler = data.RandomSampler(train_dataset, generator=torch.Generator().manual_seed(seed))
            if "context" in self.reg_type:
                seed_context = seed + 1
            else:
                seed_context = seed
            context_sampler = data.RandomSampler(context_set, generator=torch.Generator().manual_seed(seed_context))
            train_loader = data.DataLoader(train_dataset,
                            batch_size=batch_size,
                            sampler=train_sampler,
                            drop_last=True,
                            #    collate_fn=numpy_collate,
                            num_workers=num_workers_train,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers_train,
                            )
            context_loader  = data.DataLoader(context_set,
                            batch_size=context_batch_size,
                            sampler=context_sampler,
                            drop_last=False,
                            #    collate_fn=numpy_collate,
                            num_workers=num_workers_context,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers_context,
                            )

        for epoch in tqdm(range(num_epochs), leave=False):
            epoch_idx = epoch + 1
            self.train_epoch(train_loader, context_loader, epoch=epoch_idx, rng_key=rng_key)
            if epoch_idx % self.log_frequency == 0 and self.log_frequency_steps == 0:
                if self.dataset != "two-moons" and self.dataset != "snelson" and self.dataset != "oat1d" and "offline_rl" not in self.dataset:
                    self.eval_model(rng_key, self.n_batches_eval)
                    if self.logger['acc_test'][-1] >= best_eval:
                        self.logger['acc_test_best'].append(self.logger['acc_test'][-1])
                        self.save_model(step=epoch_idx, best=True)
                    else:
                        self.logger['acc_test_best'].append(self.logger['acc_test_best'][-1])
                    best_eval = self.logger['acc_test_best'][-1]

                    self.logger['epoch'].append(epoch_idx)
                    # self.logger['step_full'].append(self.step_full)

                    if self.save_to_wandb and epoch_idx < num_epochs:
                        self.wandb_logger.append({})
                        for item in self.logger.items():
                            try:
                                self.wandb_logger[-1][item[0]] = item[1][-1]
                            except:
                                pass
                        wandb.log(self.wandb_logger[-1])

                    if epoch_idx % (10 * self.log_frequency) == 0:
                        self.save_model(step=epoch_idx)

                    if self.fairness_eval:
                        if not validation_training:
                            print(f"\nEpoch {epoch_idx}  |  Train Accuracy: {self.logger['acc_train'][-1]:.2f}  |  Test Accuracy: {self.logger['acc_test'][-1]:.2f}  |  Weighted Test Accuracy: {self.logger['group_weighted_acc_test'][-1]:.2f}  |  Equality of Opportunity: {self.logger['diff_in_eop'][-1]:.2f}  |  Equality of Odds: {self.logger['diff_in_eoo'][-1]:.2f}  |  Worst Group Train Accuracy: {self.logger['worst_group_acc_train'][-1]:.2f}  |  Worst Group Test Accuracy: {self.logger['worst_group_acc_test'][-1]:.2f}  |  NLL: {self.logger['nll_test'][-1]:.3f}  |  Test ECE: {self.logger['ece_test'][-1]:.2f}  |  Uncertainty Test: {self.logger['predictive_entropy_test'][-1]:.3f}  |  Selective Accuracy Test: {self.logger['acc_sel_test'][-1]:.2f}")
                        else:
                            print(f"\nEpoch {epoch_idx}  |  Train Accuracy: {self.logger['acc_train'][-1]:.2f}  |  Val Accuracy: {self.logger['acc_val'][-1]:.2f}  |  Weighted Val Accuracy: {self.logger['group_weighted_acc_val'][-1]:.2f}  |  Test Accuracy: {self.logger['acc_test'][-1]:.2f}  |  Weighted Test Accuracy: {self.logger['group_weighted_acc_test'][-1]:.2f}  |  Equality of Opportunity: {self.logger['diff_in_eop'][-1]:.2f}  |  Equality of Odds: {self.logger['diff_in_eoo'][-1]:.2f}  |  Worst Group Train Accuracy: {self.logger['worst_group_acc_train'][-1]:.2f}  |  Worst Group Val Accuracy: {self.logger['worst_group_acc_val'][-1]:.2f}  |  Worst Group Test Accuracy: {self.logger['worst_group_acc_test'][-1]:.2f}  |  NLL: {self.logger['nll_test'][-1]:.3f}  |  Test ECE: {self.logger['ece_test'][-1]:.2f}  |  Uncertainty Test: {self.logger['predictive_entropy_test'][-1]:.3f}  |  Selective Accuracy Test: {self.logger['acc_sel_test'][-1]:.2f}")
                    else:
                        print(f"\nEpoch {epoch_idx}  |  Train Accuracy: {self.logger['acc_train'][-1]:.2f}  |  Test Accuracy: {self.logger['acc_test'][-1]:.2f}  |  Selective Accuracy Test: {self.logger['acc_sel_test'][-1]:.2f}  |  Selective Accuracy Test+OOD: {self.logger['acc_sel_test_ood'][-1]:.2f}  |  NLL: {self.logger['nll_test'][-1]:.3f}  |  Test ECE: {self.logger['ece_test'][-1]:.2f}  |  OOD AUROC: {self.logger['ood_auroc_entropy'][-1]:.2f} / {self.logger['ood_auroc_aleatoric'][-1]:.2f} / {self.logger['ood_auroc_epistemic'][-1]:.2f}  |  Uncertainty Test: {self.logger['predictive_entropy_test'][-1]:.3f} / {self.logger['aleatoric_uncertainty_test'][-1]:.3f} / {self.logger['epistemic_uncertainty_test'][-1]:.3f}  |  Uncertainty Context: {self.logger['predictive_entropy_context'][-1]:.3f} / {self.logger['aleatoric_uncertainty_context'][-1]:.3f} / {self.logger['epistemic_uncertainty_context'][-1]:.3f}  |  Uncertainty OOD: {self.logger['predictive_entropy_ood'][-1]:.3f} / {self.logger['aleatoric_uncertainty_ood'][-1]:.3f} / {self.logger['epistemic_uncertainty_ood'][-1]:.3f}")
                
                elif self.dataset == "two-moons":  # two-moons                                     
                    _preds_f_test_samples = []
                    for i, batch in enumerate(test_loader):
                        inputs_test = np.array(batch[0], dtype=np.float32)
                        inputs = inputs_test
                        _preds_f_test_samples_list = []
                        for _ in range(self.mc_samples_eval):
                            rng_key, _ = jax.random.split(rng_key)
                            sample = self.pred_fn_jit(self.state.params["params"], self.state.params["params_logvar"], self.state.batch_stats, inputs, rng_key)
                            _preds_f_test_samples_list.append(sample)
                        _preds_f_test_samples.append(jnp.stack(_preds_f_test_samples_list, axis=0))
                    preds_f_test_samples = jnp.concatenate(_preds_f_test_samples, axis=1)[:, :testset_size, :]

                    preds_y_mean = jnp.mean(jax.nn.softmax(preds_f_test_samples, -1), 0)
                    preds_y_mean = preds_y_mean[permutation_inv]
                    prediction_mean = preds_y_mean[:, 0].reshape(xx.shape)

                    preds_y_var = jnp.var(jax.nn.softmax(preds_f_test_samples, -1), 0)
                    preds_y_var = preds_y_var[permutation_inv]
                    prediction_var = preds_y_var[:, 0].reshape(xx.shape)

                    mc_samples_eval_plot = 10

                    preds_f_sample_train_list = []
                    for _ in range(mc_samples_eval_plot):
                        rng_key, _ = jax.random.split(rng_key)
                        sample = self.pred_fn_jit(self.state.params["params"], self.state.params["params_logvar"], self.state.batch_stats, x_train, rng_key)
                        preds_f_sample_train_list.append(sample)
                    preds_f_sample_train = jnp.stack(preds_f_sample_train_list, axis=0)
                    ece_train = 100 * calibration(jax.nn.one_hot(jnp.array(y_train[:, 0], dtype=int), 2), jnp.mean(jax.nn.softmax(preds_f_sample_train, -1), 0))[0]

                    print(f"ECE Train: {ece_train:.2f}")

                    fig, ax = plt.subplots(figsize=(10, 7))

                    cbar = ax.contourf(xx, yy, prediction_mean, levels=20, cmap=cm.coolwarm)
                    cb = fig.colorbar(cbar,)
                    cb.ax.set_ylabel(
                        # "$\mathbb{E}_{\Theta}[y(x_{*}, \Theta) | \mathcal{D}]$",
                        "$\mathbb{E}_{\Theta}[p(y_{*} | x_{*}, \Theta; f) | \mathcal{D}]$",
                        rotation=270,
                        labelpad=40,
                        size=30,
                    )
                    # cb.ax.set_ylabel('$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', labelpad=-90)
                    cb.ax.tick_params(labelsize=30)
                    ax.scatter(
                        x_train[y_train[:, 0] == 1, 0],
                        x_train[y_train[:, 0] == 1, 1],
                        color="cornflowerblue",
                        edgecolors="black",
                        s=90
                    )
                    ax.scatter(
                        x_train[y_train[:, 0] == 0, 0],
                        x_train[y_train[:, 0] == 0, 1],
                        color="tomato",
                        edgecolors="black",
                        s=90
                    )
                    ax.tick_params(labelsize=30)
                    plt.xticks([])
                    plt.yticks([])

                    fig1 = copy(fig)

                    fig.savefig(
                        f"figures/two_moons/two_moons_predictive_mean_{epoch_idx}.pdf",
                        bbox_inches="tight",
                    )
                    plt.close()

                    fig, ax = plt.subplots(figsize=(10, 7))

                    cbar = ax.contourf(xx, yy, prediction_var, levels=20, cmap=cm.coolwarm)
                    cb = fig.colorbar(cbar,)
                    cb.ax.set_ylabel(
                        "$\mathbb{V}_{\Theta}[y(x_{*}, \Theta) | \mathcal{D}]$",
                        rotation=270,
                        labelpad=40,
                        size=30,
                    )
                    # cb.ax.set_ylabel('$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', labelpad=-90)
                    cb.ax.tick_params(labelsize=30)
                    ax.scatter(
                        x_train[y_train[:, 0] == 0, 0],
                        x_train[y_train[:, 0] == 0, 1],
                        color="cornflowerblue",
                        edgecolors="black",
                        s=90
                    )
                    ax.scatter(
                        x_train[y_train[:, 0] == 1, 0],
                        x_train[y_train[:, 0] == 1, 1],
                        color="tomato",
                        edgecolors="black",
                        s=90
                    )
                    plt.tick_params(labelsize=30)

                    fig2 = copy(fig)

                    fig.savefig(
                        f"figures/two_moons/two_moons_predictive_var_{epoch_idx}.pdf",
                        bbox_inches="tight",
                    )
                    plt.close()

                    if self.save_to_wandb:
                        wandb.log({
                            "epoch": epoch_idx,
                            "ece_train": ece_train,
                            "two_moons_predictive_mean": wandb.Image(fig1),
                            "two_moons_predictive_var": wandb.Image(fig2),
                            })

                elif self.dataset == "snelson" or self.dataset == "oat1d":

                    params = self.state.params["params"]
                    params_logvar = self.state.params["params_logvar"]

                    # params = jax.tree_map(lambda x: x, jax.lax.stop_gradient(self.model.init(jax.random.PRNGKey(self.seed), x_train[0:1], train=True)["params"]))
                    # params_logvar = jax.tree_map(lambda x: x - 3.5, params)
                    # params_logvar = jax.tree_map(lambda x: x * 0 + jnp.log(0.1), self.state.params["params_logvar"])

                    _preds_f_test_samples = []
                    for i, batch in enumerate(test_loader):
                        inputs_test = np.array(batch[0], dtype=np.float32)
                        inputs = inputs_test
                        _preds_f_test_samples_list = []
                        for _ in range(self.mc_samples_eval):
                            rng_key, _ = jax.random.split(rng_key)
                            sample = self.pred_fn_jit(params, params_logvar, self.state.batch_stats, inputs, rng_key)
                            _preds_f_test_samples_list.append(sample)
                        _preds_f_test_samples.append(jnp.stack(_preds_f_test_samples_list, axis=0))

                    preds_f_test_samples = jnp.concatenate(_preds_f_test_samples, axis=1)[:, :testset_size, 0]

                    preds_f_test_mean = jnp.mean(preds_f_test_samples, 0)
                    preds_f_test_mean = preds_f_test_mean

                    preds_f_test_var = jnp.var(preds_f_test_samples, 0)
                    preds_f_test_var = preds_f_test_var

                    ### for plotting distribution over functions with fixed feature
                    # feature = True
                    # params_logvar_small = jax.tree_map(lambda x: x - 50, params_logvar)
                    # pred, feature = self.pred_fn_jit(params, params_logvar_small, self.state.batch_stats, inputs, rng_key, feature)

                    # preds_f_prior_cov = self.prior_likelihood_scale * jnp.matmul(feature, feature.T)  # assumes the prior is identical across output dimensions
                    # preds_f_prior_cov += jnp.ones_like(preds_f_prior_cov) * self.prior_likelihood_scale  # add bias variance
                    # preds_f_prior_cov += jnp.eye(preds_f_prior_cov.shape[0]) * self.prior_likelihood_scale  # jnp.max(jnp.array([eps_cov * prior_var, eps_cov]))  # add small constant to the diagonal to ensure positive definiteness

                    # p = tfd.MultivariateNormalFullCovariance(
                    #     # loc=jnp.zeros_like(pred[:,0]),  # assumes the prior is identical across output dimensions
                    #     loc=preds_f_test_mean,  # assumes the prior is identical across output dimensions
                    #     covariance_matrix=preds_f_prior_cov,
                    #     # covariance_matrix=jnp.eye(preds_f_prior_cov.shape[0]) * 100,
                    #     validate_args=False,
                    #     allow_nan_stats=True,
                    # )

                    # preds_f_test_samples = p.sample(seed=rng_key, sample_shape=(100,))
                    # # preds_f_test_mean = jnp.zeros_like(preds_f_test_mean)
                    # preds_f_test_var = jnp.diag(preds_f_prior_cov)

                    likelihood_var = self.likelihood_scale

                    xlim = (-x_test_lim, x_test_lim)
                    ylim = (-2.5, 2.5)

                    mc_samples_eval_plot = 10

                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.plot(x_test, preds_f_test_mean, color="k", label="Predictive Mean", zorder=5)

                    for i in range(mc_samples_eval_plot):
                        if i == 0:
                            ax.plot(
                                x_test,
                                preds_f_test_samples[i : i + 1, :].T,
                                linewidth=0.5,
                                color="xkcd:blue",
                                label="Function Draw",
                                zorder=3,
                                alpha=0.3,
                            )
                        else:
                            ax.plot(
                                x_test,
                                preds_f_test_samples[i : i + 1, :].T,
                                linewidth=0.5,
                                color="xkcd:blue",
                                zorder=3,
                                alpha=0.3,
                            )
                    ax.fill_between(
                        np.squeeze(x_test),
                        np.squeeze(preds_f_test_mean) - np.sqrt(likelihood_var + preds_f_test_var),
                        np.squeeze(preds_f_test_mean) + np.sqrt(likelihood_var + preds_f_test_var),
                        color="C0",
                        alpha=0.2,
                        )
                    ax.fill_between(
                        np.squeeze(x_test),
                        np.squeeze(preds_f_test_mean) - 2 * np.sqrt(likelihood_var + preds_f_test_var),
                        np.squeeze(preds_f_test_mean) + 2 * np.sqrt(likelihood_var + preds_f_test_var),
                        color="C0",
                        alpha=0.2,
                        )
                    ax.scatter(x_train, y_train, s=10, color="r", label="Training Data", zorder=2)
                    if xlim is not None:
                        ax.set_xlim(xlim)
                    if ylim is not None:
                        ax.set_ylim(ylim)

                    ax.legend(fontsize="20")
                    ax.grid(True)

                    if self.save_to_wandb:
                        wandb.log({
                            "epoch": epoch_idx,
                            f"{self.other_hparams['dataset']}_predictive_distribution": wandb.Image(fig),
                            })

                    fig.savefig(
                        f"figures/{self.other_hparams['dataset']}/{self.other_hparams['dataset']}_predictive_distribution_{epoch_idx}.pdf",
                        bbox_inches="tight",
                    )
                    plt.close()

                elif "offline_rl" in self.dataset:

                    params = self.state.params["params"]
                    params_logvar = self.state.params["params_logvar"]

                    _preds_f_test_samples = []
                    for i, batch in enumerate(test_loader):
                        inputs_test = np.array(batch[0], dtype=np.float32)
                        inputs = inputs_test
                        _preds_f_test_samples_list = []
                        for _ in range(self.mc_samples_eval):
                            rng_key, _ = jax.random.split(rng_key)
                            sample = self.pred_fn_jit(params, params_logvar, self.state.batch_stats, inputs, rng_key)
                            _preds_f_test_samples_list.append(sample)
                        _preds_f_test_samples.append(jnp.stack(_preds_f_test_samples_list, axis=0))

                    preds_f_test_samples = jnp.concatenate(_preds_f_test_samples, axis=1)[:, :testset_size, :]

                    preds_f_test_mean = jnp.mean(preds_f_test_samples, 0)
                    preds_f_test_mean = preds_f_test_mean

                    preds_f_test_var = jnp.var(preds_f_test_samples, 0)
                    preds_f_test_var = preds_f_test_var

                    if self.output_var:
                        preds_f_test_mean = preds_f_test_mean[:, :self.num_classes]

                    self.mse_test = jnp.mean(jnp.mean(jnp.square(preds_f_test_mean - y_test), axis=-1), axis=-1)

                    print(f"MSE Train: {self.logger['acc_train'][-1]:.2f}  |  MSE Test: {self.mse_test:.2f}")
                    
                else:
                    raise ValueError(f"Dataset {self.dataset} not recognized.")

    def pretrain(self, train_loader, context_loader, rng_key):
        data_loader = tqdm(enumerate(zip(train_loader, context_loader)), leave=False)
        for i, (batch, batch_context) in data_loader:
            inputs = np.array(batch[0], dtype=np.float32)
            targets = np.array(batch[1], dtype=np.float32)
            attributes = np.array(batch[2], dtype=np.float32) if len(batch) > 2 else None
            group = np.array(batch[3], dtype=np.float32) if len(batch) > 3 else None
            _inputs_context = np.array(batch_context[0], dtype=np.float32)
            _targets_context = np.array(batch_context[1], dtype=np.float32)
            _attributes_context = np.array(batch_context[2], dtype=np.float32) if len(batch_context) > 2 else None
            _group_context = np.array(batch_context[3], dtype=np.float32) if len(batch_context) > 3 else None

            batch = [inputs, targets]
            batch_context = [_inputs_context, _targets_context]

            self.state_pretrain, loss, acc = self.train_step(self.state_pretrain, batch, batch_context, rng_key)
            rng_key, _ = jax.random.split(rng_key)
            _, self.dropout_rng = jax.random.split(self.dropout_rng) # get new rng
            self.batch_stats_prior = self.state_pretrain.batch_stats

    def train_epoch(self, train_loader, context_loader, epoch, rng_key):
        metrics = defaultdict(list)
        data_loader = tqdm(enumerate(zip(train_loader, context_loader)), leave=False)
        train_acc = 0
        elapsed = 0
        for i, (batch, batch_context) in data_loader:
            inputs = np.array(batch[0], dtype=np.float32)
            targets = np.array(batch[1], dtype=np.float32)
            attributes = np.array(batch[2], dtype=np.float32) if len(batch) > 2 else None
            group = np.array(batch[3], dtype=np.float32) if len(batch) > 3 else None
            _inputs_context = np.array(batch_context[0], dtype=np.float32)
            _targets_context = np.array(batch_context[1], dtype=np.float32)
            _attributes_context = np.array(batch_context[2], dtype=np.float32) if len(batch_context) > 2 else None
            _group_context = np.array(batch_context[3], dtype=np.float32) if len(batch_context) > 3 else None
            batch = [inputs, targets]
            batch_context = [_inputs_context, _targets_context]

            if self.debug_print:
                if i % 1000 == 0:
                    debug_print = True
                    print(f"\nEpoch {epoch} Batch {i}   \n")
                else:
                    debug_print = False
            else:
                debug_print = False
            if self.group_dro:
                self.state, loss, acc, self.gdro_weights = self.train_step(self.state, batch, batch_context, rng_key, debug_print, group_attr=group, group_attr_context=_group_context, gdro_weights=self.gdro_weights) 
            else:
                self.state, loss, acc = self.train_step(self.state, batch, batch_context, rng_key, debug_print) 
            rng_key, _ = jax.random.split(rng_key)
            metrics['loss'].append(loss)
            metrics['acc'].append(acc)
            if (data_loader.format_dict["elapsed"] - elapsed) >= 0.5:  # update every 5 seconds
                train_acc = np.stack(jax.device_get(metrics["acc"]))[-40:].mean()  # average accuracy of last 40 batches
                train_loss = np.stack(jax.device_get(metrics["loss"]))[-40:].mean()  # average accuracy of last 40 batches
                data_loader.set_postfix({'accuracy': train_acc, 'loss': train_loss})
                elapsed = data_loader.format_dict["elapsed"]

            step = i + 1
            self.step_full = i + (epoch - 1) * min(len(train_loader), len(context_loader))

            if self.log_frequency_steps != 0 and (step % self.log_frequency_steps == 0):
                self.eval_model(rng_key, self.n_batches_eval)

                self.logger['acc_test_best'].append(0)
                self.save_model(step=self.step_full, best=True)

                self.logger['epoch'].append(self.step_full)
                # self.logger['step_full'].append(self.step_full)

                if self.save_to_wandb:
                    self.wandb_logger.append({})
                    for item in self.logger.items():
                        try:
                            self.wandb_logger[-1][item[0]] = item[1][-1]
                        except:
                            pass
                    wandb.log(self.wandb_logger[-1])

                if self.fairness_eval:
                    if not validation_training:
                        print(f"\nEpoch {epoch}  |  Step {self.step_full}  |  Train Accuracy: {train_acc:.2f}  |  Test Accuracy: {self.logger['acc_test'][-1]:.2f}  |  Weighted Test Accuracy: {self.logger['group_weighted_acc_test'][-1]:.2f}  |  Equality of Opportunity: {self.logger['diff_in_eop'][-1]:.2f}  |  Equality of Odds: {self.logger['diff_in_eoo'][-1]:.2f}  |  Worst Group Train Accuracy: {self.logger['worst_group_acc_train'][-1]:.2f}  |  Worst Group Test Accuracy: {self.logger['worst_group_acc_test'][-1]:.2f}  |  NLL: {self.logger['nll_test'][-1]:.3f}  |  Test ECE: {self.logger['ece_test'][-1]:.2f}  |  Uncertainty Test: {self.logger['predictive_entropy_test'][-1]:.3f}  |  Selective Accuracy Test: {self.logger['acc_sel_test'][-1]:.2f}")
                    else:
                        print(f"\nEpoch {epoch}  |  Step {self.step_full}   |  Train Accuracy: {train_acc:.2f}  |  Val Accuracy: {self.logger['acc_val'][-1]:.2f}  |  Weighted Val Accuracy: {self.logger['group_weighted_acc_val'][-1]:.2f}  |  Test Accuracy: {self.logger['acc_test'][-1]:.2f}  |  Weighted Test Accuracy: {self.logger['group_weighted_acc_test'][-1]:.2f}  |  Equality of Opportunity: {self.logger['diff_in_eop'][-1]:.2f}  |  Equality of Odds: {self.logger['diff_in_eoo'][-1]:.2f}  |  Worst Group Train Accuracy: {self.logger['worst_group_acc_train'][-1]:.2f}  |  Worst Group Val Accuracy: {self.logger['worst_group_acc_val'][-1]:.2f}  |  Worst Group Test Accuracy: {self.logger['worst_group_acc_test'][-1]:.2f}  |  NLL: {self.logger['nll_test'][-1]:.3f}  |  Test ECE: {self.logger['ece_test'][-1]:.2f}  |  Uncertainty Test: {self.logger['predictive_entropy_test'][-1]:.3f}  |  Selective Accuracy Test: {self.logger['acc_sel_test'][-1]:.2f}")
                else:
                    print(f"\nEpoch {epoch}  |  Step {self.step_full}   |  Train Accuracy: {train_acc:.2f}  |  Test Accuracy: {self.logger['acc_test'][-1]:.2f}  |  Selective Accuracy Test: {self.logger['acc_sel_test'][-1]:.2f}  |  Selective Accuracy Test+OOD: {self.logger['acc_sel_test_ood'][-1]:.2f}  |  NLL: {self.logger['nll_test'][-1]:.3f}  |  Test ECE: {self.logger['ece_test'][-1]:.2f}  |  OOD AUROC: {self.logger['ood_auroc_entropy'][-1]:.2f} / {self.logger['ood_auroc_aleatoric'][-1]:.2f} / {self.logger['ood_auroc_epistemic'][-1]:.2f}  |  Uncertainty Test: {self.logger['predictive_entropy_test'][-1]:.3f} / {self.logger['aleatoric_uncertainty_test'][-1]:.3f} / {self.logger['epistemic_uncertainty_test'][-1]:.3f}  |  Uncertainty Context: {self.logger['predictive_entropy_context'][-1]:.3f} / {self.logger['aleatoric_uncertainty_context'][-1]:.3f} / {self.logger['epistemic_uncertainty_context'][-1]:.3f}  |  Uncertainty OOD: {self.logger['predictive_entropy_ood'][-1]:.3f} / {self.logger['aleatoric_uncertainty_ood'][-1]:.3f} / {self.logger['epistemic_uncertainty_ood'][-1]:.3f}")

        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            self.logger[f"{key}_train"].append(avg_val)

    def eval_model(self, rng_key, n_batches_eval, full_eval=False):
        self.eval_step(self.state, rng_key, n_batches_eval, full_eval)

    def save_model(self, step=0, best=False):
        if best:
            checkpoints.save_checkpoint(
                ckpt_dir=f"{self.log_dir}_{best}",
                target={
                    'params': self.state.params["params"],
                    'params_logvar': self.state.params["params_logvar"],
                    'batch_stats': self.state.batch_stats,
                    'batch_stats_prior': self.batch_stats_prior
                },
                step=0,
                overwrite=True,
                prefix=f"checkpoint_best_"
                )
            if self.save_to_wandb:
                wandb.save(f'{self.log_dir}/checkpoint_best_0')
        else:
            checkpoints.save_checkpoint(
                ckpt_dir=f"{self.log_dir}_{step}",
                target={
                    'params': self.state.params["params"],
                    'params_logvar': self.state.params["params_logvar"],
                    'batch_stats': self.state.batch_stats,
                    'batch_stats_prior': self.batch_stats_prior
                },
                step=step,
                overwrite=True,
                prefix=f"checkpoint_"
                )
            if self.save_to_wandb:
                wandb.save(f'{self.log_dir}/checkpoint_{step}')

    def load_model(self, stochastic=False, pretrained_prior=False, restore_checkpoint=False):
        if not stochastic:
            if "Pretrained" in self.model_name and not restore_checkpoint:
                state_dict = self.model.init(rng_key, jnp.ones((1, 224, 224, 3)))
            else:
                if self.checkpoint_dir != '':
                    ckpt_path = self.checkpoint_dir
                    print(f"Loaded checkpoint: {self.checkpoint_dir}")

                state_dict = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None)
                    
                state_dict = freeze(state_dict)
            params_logvar = None
            params_epsilon = None
        else:
            if "Pretrained" in self.model_name and not restore_checkpoint:
                state_dict = self.model.init(rng_key, jnp.ones((1, 224, 224, 3)))

                params_logvar = jax.tree_map(lambda x: x + self.init_logvar, state_dict['params'])
                params_feature_logvar, params_final_layer_logvar = self.split_params(params_logvar, "dense")
                params_final_layer_logvar[self.final_layer_key]["bias"] = params_final_layer_logvar[self.final_layer_key]["bias"] * 0 + self.init_final_layer_bias_logvar
                params_logvar = merge_params(params_feature_logvar, params_final_layer_logvar)
                params_epsilon = None
            else:
                if self.checkpoint_dir != '':
                    ckpt_path = self.checkpoint_dir
                    print(f"Loaded checkpoint: {self.checkpoint_dir}")

                state_dict = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None)
                    
                state_dict = freeze(state_dict)

                if state_dict["params_logvar"] is None:
                    params_logvar = jax.tree_map(lambda x: x + self.init_logvar, state_dict['params'])
                    params_feature_logvar, params_final_layer_logvar = self.split_params(params_logvar, "dense")
                    params_final_layer_logvar[self.final_layer_key]["bias"] = params_final_layer_logvar[self.final_layer_key]["bias"] * 0 + self.init_final_layer_bias_logvar
                    params_final_layer_logvar[self.final_layer_key]["kernel"] = params_final_layer_logvar[self.final_layer_key]["kernel"] * 0 + self.init_final_layer_weights_logvar
                    params_logvar = merge_params(params_feature_logvar, params_final_layer_logvar)
                else:
                    params_logvar = state_dict['params_logvar']
                params_epsilon = None

        if pretrained_prior:
            if 'bert' in self.model_name:
                self.params_prior_mean = unfreeze(state_dict['params'])
            else:
                self.params_prior_mean = state_dict['params']
            self.batch_stats_prior = state_dict['batch_stats']
            self.params_prior_logvar = jax.tree_map(lambda x: x * 0 + jnp.log(self.prior_var), self.params_prior_mean)

        if self.final_layer_random_init:
            state_dict = unfreeze(state_dict)
            if 'bert' in self.model_name:
                state_dict["params"]["classifier"] = self.model.params["classifier"]
            else:
                state_dict_ = self.model.init(rng_key, jnp.ones((1, 224, 224, 3)))
                state_dict["params"]["Dense_0"] = state_dict_["params"]["Dense_0"]
                state_dict = freeze(state_dict)

        # self.batch_stats_prior = state_dict['batch_stats']  

        # if self.linearize:
        #     self.linearization_params = state_dict
        #     self.linearization_batch_stats = state_dict['batch_stats']

        if self.linearize and self.pred_fn is None:
            NotImplementedError
        else:
            self.pred_fn = self.model.apply if 'bert' not in self.model_name else self.model.__call__

        if 'bert' in self.model_name:
            params = {"params": state_dict['params'], "params_logvar": params_logvar, "params_epsilon": params_epsilon}
        else:
            params = freeze({"params": state_dict['params'], "params_logvar": params_logvar, "params_epsilon": params_epsilon})

        self.state = TrainState.create(apply_fn=self.model.apply if 'bert' not in self.model_name else self.model.__call__,
                                    params=params,
                                    # params_logvar=params_logvar,
                                    batch_stats=state_dict['batch_stats'],
                                    tx=self.state.tx if self.state else optax.sgd(0.1)   # Default optimizer
                                    )

    def checkpoint_exists(self):
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'))


def trainer(*args, rng_key, **kwargs):
    kwargs_copy = deepcopy(kwargs)
    del kwargs_copy['exmp_inputs']

    trainer = TrainerModule(*args, **kwargs)

    pprint(kwargs_copy)
    if trainer.save_to_wandb:
        wandb.config = kwargs_copy
        wandb.init(
            project=trainer.wandb_project,
            name=trainer.run_name,
            entity=trainer.wandb_account,
            config=wandb.config,
        )

    train = not trainer.evaluate
    
    if "Pretrained" in kwargs['model_name']:
        prior = True
    else:
        prior = False

    if train and not trainer.restore_checkpoint and not "Pretrained" in kwargs['model_name']:  # train from scratch
        trainer.train_model(train_loader, context_loader, val_loader, rng_key, num_epochs=trainer.num_epochs)
    elif train and ("Pretrained" in kwargs['model_name'] or trainer.restore_checkpoint):  # load trained model and continue training
        trainer.load_model(stochastic=trainer.stochastic, pretrained_prior=trainer.pretrained_prior, restore_checkpoint=trainer.restore_checkpoint)
        trainer.train_model(train_loader, context_loader, val_loader, rng_key, num_epochs=trainer.num_epochs)
    else:  # load trained model and evaluate
        trainer.load_model(stochastic=trainer.stochastic, pretrained_prior=trainer.pretrained_prior, restore_checkpoint=trainer.restore_checkpoint)
        trainer.logger['acc_train'].append(0)
        trainer.logger['acc_test_best'].append(0)
        trainer.logger['loss_train'].append(0)
        trainer.logger['epoch'].append(trainer.num_epochs)

    if trainer.dataset != "two-moons" and trainer.dataset != "snelson" and trainer.dataset != "oat1d" and "offline_rl" not in trainer.dataset:
        # val_acc = trainer.eval_model(val_loader, rng_key)
        trainer.eval_model(rng_key, trainer.n_batches_eval_final, full_eval=trainer.full_eval)
        # print(f"\nValidation Accuracy: {val_acc*100:.2f}")
        if fairness_eval:
            if not validation_training:
                print(f"Train Accuracy: {trainer.logger['acc_train'][-1]:.2f}  |  Test Accuracy: {trainer.logger['acc_test'][-1]:.2f}  |  Weighted Test Accuracy: {trainer.logger['group_weighted_acc_test'][-1]:.2f}  |  Equality of Opportunity: {trainer.logger['diff_in_eop'][-1]:.2f}  |  Equality of Odds: {trainer.logger['diff_in_eoo'][-1]:.2f}  |  Worst Group Train Accuracy: {trainer.logger['worst_group_acc_train'][-1]:.2f}  |  Worst Group Test Accuracy: {trainer.logger['worst_group_acc_test'][-1]:.2f}  |  NLL: {trainer.logger['nll_test'][-1]:.3f}  |  Test ECE: {trainer.logger['ece_test'][-1]:.2f}  |  Uncertainty Test: {trainer.logger['predictive_entropy_test'][-1]:.3f}  |  Selective Accuracy Test: {trainer.logger['acc_sel_test'][-1]:.2f}")
            else:
                print(f"Train Accuracy: {trainer.logger['acc_train'][-1]:.2f}  |  Val Accuracy: {trainer.logger['acc_val'][-1]:.2f}  |  Weighted Val Accuracy: {trainer.logger['group_weighted_acc_val'][-1]:.2f}  |  Test Accuracy: {trainer.logger['acc_test'][-1]:.2f}  |  Weighted Test Accuracy: {trainer.logger['group_weighted_acc_test'][-1]:.2f}  |  Equality of Opportunity: {trainer.logger['diff_in_eop'][-1]:.2f}  |  Equality of Odds: {trainer.logger['diff_in_eoo'][-1]:.2f}  |  Worst Group Train Accuracy: {trainer.logger['worst_group_acc_train'][-1]:.2f}  |  Worst Group Val Accuracy: {trainer.logger['worst_group_acc_val'][-1]:.2f}  |  Worst Group Test Accuracy: {trainer.logger['worst_group_acc_test'][-1]:.2f}  |  NLL: {trainer.logger['nll_test'][-1]:.3f}  |  Test ECE: {trainer.logger['ece_test'][-1]:.2f}  |  Uncertainty Test: {trainer.logger['predictive_entropy_test'][-1]:.3f}  |  Selective Accuracy Test: {trainer.logger['acc_sel_test'][-1]:.2f}")
        else: 
            print(f"Train Accuracy: {trainer.logger['acc_train'][-1]:.2f}  |  Test Accuracy: {trainer.logger['acc_test'][-1]:.2f}  |  Selective Accuracy: {trainer.logger['acc_sel_test'][-1]:.2f}  |  Selective Accuracy Test+OOD: {trainer.logger['acc_sel_test_ood'][-1]:.2f}  |  NLL: {trainer.logger['nll_test'][-1]:.3f}  |  Test ECE: {trainer.logger['ece_test'][-1]:.2f}  |  OOD AUROC: {trainer.logger['ood_auroc_entropy'][-1]:.2f} / {trainer.logger['ood_auroc_aleatoric'][-1]:.2f} / {trainer.logger['ood_auroc_epistemic'][-1]:.2f}  |  Uncertainty Test: {trainer.logger['predictive_entropy_test'][-1]:.3f} / {trainer.logger['aleatoric_uncertainty_test'][-1]:.3f} / {trainer.logger['epistemic_uncertainty_test'][-1]:.3f}  |  Uncertainty Context: {trainer.logger['predictive_entropy_context'][-1]:.3f} / {trainer.logger['aleatoric_uncertainty_context'][-1]:.3f} / {trainer.logger['epistemic_uncertainty_context'][-1]:.3f}  |  Uncertainty OOD: {trainer.logger['predictive_entropy_ood'][-1]:.3f} / {trainer.logger['aleatoric_uncertainty_ood'][-1]:.3f} / {trainer.logger['epistemic_uncertainty_ood'][-1]:.3f}")
        
        trainer.wandb_logger.append({})
        for item in trainer.logger.items():
            trainer.wandb_logger[-1][item[0]] = item[1][-1]

        if trainer.save_to_wandb:
            wandb.log(trainer.wandb_logger[-1])
            time.sleep(10)
            
            pprint(trainer.wandb_logger[-1])

    if "offline_rl" in trainer.dataset:
        Path(trainer.log_dir,).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(trainer.log_dir, "params_pickle_"+str(trainer.num_epochs)), "wb") as file:
            state_normalize = {}
            state_normalize["normalize"] = normalize
            state_normalize["x_mean"] = x_mean
            state_normalize["x_std"] = x_std

            kwargs.pop("model_hparams")
            to_save = {
                "params": trainer.state.params,
                "kwargs": kwargs,
                "epoch": trainer.num_epochs,
                "mse_test": trainer.mse_test,
                "state_normalize": state_normalize,
            }

            pickle.dump(to_save, file)
    
    return trainer, trainer.logger


# conv_kernel_init = lecun_normal()  # flax default
conv_kernel_init = nn.initializers.variance_scaling(1/20, mode='fan_in', distribution='uniform')
# conv_kernel_init = nn.initializers.variance_scaling(1/20, mode='fan_in', distribution='normal')
# conv_kernel_init = nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal')

class BigCNN(nn.Module): # BigCNN doesn't work :(
    """A bigger CNN model."""
    num_classes : int 
    act_fn : callable
    block_class : None
    num_blocks : None
    c_hidden : None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True, feature=False):
        _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)
        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=conv_kernel_init, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=conv_kernel_init, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=conv_kernel_init, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(features=num_classes, dtype=self.dtype)(x)


class CNN(nn.Module):
    """A simple CNN model."""
    num_classes : int
    act_fn : callable
    block_class : None
    num_blocks : None
    c_hidden : None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True, feature=False):
        # x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=conv_kernel_init, dtype=self.dtype)(x)
        # x = nn.relu(x)
        # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        # x = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=conv_kernel_init, dtype=self.dtype)(x)
        # x = nn.relu(x)
        # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        # x = x.reshape((x.shape[0], -1))  # flatten
        # x = nn.Dense(features=256, dtype=self.dtype)(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=num_classes, dtype=self.dtype)(x)
        # _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)

        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=conv_kernel_init, dtype=self.dtype)(x)
        _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=conv_kernel_init, dtype=self.dtype)(x)
        _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256, dtype=self.dtype)(x)
        x = nn.relu(x)

        feature_map = x

        x = nn.Dense(features=num_classes, dtype=self.dtype)(x)
        
        if feature:
            return (x, feature_map)
        else:
            return x

class MLP(nn.Module):
    """Even simpler MLP model."""
    num_classes : int
    act_fn : callable  # unused
    block_class : None  # unused
    num_blocks : None  # unused
    c_hidden : None  # unused
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True, feature=False):
        _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)
        x = nn.Dense(features=100, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(features=100, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(features=num_classes, dtype=self.dtype)(x) 
        x = nn.sigmoid(x) if self.num_classes == 1 else x
        return x

class MLP_Toy(nn.Module):
    """A simple MLP model."""
    num_classes : int
    act_fn : callable  # unused
    block_class : None  # unused
    num_blocks : None  # unused
    c_hidden : None  # unused
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True, feature=False):
        _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)
        x = nn.Dense(features=200, dtype=self.dtype)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=200, dtype=self.dtype)(x)
        # x = nn.tanh(x)
        # x = nn.Dense(features=200, dtype=self.dtype)(x)
        x = nn.tanh(x)

        feature_map = x

        x = nn.Dense(features=num_classes, dtype=self.dtype)(x)
        
        if feature:
            return (x, feature_map)
        else:
            return x


class MLP_OfflineRL(nn.Module):
    """A simple MLP model."""
    num_classes : int
    act_fn : callable  # unused
    block_class : None  # unused
    num_blocks : None  # unused
    c_hidden : None  # unused
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True, feature=False):
        _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)
        x = nn.Dense(features=200, dtype=self.dtype)(x)
        x = nn.swish(x)
        x = nn.Dense(features=200, dtype=self.dtype)(x)
        x = nn.swish(x)
        x = nn.Dense(features=200, dtype=self.dtype)(x)
        x = nn.swish(x)
        x = nn.Dense(features=200, dtype=self.dtype)(x)
        x = nn.swish(x)

        feature_map = x

        mean = nn.Dense(features=num_classes, dtype=self.dtype)(x)
        var = jax.nn.softplus(nn.Dense(features=num_classes, dtype=self.dtype)(x))

        x = jnp.concatenate([mean, var], axis=-1)  # concatenate across output dimensions
        
        if feature:
            return (x, feature_map)
        else:
            return x


class ResNetBlock(nn.Module):
    act_fn : callable  # Activation function
    c_out : int   # Output feature size
    subsample : bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            strides=(1, 1) if not self.subsample else (2, 2),
            kernel_init=conv_kernel_init,
            use_bias=False
            )(x)
        z = nn.BatchNorm()(z, use_running_average=not train)

        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            kernel_init=conv_kernel_init,
            use_bias=False
            )(z)
        z = nn.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(
                self.c_out,
                kernel_size=(1, 1),
                strides=(2, 2),
                kernel_init=conv_kernel_init
            )(x)

        x_out = self.act_fn(z + x)
        return x_out


class PreActResNetBlock(ResNetBlock):

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=conv_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=conv_kernel_init,
                    use_bias=False)(z)

        if self.subsample:
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)
            x = nn.Conv(self.c_out,
                        kernel_size=(1, 1),
                        strides=(2, 2),
                        kernel_init=conv_kernel_init,
                        use_bias=False)(x)

        x_out = z + x
        return x_out


class ResNetMod(nn.Module):
    num_classes : int
    act_fn : callable
    block_class : nn.Module
    num_blocks : tuple = (3, 3, 3)
    c_hidden : tuple = (16, 32, 64)

    @nn.compact
    def __call__(self, x, train=True, feature=False):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(
            self.c_hidden[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            kernel_init=conv_kernel_init,
            use_bias=False
        )(x)
        # x = nn.Conv(self.c_hidden[0], kernel_size=(3, 3), kernel_init=conv_kernel_init, use_bias=False)(x)  # flax default
        if self.block_class == ResNetBlock:
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = self.block_class(
                    c_out=self.c_hidden[block_idx],
                    act_fn=self.act_fn,
                    subsample=subsample
                    )(x, train=train)

        # Mapping to classification output
        feature_map = x.mean(axis=(1, 2))

        x = nn.Dense(self.num_classes)(feature_map)

        if feature:
            return (x, feature_map)

        return x


URLS = {'resnet18': 'https://www.dropbox.com/s/wx3vt76s5gpdcw5/resnet18_weights.h5?dl=1',
        'resnet34': 'https://www.dropbox.com/s/rnqn2x6trnztg4c/resnet34_weights.h5?dl=1',
        'resnet50': 'https://www.dropbox.com/s/fcc8iii38ezvqog/resnet50_weights.h5?dl=1',
        'resnet101': 'https://www.dropbox.com/s/hgtnk586pnz0xug/resnet101_weights.h5?dl=1',
        'resnet152': 'https://www.dropbox.com/s/tvi28uwiy54mcfr/resnet152_weights.h5?dl=1'}

LAYERS = {'resnet18': [2, 2, 2, 2],
          'resnet34': [3, 4, 6, 3],
          'resnet50': [3, 4, 6, 3],
          'resnet101': [3, 4, 23, 3],
          'resnet152': [3, 8, 36, 3]}


class BasicBlock(nn.Module):
    """
    Basic Block.

    Attributes:
        features (int): Number of output channels.
        kernel_size (Tuple): Kernel size.
        downsample (bool): If True, downsample spatial resolution.
        stride (bool): If True, use strides (2, 2). Not used in this module.
                       The attribute is only here for compatibility with Bottleneck.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        kernel_init (functools.partial): Kernel initializer.
        bias_init (functools.partial): Bias initializer.
        block_name (str): Name of block.
        dtype (str): Data type.
    """
    features: int
    kernel_size: Union[int, Iterable[int]]=(3, 3)
    downsample: bool=False
    stride: bool=True
    param_dict: h5py.Group=None
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    block_name: str=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, act, train=True):
        """
        Run Basic Block.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            act (dict): Dictionary containing activations.
            train (bool): Training mode.

        Returns:
            (tensor): Output shape of shape [N, H', W', features].
        """
        residual = x 
        
        x = nn.Conv(features=self.features, 
                    kernel_size=self.kernel_size, 
                    strides=(2, 2) if self.downsample else (1, 1),
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv1']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn1'],
                           dtype=self.dtype) 
        x = nn.relu(x)

        x = nn.Conv(features=self.features, 
                    kernel_size=self.kernel_size, 
                    strides=(1, 1), 
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv2']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn2'],
                           dtype=self.dtype) 

        if self.downsample:
            residual = nn.Conv(features=self.features, 
                               kernel_size=(1, 1), 
                               strides=(2, 2), 
                               kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['downsample']['conv']['weight']), 
                               use_bias=False,
                               dtype=self.dtype)(residual)

            residual = ops.batch_norm(residual,
                                      train=train,
                                      epsilon=1e-05,
                                      momentum=0.1,
                                      params=None if self.param_dict is None else self.param_dict['downsample']['bn'],
                                      dtype=self.dtype) 
        
        x += residual
        x = nn.relu(x)
        act[self.block_name] = x
        return x


class Bottleneck(nn.Module):
    """
    Bottleneck.

    Attributes:
        features (int): Number of output channels.
        kernel_size (Tuple): Kernel size.
        downsample (bool): If True, downsample spatial resolution.
        stride (bool): If True, use strides (2, 2). Not used in this module.
                       The attribute is only here for compatibility with Bottleneck.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        kernel_init (functools.partial): Kernel initializer.
        bias_init (functools.partial): Bias initializer.
        block_name (str): Name of block.
        expansion (int): Factor to multiply number of output channels with.
        dtype (str): Data type.
    """
    features: int
    kernel_size: Union[int, Iterable[int]]=(3, 3)
    downsample: bool=False
    stride: bool=True
    param_dict: Any=None
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    block_name: str=None
    expansion: int=4
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, act, train=True):
        """
        Run Bottleneck.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            act (dict): Dictionary containing activations.
            train (bool): Training mode.

        Returns:
            (tensor): Output shape of shape [N, H', W', features].
        """
        residual = x 
        
        x = nn.Conv(features=self.features, 
                    kernel_size=(1, 1), 
                    strides=(1, 1),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv1']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn1'],
                           dtype=self.dtype) 
        x = nn.relu(x)

        x = nn.Conv(features=self.features, 
                    kernel_size=(3, 3), 
                    strides=(2, 2) if self.downsample and self.stride else (1, 1), 
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv2']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)
        
        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn2'],
                           dtype=self.dtype) 
        x = nn.relu(x)

        x = nn.Conv(features=self.features * self.expansion, 
                    kernel_size=(1, 1), 
                    strides=(1, 1), 
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv3']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn3'],
                           dtype=self.dtype) 

        if self.downsample:
            residual = nn.Conv(features=self.features * self.expansion, 
                               kernel_size=(1, 1), 
                               strides=(2, 2) if self.stride else (1, 1), 
                               kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['downsample']['conv']['weight']), 
                               use_bias=False,
                               dtype=self.dtype)(residual)

            residual = ops.batch_norm(residual,
                                      train=train,
                                      epsilon=1e-05,
                                      momentum=0.1,
                                      params=None if self.param_dict is None else self.param_dict['downsample']['bn'],
                                      dtype=self.dtype) 
        
        x += residual
        x = nn.relu(x)
        act[self.block_name] = x
        return x


class ResNet(nn.Module):
    """
    ResNet.

    Attributes:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        architecture (str): 
            Which ResNet model to use:
                - 'resnet18'
                - 'resnet34'
                - 'resnet50'
                - 'resnet101'
                - 'resnet152'
        num_classes (int):
            Number of classes.
        block (nn.Module):
            Type of residual block:
                - BasicBlock
                - Bottleneck
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.
    """
    output: str='softmax'
    pretrained: str='imagenet'
    normalize: bool=True
    architecture: str='resnet18'
    num_classes: int=1000
    block: nn.Module=BasicBlock
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    ckpt_dir: str=None
    dtype: str='float32'

    def setup(self):
        self.param_dict = None
        if self.pretrained == 'imagenet':
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.architecture])
            self.param_dict = h5py.File(ckpt_file, 'r')

    @nn.compact
    def __call__(self, x, train=True, feature=False):
        """
        Args:
            x (tensor): Input tensor of shape [N, H, W, 3]. Images must be in range [0, 1].
            train (bool): Training mode.

        Returns:
            (tensor): Out
            If output == 'logits' or output == 'softmax':
                (tensor): Output tensor of shape [N, num_classes].
            If output == 'activations':
                (dict): Dictionary of activations.
        """
        if self.normalize:
            mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1).astype(self.dtype)  # EDITED
            std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1).astype(self.dtype)  # EDITED
            x = (x - mean) / std

        if self.pretrained == 'imagenet':
            # if self.num_classes != 1000: # EDITED
            #     warnings.warn(f'The user specified parameter \'num_classes\' was set to {self.num_classes} ' # EDITED
            #                     'but will be overwritten with 1000 to match the specified pretrained checkpoint \'imagenet\', if ', UserWarning) # EDITED
            # num_classes = 1000 # EDITED
            num_classes = self.num_classes # EDITED
        else:
            num_classes = self.num_classes
 
        act = {}

        x = nn.Conv(features=64, 
                    kernel_size=(7, 7),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv1']['weight']),
                    strides=(2, 2), 
                    padding=((3, 3), (3, 3)),
                    use_bias=False,
                    dtype=self.dtype)(x)
        # if self.pretrained:
        #     x = nn.Conv(features=64, 
        #                 kernel_size=(7, 7),
        #                 kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv1']['weight']),
        #                 strides=(2, 2), 
        #                 padding=((3, 3), (3, 3)),
        #                 use_bias=False,
        #                 dtype=self.dtype)(x)
        # else:
        #     x = nn.Conv(
        #         features=64,
        #         kernel_size=(3, 3),
        #         strides=(1, 1),
        #         padding=[(1, 1), (1, 1)],
        #         kernel_init=conv_kernel_init,
        #         use_bias=False
        #     )(x)
        act['conv1'] = x

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn1'],
                           dtype=self.dtype)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        # Layer 1
        down = self.block.__name__ == 'Bottleneck'
        for i in range(LAYERS[self.architecture][0]):
            params = None if self.param_dict is None else self.param_dict['layer1'][f'block{i}']
            x = self.block(features=64,
                           kernel_size=(3, 3),
                           downsample=i == 0 and down,
                           stride=i != 0,
                           param_dict=params,
                           block_name=f'block1_{i}',
                           dtype=self.dtype)(x, act, train)
        
        # Layer 2
        for i in range(LAYERS[self.architecture][1]):
            params = None if self.param_dict is None else self.param_dict['layer2'][f'block{i}']
            x = self.block(features=128,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block2_{i}',
                           dtype=self.dtype)(x, act, train)
        
        # Layer 3
        for i in range(LAYERS[self.architecture][2]):
            params = None if self.param_dict is None else self.param_dict['layer3'][f'block{i}']
            x = self.block(features=256,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block3_{i}',
                           dtype=self.dtype)(x, act, train)

        # Layer 4
        for i in range(LAYERS[self.architecture][3]):
            params = None if self.param_dict is None else self.param_dict['layer4'][f'block{i}']
            x = self.block(features=512,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block4_{i}',
                           dtype=self.dtype)(x, act, train)

        # Classifier
        x = jnp.mean(x, axis=(1, 2))

        if final_layer_retraining:  # TODO: add as class argument
            x = jax.lax.stop_gradient(x)

        feature_map = x

        x = nn.Dense(features=num_classes,
                     kernel_init=self.kernel_init if (self.param_dict is None or self.num_classes != 1000) else lambda *_ : jnp.array(self.param_dict['fc']['weight']),  # EDITED
                     bias_init=self.bias_init if (self.param_dict is None or self.num_classes != 1000)  else lambda *_ : jnp.array(self.param_dict['fc']['bias']),  # EDITED
                     dtype=self.dtype)(x)
        act['fc'] = x
        
        if self.output == 'softmax':
            return nn.softmax(x)
        if self.output == 'log_softmax':
            return nn.log_softmax(x)
        if self.output == 'activations':
            return act

        if feature:
            return (x, feature_map)
        else:
            return x


def ResNet18(output='softmax',
             pretrained='imagenet',
             normalize=True,
             num_classes=1000,
             kernel_init=nn.initializers.lecun_normal(),
             bias_init=nn.initializers.zeros,
             ckpt_dir=None,
             dtype='float32'):
    """
    Implementation of the ResNet18 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet18',
                  num_classes=num_classes,
                  block=BasicBlock,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


def ResNet34(output='softmax',
             pretrained='imagenet',
             normalize=True,
             num_classes=1000,
             kernel_init=nn.initializers.lecun_normal(),
             bias_init=nn.initializers.zeros,
             ckpt_dir=None,
             dtype='float32'):
    """
    Implementation of the ResNet34 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet34',
                  num_classes=num_classes,
                  block=BasicBlock,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


def ResNet50(output='softmax',
             pretrained='imagenet',
             normalize=True,
             num_classes=1000,
             kernel_init=nn.initializers.lecun_normal(),
             bias_init=nn.initializers.zeros,
             ckpt_dir=None,
             dtype='float32'):
    """
    Implementation of the ResNet50 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet50',
                  num_classes=num_classes,
                  block=Bottleneck,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


def ResNet101(output='softmax',
              pretrained='imagenet',
              normalize=True,
              num_classes=1000,
              kernel_init=nn.initializers.lecun_normal(),
              bias_init=nn.initializers.zeros,
              ckpt_dir=None,
              dtype='float32'):
    """
    Implementation of the ResNet101 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet101',
                  num_classes=num_classes,
                  block=Bottleneck,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


def ResNet152(output='softmax',
              pretrained='imagenet',
              normalize=True,
              num_classes=1000,
              kernel_init=nn.initializers.lecun_normal(),
              bias_init=nn.initializers.zeros,
              ckpt_dir=None,
              dtype='float32'):
    """
    Implementation of the ResNet152 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet152',
                  num_classes=num_classes,
                  block=Bottleneck,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


rng_key = main_rng

if 'CNN' in model_name:
    model_class = CNN
    num_blocks = None
    c_hidden = None
if 'BigCNN' in model_name:
    model_class = BigCNN
    num_blocks = None
    c_hidden = None
if 'MLP' in model_name:
    model_class = MLP
    num_blocks = None
    c_hidden = None
if 'MLP_Toy' in model_name:
    model_class = MLP_Toy
    num_blocks = None
    c_hidden = None
if 'MLP_OfflineRL' in model_name:
    model_class = MLP_OfflineRL
    num_blocks = None
    c_hidden = None
if 'ResNet9' in model_name:
    model_class = ResNetMod
    num_blocks = (3, 3, 3)
    c_hidden = (16, 32, 64)
if 'ResNet18' in model_name:
    model_class = ResNetMod
    # model_class = ResNet18
    num_blocks = (2, 2, 2, 2)
    c_hidden = (64, 128, 256, 512)
if 'ResNet50' in model_name:
    model_class = ResNetMod
    # model_class = ResNet50
    num_blocks = (3, 4, 6, 3)
    c_hidden = (64, 128, 256, 512)
if 'bert' in model_name:
    model_class = FlaxBertForSequenceClassification
    num_blocks = None
    c_hidden = None
if 'S4D' in model_name or ssm:
    model_class = None # BatchStackedModel
    num_blocks = None # TODO: double check on this. likely not applicable
    c_hidden = None # TODO: double check on this. likely not applicable
    
block_class = ResNetBlock
# block_class = PreActResNetBlock

act_fn = nn.relu
# act_fn = nn.swish

if prior_precision == 0:
    prior_precision = 1 / prior_var
elif prior_var == 0:
    prior_var = 1 / prior_precision
else:
    raise ValueError("Only one of prior_precision and prior_var can be set.")

# prior_mean = "Pretrained Mean" if "Pretrained" in model_name else prior_mean

if method == "psmap":
    stochastic = False
if method == "psvi":
    stochastic = True

resnet_trainer, resnet_results = trainer(
    model_name=model_name,
    model_class=model_class,
    model_hparams={
                        "num_classes": num_classes,
                        "c_hidden": c_hidden,
                        "num_blocks": num_blocks,
                        "act_fn": act_fn,
                        "block_class": block_class,
                        },
    optimizer_name=optimizer_name,
    optimizer_hparams={
                        "learning_rate": learning_rate,
                        "learning_rate_scale_logvar": learning_rate_scale_logvar,
                        "momentum": momentum,
                        "alpha": alpha,
                        "weight_decay": weight_decay,
                        },
    objective_hparams={
                        "method": method,
                        "lr_schedule_name": lr_schedule_name,
                        "stochastic": stochastic,
                        "reg_type": reg_type,
                        "reg_scale": reg_scale,
                        "empirical_fairness_prior_scale": empirical_fairness_prior_scale,
                        "llm_dropout": llm_dropout,
                        "prior_mean": prior_mean,
                        "prior_var": prior_var,
                        "prior_likelihood_scale": prior_likelihood_scale,
                        "prior_likelihood_f_scale": prior_likelihood_f_scale,
                        "prior_likelihood_cov_scale": prior_likelihood_cov_scale,
                        "prior_likelihood_cov_diag": prior_likelihood_cov_diag,
                        "prior_likelihood_mean": prior_likelihood_mean,
                        "prior_likelihood_normalize_feature": prior_likelihood_normalize_feature,
                        "likelihood_scale": likelihood_scale,
                        "rho_sam": rho_sam,
                        "rho_adversarial": rho_adversarial,
                        "dropout_rate_sam": dropout_rate_sam,
                        "context_points": context_points,
                        "forward_points": forward_points,
                        "reg_points": reg_points,
                        "mc_samples_llk": mc_samples_llk,
                        "mc_samples_reg": mc_samples_reg,
                        "training_dataset_size": training_dataset_size,
                        "batch_size": batch_size,
                        "init_logvar": init_logvar,
                        "init_final_layer_weights_logvar": init_final_layer_weights_logvar,
                        "init_final_layer_bias_logvar": init_final_layer_bias_logvar,
                        "prior_feature_logvar": prior_feature_logvar,
                        "pretrained_prior": pretrained_prior,
                        },
    ssm_hparams={       
                        "ssm": ssm,
                        "primary_type": primary_type,
                        "secondary_type": secondary_type,
                        "tertiary_type": tertiary_type,
                        "ssm_model_hparams": {
                                'd_model': init_d_model, 
                                'n_layers': init_n_layers, 
                                'dropout': init_dropout, 
                                'prenorm': init_prenorm, 
                                'embedding': init_embedding, 
                                'layer': {
                                    'N': init_layer_N,
                                    'l_max': init_layer_l_max,
                                    'decode': False,
                                    'scaling': init_layer_scaling,
                                    'log_step_dt_min': log_step_dt_min,
                                    'log_step_dt_max': log_step_dt_max
                                },
                            },
                        "ssm_optimizer_hparams": {
                                "optimizer_name": optimizer_name_ssm,
                                "learning_rate": learning_rate_ssm,
                                "momentum": momentum_ssm,
                                "alpha": alpha_ssm,
                                "weight_decay": weight_decay_ssm,
                            },
                        
                        },
    other_hparams={
                        "linearize": linearize,
                        "output_var": output_var,
                        "stochastic": stochastic,
                        "evaluate": evaluate,
                        "restore_checkpoint": restore_checkpoint,
                        "checkpoint_dir": checkpoint_dir,
                        "final_layer_random_init": final_layer_random_init,
                        "batch_stats_init_epochs": batch_stats_init_epochs,
                        "dataset": dataset,
                        "prediction_type": prediction_type,
                        "ood_points": ood_points,
                        "batch_size": batch_size,
                        "context_batch_size": context_batch_size,
                        "context_dataset_size": context_dataset_size,
                        "num_epochs": num_epochs,
                        "seed": seed,
                        "mc_samples_eval": mc_samples_eval,
                        "config_name": config_name,
                        "debug_print": debug_print,
                        "log_frequency": log_frequency,
                        "log_frequency_steps": log_frequency_steps,
                        "full_eval": full_eval,
                        "sensitive_attribute": sensitive_attribute,
                        "group_dro": group_dro,
                        "gdro_step_size": gdro_step_size,
                        # "gdro_weights": gdro_weights,
                        "quick_eval": quick_eval,
                        "data_balancing": data_balancing,
                        "fairness_eval": fairness_eval,
                        "fairness_train": fairness_train,
                        "val_train_frac": val_train_frac,
                        "dataset_group_scale": dataset_group_scale,
                        "validation_training": validation_training,
                        "final_layer_retraining": final_layer_retraining,
                        "save_to_wandb": save_to_wandb,
                        "wandb_project": wandb_project,
                        "wandb_account": wandb_account,
                        },
    exmp_inputs=exmp_input,
    rng_key=rng_key,
    )


print(f"\nDone\n")
