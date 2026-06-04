# Reproducing Experiments
Reproduce all figures from the paper.

For custom models, see the [optional section below](#running-your-memory-model).

## Train
Train a model and log to Weights & Biases (wandb). Replace `MEMORY_TYPE` with one of the supported models.

```bash
python train.py PQN_RNN --MEMORY_TYPE=my_rnn --PROJECT=my_project
```
Model weights will be saved locally after training.


## Observability Gap and Memory Bias

Download the run history from wandb and plot the gap/bias.
```python
python plotting/download_csv.py \ 
    --entity wandb_entity \
    --project=wandb_project_name \
    --model-group-csv my_rnn.csv

python plotting/return_gap_bias.py \
    --input-csv my_rnn.csv \
    --output gap_bias.pdf
```


## Pixel Saliency
```python
python plotting/pixel_vis_pqn \
    --model-path my_rnn_weight.pkl \
    --env-name CartPoleEasy \
    --memory_type my_rnn \
    --output pixels.pdf
```

## Recall Density
```python
python plotting/density_analysis_pqn.py \
    --model-dir model_weights_dir \
    --out_dir your_recall_density_dir

python plotting/plot_density_summary.py \
    --recall_density_dir your_recall_density_dir \
    --output density.pdf
```

## Memory Contamination
```python
python plotting/noiseva.py \
    --model-dir /path/to/checkpoints \
    --memory-types my_rnn \
    --env-names CartPoleEasy
```

## Running Your Memory Model
For a new a memory model, you must implement a memory model then register it.

Your model must expose two methods:

1. `__call__(self, recurrent_state, (obs, done)) -> recurrent_state, markov_state`

2. `initialize_carry(self) -> recurrent_state`

Consult the [`memax`](https://github.com/smorad/memax) library for exact registration details.