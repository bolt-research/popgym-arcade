import functools
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import wandb
import tqdm
from multiprocessing.pool import ThreadPool
import scipy.stats

import os

model_renames = {
    "StackedLRU": "LRU",
    "StackedS5": "S5",
    "LinearAttention": "LinAttn",
    "popgym.envs.CountRecallEasy": "Count Recall",
    "popgym.envs.PositionOnlyCartPoleEasy": "Pos. Cartpole",
    "popgym.envs.RepeatFirstEasy": "Repeat First",
    "popgym.envs.RepeatPrevious": "Repeat Previous",
    "popgym.envs.RepeatPreviousEasy": "Repeat Previous",
}
models = [
    "LRU",
    "S5",
    "FFM",
    "LinAttn",
]

def confidence_interval(a, confidence=0.95):
    """Returns the confidence interval of a 1D array of values or a pandas.Series"""
    return scipy.stats.t.interval(confidence, len(a)-1, loc=np.mean(a), scale=scipy.stats.sem(a))

def sort_by_cat(df, sort_key, value_key) -> pd.DataFrame:
    """Order a categorical column value_key by the order of the sort_key column"""
    order = df.sort_values(sort_key)[value_key].unique().tolist()
    df = df.copy()
    df[value_key] = df[value_key].cat.reorder_categories(order, ordered=True)
    return df


def recategorize(df, keys=["Env", "Base Env", "Difficulty", "Model"]):
    """Set specific columns to categorical dtypes"""
    # Convert to categoricals for easy sorting
    for key in keys:
        df[key] = df[key].astype("category")

    if "Difficulty" in keys:
        order = []
        uniqs = df["Difficulty"].unique()
        if "Easy" in uniqs:
            order.append("Easy")
        if "Medium" in uniqs:
            order.append("Medium")
        if "Hard" in uniqs:
            order.append("Hard")
        df["Difficulty"] = df["Difficulty"].cat.reorder_categories(
            order, ordered=True
        )
    return df

def finished_run_filter(run: wandb.sdk.wandb_run.Run) -> bool:
    return run.state.lower() == "finished"

def process_run(
    run: wandb.sdk.wandb_run.Run,
    timesteps: int,
    x_key: str="collect/train_epoch",
    metric_keys: Dict[str, str]={
        "eval/collect/best_reward": "Return",
        "collect/train_epoch": "Train Epoch",
    },
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process a run and return the resulting dataframe.

    Args:
        run: A wandb run object
        timesteps: The maximum number of timesteps to include in the dataframe
        x_key: The key to use for the dataframe index (x axis)
        metric_keys: A dictionary mapping metric keys we want to track to new, human-readable names.
            Note that these are used to index the wandb data, your dataframe might have additional columns.

    Returns:
        A dataframe with the processed metrics
    """
    try:
        #run_df = run.history(keys=list(metric_keys.keys()), x_axis=x_key, pandas=True)
        run_df = pd.DataFrame(run.scan_history(keys=list(metric_keys.keys())))
    except wandb.errors.CommError:
        #run_df = run.history(keys=list(metric_keys.keys()), x_axis=x_key, pandas=True)
        run_df = pd.DataFrame(run.scan_history(keys=list(metric_keys.keys())))

    # Subsample at fixed intervals
    # run_df = run_df[
    #     (run_df["collect/train_epoch"] == 1) | 
    #     (run_df["collect/train_epoch"] % 10 == 0)
    # ]
    run_df = run_df.reset_index(drop=True)

    timesteps_total = run_df.get(x_key, np.array([0])).max()

    if (timesteps_total < timesteps).any():
        print(f"run {run.name} {run.id} is corrupted (t={timesteps_total})")
    #run_df = run_df[run_df.get(x_key, np.array([0])) <= timesteps]

    # run_df["name"] = run.name
    # run_df["Env"] = env
    # run_df["Model"] = model
    # run_df["run_id"] = run.id
    # run_df["Reward"] = run_df["eval/collect/best_reward"]
    # run_df["Batch Mode"] = "SBB" if run.config['collect'].get("segment_length") else "TBB"
    # run_df["Segment Length"] = run.config['collect'].get("segment_length")
    # run_df["Batch Size"] = run.config['train']['batch_size']

    run_df["State"] = run.state
    
    run_df["Model"] = run.config.get("MEMORY_TYPE", "Unknown")
    run_df["Env"] = run.config.get("ENV_NAME", "Unknown")
    run_df["Algorithm"] = run.config.get("ALG_NAME", "Unknown")
    if run_df["Algorithm"] == "PQN" or run_df["ALG_NAME"] == "PQN_RNN":
        run_df["Return"] = run_df["returned_episode_returns"]
        run_df["Steps"] = run_df["TOTAL_TIMESTEPS"]
    else:
        run_df["Return"] = run_df["episodic return"] 
        run_df["Steps"] = run_df["global step"]
    
    
    summary_df = pd.DataFrame(
        {
            "name": run_df["name"][0],
            "Env": run_df["Env"][0],
            "Model": run_df["Model"][0],
            "run_id": [run.id],
            "Reward": run_df["Reward"][-1:],
            "Batch Mode": run_df["Batch Mode"][0],
            "Segment Length": run_df["Segment Length"][0],
            "Batch Size": run_df["Batch Size"][0],
        }
    )
    return run_df, summary_df


def build_df(
    wandb_project: str,
    csv_dir: str,
    timesteps: int,
    clean=False,
    metric_keys: Dict[str, str]={
        "episode_reward_mean": "Episodic Reward",
        "training_iteration": "Epoch",
    },
    x_key: str="timesteps_total",
    column_renames: Dict[str, str]=model_renames,
    process_fn: Callable[[wandb.sdk.wandb_run.Run, int, str, Dict[str, str]], Tuple[pd.DataFrame, pd.DataFrame]]=process_run,
    run_filter: Callable[[wandb.sdk.wandb_run.Run], bool]=lambda run: True,
    multiprocess: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a dataframe from wandb project and save it to csv_path.
    If the csv_path already exists, it will be loaded as a dataframe instead.

    Args:
        wandb_project: The wandb project to load
        csv_dir: The directory to save the csv files to
        timesteps: The maximum number of timesteps to include in the dataframe
        clean: If True, the csv files will be deleted and rebuilt
        metric_keys: A dictionary mapping metric keys (cols) we want to track to new, human-readable names.
        x_key: The key to use for the dataframe index (x axis)
        column_renames: A dictionary mapping metric values (rows) to new, human-readable names.
        process_fn: The function to use to process the wandb run and return two dataframes

    Returns:
        A tuple of dataframes, the first is the run dataframe and the second is the summary dataframe

    Side Effects:
        Saves the run and summary dataframes to {project}-runs.csv and {projects}-summary.csv in csv_dir
    """
    os.makedirs(csv_dir, exist_ok=True)
    run_csv_path = f"{csv_dir}/{wandb_project}-runs.csv"
    summary_csv_path = f"{csv_dir}/{wandb_project}-summary.csv"
    if os.path.isfile(run_csv_path) and os.path.isfile(summary_csv_path) and not clean:
        return pd.read_csv(run_csv_path), pd.read_csv(summary_csv_path)

    api = wandb.Api(timeout=90)
    project = api.runs(wandb_project)
    run_dfs = []
    summary_dfs = []
    kwargs = {k: v for k, v in {"timesteps": timesteps, "x_key": x_key, "metric_keys": metric_keys}.items() if v is not None}
    if multiprocess:
        #runs = list(project)
        #fn = functools.partial(process_fn, timesteps=timesteps, x_key=x_key, metric_keys=metric_keys)
        pool = ThreadPool(16)
        fn = lambda run: process_fn(run, **kwargs) if run_filter(run) else (None, None)
        result = tqdm.tqdm(pool.imap_unordered(fn, project), total=len(project))
        result = [r for r in result if r is not (None, None)]
        run_dfs, summary_dfs = zip(*result)
    else:
        for run in tqdm.tqdm(project):
            run_df, summary_df = process_fn(run, **kwargs) if run_filter(run) else (None, None)
            run_dfs.append(run_df)
            summary_dfs.append(summary_df)
    project_df = pd.concat(run_dfs, ignore_index=True)
    summary_df = pd.concat(summary_dfs, ignore_index=True)
    # Sometimes wandb will return duplicates, no clue why...
    project_df = project_df.drop_duplicates().reset_index(drop=True)
    summary_df = summary_df.drop_duplicates().reset_index(drop=True)
    # Add trial index
    summary_df['trial_idx'] = -1
    for name in summary_df['name'].unique():
        mask = summary_df['name'] == name
        summary_df.loc[mask, 'trial_idx'] = range(mask.sum())

    # Add trial index to project_df
    for run_id in summary_df['run_id'].unique():
        proj_mask = project_df['run_id'] == run_id
        sum_mask = summary_df['run_id'] == run_id
        project_df.loc[proj_mask, 'trial_idx'] = summary_df[sum_mask]['trial_idx'].values[0]

    project_df = project_df.replace(column_renames)
    project_df = project_df.rename(columns=metric_keys)
    summary_df = summary_df.rename(columns=metric_keys)
    summary_df = summary_df.replace(column_renames)

    project_df.to_csv(run_csv_path)
    summary_df.to_csv(summary_csv_path)
    return project_df, summary_df


def build_projects(
    projects: Dict, workdir: str, clean: bool = False, recategorize_keys=["Env", "Base Env", "Difficulty", "Model"], **build_df_kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds all projects into a run and summary dataframe
    Args:
        projects: A dictionary mapping project names to (timesteps, epochs) tuples
        clean: If True, the csv files will be deleted and rebuilt
    Returns:
        A tuple of dataframes, the first is the run dataframe and the second is the summary dataframe
    """
    runs, summaries = [], []
    for project, (timesteps, epochs) in projects.items():
        run, summary = build_df(
            project, workdir, timesteps, clean=clean, **build_df_kwargs
        )
        runs.append(run)
        summaries.append(summary)
    runs = recategorize(pd.concat(runs), recategorize_keys)
    summary = recategorize(pd.concat(summaries), recategorize_keys)
    runs = runs.reset_index(drop=True)
    summary = summary.reset_index(drop=True)
    return runs, summary