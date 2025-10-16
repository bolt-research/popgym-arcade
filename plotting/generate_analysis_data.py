import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import the function from run_multi_seed_analysis.py
from plotting.run_multi_seed_analysis import run_multiple_seeds_and_save_csv


def analyze_model_saliency(
    config, seeds=[0, 1, 2, 3, 4], max_steps=200, visualize=True
):
    """
    Analyze the saliency maps of a model with the given configuration.

    Args:
        config (dict): Dictionary containing model configuration with keys:
            - ENV_NAME: Environment name
            - PARTIAL: Whether to use partial observations
            - MEMORY_TYPE: Type of memory to use
            - OBS_SIZE: Size of observations
            - MODEL_SEED: Seed used for the model (to locate model file)
        seeds (list): List of seeds to run the analysis with
        max_steps (int): Maximum number of steps per episode
        visualize (bool): Whether to create visualization plots

    Returns:
        dict: A dictionary containing:
            - csv_path: Path to the CSV file with results
            - avg_plot_path: Path to the average saliency plot
            - individual_plot_path: Path to the individual seeds saliency plot
    """
    output_csv = f'saliency_results_{config["MEMORY_TYPE"]}_{config["ENV_NAME"]}_Partial={config["PARTIAL"]}.csv'

    # Run the analysis for all seeds
    output_csv = run_multiple_seeds_and_save_csv(
        config, seeds, max_steps=max_steps, output_csv=output_csv
    )

    result_paths = {"csv_path": output_csv}

    if visualize:
        # Load the results for visualization
        results_df = pd.read_csv(output_csv)

        # Create a visualization of the distributions for each seed
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")

        # Filter columns that represent positions
        pos_columns = [col for col in results_df.columns if col.startswith("pos_")]

        # Plot each seed's distribution
        for idx, row in results_df.iterrows():
            seed = row["seed"]
            positions = [float(col.split("_")[1]) for col in pos_columns]
            values = [row[col] for col in pos_columns]
            plt.plot(positions, values, marker="o", markersize=3, label=f"Seed {seed}")

        plt.xlabel("Normalized Episode Position")
        plt.ylabel("Saliency Magnitude")
        plt.title(
            f"Terminal Saliency Distribution by Seed\n{config['MEMORY_TYPE']} on {config['ENV_NAME']}"
        )
        plt.legend()
        plt.tight_layout()

        individual_plot_path = f"saliency_plot_{config['MEMORY_TYPE']}_{config['ENV_NAME']}_Partial={config['PARTIAL']}.png"
        plt.savefig(individual_plot_path, dpi=300)
        plt.close()

        # Calculate average distribution across seeds
        avg_values = [results_df[col].mean() for col in pos_columns]
        std_values = [results_df[col].std() for col in pos_columns]
        positions = [float(col.split("_")[1]) for col in pos_columns]

        plt.figure(figsize=(12, 8))
        plt.plot(positions, avg_values, "b-", linewidth=2, label="Mean Distribution")
        plt.fill_between(
            positions,
            [avg - std for avg, std in zip(avg_values, std_values)],
            [avg + std for avg, std in zip(avg_values, std_values)],
            color="b",
            alpha=0.2,
            label="Standard Deviation",
        )

        plt.xlabel("Normalized Episode Position")
        plt.ylabel("Average Saliency Magnitude")
        plt.title(
            f"Average Terminal Saliency Distribution Across Seeds\n{config['MEMORY_TYPE']} on {config['ENV_NAME']}"
        )
        plt.legend()
        plt.tight_layout()

        avg_plot_path = f"avg_saliency_plot_{config['MEMORY_TYPE']}_{config['ENV_NAME']}_Partial={config['PARTIAL']}.png"
        plt.savefig(avg_plot_path, dpi=300)
        plt.close()

        result_paths["individual_plot_path"] = individual_plot_path
        result_paths["avg_plot_path"] = avg_plot_path

    print(f"Analysis complete. Results saved to: {output_csv}")
    return result_paths


if __name__ == "__main__":
    configs = [
        {
            "ENV_NAME": "AutoEncodeEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 3,
        },
        {
            "ENV_NAME": "AutoEncodeEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 4,
        },
        {
            "ENV_NAME": "BattleShipEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 0,
        },
        {
            "ENV_NAME": "BattleShipEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 0,
        },
        {
            "ENV_NAME": "CartPoleEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 0,
        },
        {
            "ENV_NAME": "CartPoleEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 1,
        },
        {
            "ENV_NAME": "CountRecallEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 0,
        },
        {
            "ENV_NAME": "CountRecallEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 0,
        },
        {
            "ENV_NAME": "MineSweeperEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 4,
        },
        {
            "ENV_NAME": "MineSweeperEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 3,
        },
        {
            "ENV_NAME": "NavigatorEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 2,
        },
        {
            "ENV_NAME": "NavigatorEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 1,
        },
        {
            "ENV_NAME": "NoisyCartPoleEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 4,
        },
        {
            "ENV_NAME": "NoisyCartPoleEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "fart",
            "OBS_SIZE": 128,
            "MODEL_SEED": 0,
        },
        # lru models
        {
            "ENV_NAME": "AutoEncodeEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 0,
        },
        {
            "ENV_NAME": "AutoEncodeEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 0,
        },
        {
            "ENV_NAME": "BattleShipEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 1,
        },
        {
            "ENV_NAME": "BattleShipEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 2,
        },
        {
            "ENV_NAME": "CartPoleEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 3,
        },
        {
            "ENV_NAME": "CartPoleEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 3,
        },
        {
            "ENV_NAME": "CountRecallEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 0,
        },
        {
            "ENV_NAME": "CountRecallEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 1,
        },
        {
            "ENV_NAME": "MineSweeperEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 4,
        },
        {
            "ENV_NAME": "MineSweeperEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 1,
        },
        {
            "ENV_NAME": "NavigatorEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 2,
        },
        {
            "ENV_NAME": "NavigatorEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 4,
        },
        {
            "ENV_NAME": "NoisyCartPoleEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 2,
        },
        {
            "ENV_NAME": "NoisyCartPoleEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "lru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 2,
        },
        # mingru models
        {
            "ENV_NAME": "AutoEncodeEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 1,
        },
        {
            "ENV_NAME": "AutoEncodeEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 1,
        },
        {
            "ENV_NAME": "BattleShipEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 2,
        },
        {
            "ENV_NAME": "BattleShipEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 2,
        },
        {
            "ENV_NAME": "CartPoleEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 4,
        },
        {
            "ENV_NAME": "CartPoleEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 4,
        },
        {
            "ENV_NAME": "CountRecallEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 2,
        },
        {
            "ENV_NAME": "CountRecallEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 2,
        },
        {
            "ENV_NAME": "MineSweeperEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 0,
        },
        {
            "ENV_NAME": "MineSweeperEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 2,
        },
        {
            "ENV_NAME": "NavigatorEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 0,
        },
        {
            "ENV_NAME": "NavigatorEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 0,
        },
        {
            "ENV_NAME": "NoisyCartPoleEasy",
            "PARTIAL": False,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 3,
        },
        {
            "ENV_NAME": "NoisyCartPoleEasy",
            "PARTIAL": True,
            "MEMORY_TYPE": "mingru",
            "OBS_SIZE": 128,
            "MODEL_SEED": 4,
        },
    ]

    seeds = [0, 1, 2, 3, 4]

    for config in configs:
        print(
            f"Analyzing {config['MEMORY_TYPE']} on {config['ENV_NAME']} (Partial={config['PARTIAL']}, Seed={config['MODEL_SEED']})"
        )

        results = analyze_model_saliency(
            config=config, seeds=seeds, max_steps=200, visualize=True
        )
