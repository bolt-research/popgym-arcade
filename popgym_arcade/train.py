import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    # model settings
    subparsers = parser.add_subparsers(dest="TRAIN_TYPE")
    # ppo parser
    ppo_parser = subparsers.add_parser("PPO", help="training with PPO")

    ppo_parser.add_argument("--SEED", type=int, default=0, help="Random seed")
    ppo_parser.add_argument(
        "--NUM_SEEDS", type=int, default=1, help="Number of Random seeds"
    )

    ppo_parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
    ppo_parser.add_argument(
        "--NUM_ENVS", type=int, default=16, help="Number of environments"
    )
    ppo_parser.add_argument(
        "--NUM_STEPS", type=int, default=128, help="Number of steps"
    )
    ppo_parser.add_argument(
        "--TOTAL_TIMESTEPS", type=int, default=1e7, help="Total timesteps"
    )
    ppo_parser.add_argument(
        "--UPDATE_EPOCHS", type=int, default=4, help="Number of update epochs"
    )
    ppo_parser.add_argument(
        "--NUM_MINIBATCHES", type=int, default=16, help="Number of minibatches"
    )
    ppo_parser.add_argument(
        "--GAMMA", type=float, default=0.99, help="Discount factor for rewards"
    )
    ppo_parser.add_argument("--GAE_LAMBDA", type=float, default=0.95, help="GAE lambda")
    ppo_parser.add_argument(
        "--CLIP_EPS", type=float, default=0.2, help="Clipping gradients epsilon"
    )
    ppo_parser.add_argument(
        "--ENT_COEF", type=float, default=0.01, help="Entropy coefficient"
    )
    ppo_parser.add_argument(
        "--VF_COEF", type=float, default=0.5, help="Value function coefficient"
    )
    ppo_parser.add_argument(
        "--MAX_GRAD_NORM", type=float, default=0.5, help="Max gradient norm"
    )
    ppo_parser.add_argument(
        "--ENV_NAME", type=str, default="CartPoleHard", help="Environment name"
    )
    ppo_parser.add_argument(
        "--PARTIAL", action="store_true", help="Partial Observations"
    )
    ppo_parser.add_argument(
        "--ANNEAL_LR", type=bool, default=True, help="Anneal learning rate"
    )
    ppo_parser.add_argument("--DEBUG", type=bool, default=True, help="Debug mode")
    ppo_parser.add_argument(
        "--PROJECT",
        type=str,
        default="popgym_arcade-acrade-",
        help="WanDB Project name",
    )
    ppo_parser.add_argument("--ENTITY", type=str, default="", help="Entity name")
    ppo_parser.add_argument(
        "--WANDB_MODE", type=str, default="online", help="WanDB mode"
    )
    ppo_parser.add_argument(
        "--OBS_SIZE", type=int, default=128, help="Observation size"
    )
    # ppo with rnn parser
    ppo_rnn_parser = subparsers.add_parser(
        "PPO_RNN", help="training with PPO using RNN models"
    )

    ppo_rnn_parser.add_argument(
        "--MEMORY_TYPE", type=str, default="lru", help="Memory model type."
    )

    ppo_rnn_parser.add_argument("--SEED", type=int, default=0, help="Random seed")
    ppo_rnn_parser.add_argument(
        "--NUM_SEEDS", type=int, default=1, help="Number of Random seeds"
    )

    ppo_rnn_parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
    ppo_rnn_parser.add_argument(
        "--NUM_ENVS", type=int, default=16, help="Number of environments"
    )
    ppo_rnn_parser.add_argument(
        "--NUM_STEPS", type=int, default=128, help="Number of steps"
    )
    ppo_rnn_parser.add_argument(
        "--TOTAL_TIMESTEPS", type=int, default=1e7, help="Total timesteps"
    )
    ppo_rnn_parser.add_argument(
        "--UPDATE_EPOCHS", type=int, default=4, help="Number of update epochs"
    )
    ppo_rnn_parser.add_argument(
        "--NUM_MINIBATCHES", type=int, default=16, help="Number of minibatches"
    )
    ppo_rnn_parser.add_argument(
        "--GAMMA", type=float, default=0.99, help="Discount factor for rewards"
    )
    ppo_rnn_parser.add_argument(
        "--GAE_LAMBDA", type=float, default=0.95, help="GAE lambda"
    )
    ppo_rnn_parser.add_argument(
        "--CLIP_EPS", type=float, default=0.2, help="Clipping gradients epsilon"
    )
    ppo_rnn_parser.add_argument(
        "--ENT_COEF", type=float, default=0.01, help="Entropy coefficient"
    )
    ppo_rnn_parser.add_argument(
        "--VF_COEF", type=float, default=0.5, help="Value function coefficient"
    )
    ppo_rnn_parser.add_argument(
        "--MAX_GRAD_NORM", type=float, default=0.5, help="Max gradient norm"
    )
    ppo_rnn_parser.add_argument(
        "--ENV_NAME", type=str, default="CartPoleHard", help="Environment name"
    )
    ppo_rnn_parser.add_argument(
        "--PARTIAL", action="store_true", help="Partial Observations"
    )
    ppo_rnn_parser.add_argument(
        "--ANNEAL_LR", type=bool, default=True, help="Anneal learning rate"
    )
    ppo_rnn_parser.add_argument("--DEBUG", type=bool, default=True, help="Debug mode")
    ppo_rnn_parser.add_argument(
        "--PROJECT",
        type=str,
        default="popgym_arcade-acrade-",
        help="WanDB Project name",
    )
    ppo_rnn_parser.add_argument("--ENTITY", type=str, default="", help="Entity name")
    ppo_rnn_parser.add_argument(
        "--WANDB_MODE", type=str, default="online", help="WanDB mode"
    )
    ppo_rnn_parser.add_argument(
        "--OBS_SIZE", type=int, default=128, help="Observation size"
    )

    # pqn parser
    pqn_parser = subparsers.add_parser("PQN", help="Training with PQN")
    pqn_parser.add_argument(
        "--MEMORY_TYPE", type=str, default="MLP", help="Memory model type."
    )
    pqn_parser.add_argument(
        "--TOTAL_TIMESTEPS", type=int, default=2e7, help="Total timesteps"
    )
    pqn_parser.add_argument(
        "--TOTAL_TIMESTEPS_DECAY",
        type=int,
        default=2e6,
        help="Total timesteps decay will be used for decay functions, in case you want to test for less timesteps and keep decays same.",
    )
    pqn_parser.add_argument(
        "--NUM_ENVS", type=int, default=16, help="Parallel Environments"
    )
    pqn_parser.add_argument(
        "--MEMORY_WINDOW",
        type=int,
        default=4,
        help="steps of previous episode added in the rnn training horizon",
    )
    pqn_parser.add_argument(
        "--NUM_STEPS",
        type=int,
        default=128,
        help="steps per environment in each update",
    )
    pqn_parser.add_argument("--EPS_START", type=float, default=1, help="Epsilon start")
    pqn_parser.add_argument(
        "--EPS_FINISH", type=float, default=0.05, help="Epsilon finish"
    )
    pqn_parser.add_argument(
        "--EPS_DECAY", type=float, default=0.25, help="Epsilon decay"
    )
    pqn_parser.add_argument(
        "--NUM_MINIBATCHES", type=int, default=16, help="minibatches per epoch"
    )
    pqn_parser.add_argument(
        "--NUM_EPOCHS", type=int, default=4, help="minibatches per epoch"
    )
    pqn_parser.add_argument(
        "--NORM_INPUT", type=bool, default=False, help="Normalize input using LayerNorm"
    )
    pqn_parser.add_argument("--HIDDEN_SIZE", type=int, default=256, help="Hidden size")
    pqn_parser.add_argument(
        "--NUM_LAYERS", type=int, default=2, help="Number of layers"
    )
    pqn_parser.add_argument(
        "--NORM_TYPE", type=str, default="layer_norm", help="Normalization type"
    )
    pqn_parser.add_argument("--LR", type=float, default=0.00005, help="Learning rate")
    pqn_parser.add_argument(
        "--MAX_GRAD_NORM", type=float, default=0.5, help="Max gradient norm"
    )
    pqn_parser.add_argument(
        "--LR_LINEAR_DECAY", type=bool, default=True, help="Linear decay learning rate"
    )
    pqn_parser.add_argument("--REW_SCALE", type=float, default=1, help="Reward scale")
    pqn_parser.add_argument(
        "--GAMMA", type=float, default=0.99, help="Discount factor for rewards"
    )
    pqn_parser.add_argument("--LAMBDA", type=float, default=0.95, help="Lambda")
    pqn_parser.add_argument(
        "--HYP_TUNE", type=bool, default=False, help="Hyperparameter tuning"
    )
    pqn_parser.add_argument("--ENTITY", type=str, default="", help="Entity name")
    pqn_parser.add_argument(
        "--PROJECT", type=str, default="NavigatorEasy", help="WanDB Project name"
    )
    pqn_parser.add_argument(
        "--WANDB_MODE", type=str, default="online", help="WanDB mode"
    )
    pqn_parser.add_argument("--SEED", type=int, default=0, help="Random seed")
    pqn_parser.add_argument(
        "--NUM_SEEDS", type=int, default=1, help="Number of Random seeds"
    )
    pqn_parser.add_argument(
        "--PARTIAL", action="store_true", help="Partial Observations"
    )
    pqn_parser.add_argument(
        "--ENV_NAME", type=str, default="BattleShipEasy", help="Environment name"
    )
    pqn_parser.add_argument(
        "--ENV_KWARGS", type=dict, default={}, help="Environment kwargs"
    )
    pqn_parser.add_argument(
        "--TEST_DURING_TRAINING", type=bool, default=False, help="Test during training"
    )
    pqn_parser.add_argument(
        "--TEST_INTERVAL", type=float, default=0.05, help="In terms of total updatesl"
    )
    pqn_parser.add_argument(
        "--TEST_NUM_ENVS", type=int, default=128, help="Number of test environments"
    )
    pqn_parser.add_argument(
        "--EPS_TEST", type=float, default=0, help="0 for greedy policy"
    )
    pqn_parser.add_argument(
        "--ALG_NAME", type=str, default="PQN", help="Algorithm name"
    )
    pqn_parser.add_argument(
        "--OBS_SIZE", type=int, default=128, help="Observation size"
    )


    # PQN_RNN
    pqn_rnn_parser = subparsers.add_parser("PQN_RNN", help="Training with PQN_RNN")
    pqn_rnn_parser.add_argument(
        "--MEMORY_TYPE", type=str, default="MLP", help="Memory model type."
    )
    pqn_rnn_parser.add_argument(
        "--TOTAL_TIMESTEPS", type=int, default=2e7, help="Total timesteps"
    )
    pqn_rnn_parser.add_argument(
        "--TOTAL_TIMESTEPS_DECAY",
        type=int,
        default=2e6,
        help="Total timesteps decay will be used for decay functions, in case you want to test for less timesteps and keep decays same.",
    )
    pqn_rnn_parser.add_argument(
        "--NUM_ENVS", type=int, default=16, help="Parallel Environments"
    )
    pqn_rnn_parser.add_argument(
        "--MEMORY_WINDOW",
        type=int,
        default=4,
        help="steps of previous episode added in the rnn training horizon",
    )
    pqn_rnn_parser.add_argument(
        "--NUM_STEPS",
        type=int,
        default=128,
        help="steps per environment in each update",
    )
    pqn_rnn_parser.add_argument(
        "--EPS_START", type=float, default=1, help="Epsilon start"
    )
    pqn_rnn_parser.add_argument(
        "--EPS_FINISH", type=float, default=0.05, help="Epsilon finish"
    )
    pqn_rnn_parser.add_argument(
        "--EPS_DECAY", type=float, default=0.25, help="Epsilon decay"
    )
    pqn_rnn_parser.add_argument(
        "--NUM_MINIBATCHES", type=int, default=16, help="minibatches per epoch"
    )
    pqn_rnn_parser.add_argument(
        "--NUM_EPOCHS", type=int, default=4, help="minibatches per epoch"
    )
    pqn_rnn_parser.add_argument(
        "--NORM_INPUT", type=bool, default=False, help="Normalize input using LayerNorm"
    )
    pqn_rnn_parser.add_argument(
        "--HIDDEN_SIZE", type=int, default=256, help="Hidden size"
    )
    pqn_rnn_parser.add_argument(
        "--NUM_LAYERS", type=int, default=2, help="Number of layers"
    )
    pqn_rnn_parser.add_argument(
        "--NORM_TYPE", type=str, default="layer_norm", help="Normalization type"
    )
    pqn_rnn_parser.add_argument(
        "--LR", type=float, default=0.00005, help="Learning rate"
    )
    pqn_rnn_parser.add_argument(
        "--MAX_GRAD_NORM", type=float, default=0.5, help="Max gradient norm"
    )
    pqn_rnn_parser.add_argument(
        "--LR_LINEAR_DECAY", type=bool, default=True, help="Linear decay learning rate"
    )
    pqn_rnn_parser.add_argument(
        "--REW_SCALE", type=float, default=1, help="Reward scale"
    )
    pqn_rnn_parser.add_argument(
        "--GAMMA", type=float, default=0.99, help="Discount factor for rewards"
    )
    pqn_rnn_parser.add_argument("--LAMBDA", type=float, default=0.95, help="Lambda")
    pqn_rnn_parser.add_argument(
        "--HYP_TUNE", type=bool, default=False, help="Hyperparameter tuning"
    )
    pqn_rnn_parser.add_argument("--ENTITY", type=str, default="", help="Entity name")
    pqn_rnn_parser.add_argument(
        "--PROJECT", type=str, default="NavigatorEasy", help="WanDB Project name"
    )
    pqn_rnn_parser.add_argument(
        "--WANDB_MODE", type=str, default="online", help="WanDB mode"
    )
    pqn_rnn_parser.add_argument("--SEED", type=int, default=0, help="Random seed")
    pqn_rnn_parser.add_argument(
        "--NUM_SEEDS", type=int, default=1, help="Number of Random seeds"
    )
    pqn_rnn_parser.add_argument(
        "--PARTIAL", action="store_true", help="Partial Observations"
    )
    pqn_rnn_parser.add_argument(
        "--ENV_NAME", type=str, default="BattleShipEasy", help="Environment name"
    )
    pqn_rnn_parser.add_argument(
        "--ENV_KWARGS", type=dict, default={}, help="Environment kwargs"
    )
    pqn_rnn_parser.add_argument(
        "--TEST_DURING_TRAINING", type=bool, default=False, help="Test during training"
    )
    pqn_rnn_parser.add_argument(
        "--TEST_INTERVAL", type=float, default=0.05, help="In terms of total updatesl"
    )
    pqn_rnn_parser.add_argument(
        "--TEST_NUM_ENVS", type=int, default=128, help="Number of test environments"
    )
    pqn_rnn_parser.add_argument(
        "--EPS_TEST", type=float, default=0, help="0 for greedy policy"
    )
    pqn_rnn_parser.add_argument(
        "--ALG_NAME", type=str, default="PQN_RNN", help="Algorithm name"
    )
    pqn_rnn_parser.add_argument(
        "--OBS_SIZE", type=int, default=128, help="Observation size"
    )


    # DQN
    dqn_parser = subparsers.add_parser("DQN", help="Training with DQN")
    dqn_parser.add_argument(
        "--MEMORY_TYPE", type=str, default="MLP", help="Memory model type."
    )
    dqn_parser.add_argument(
        "--TOTAL_TIMESTEPS", type=int, default=2e7, help="Total timesteps"
    )
    dqn_parser.add_argument(
        "--TOTAL_TIMESTEPS_DECAY",
        type=int,
        default=2e6,
        help="Total timesteps decay will be used for decay functions, in case you want to test for less timesteps and keep decays same.",
    )
    dqn_parser.add_argument(
        "--NUM_ENVS", type=int, default=16, help="Parallel Environments"
    )
    dqn_parser.add_argument(
        "--MEMORY_WINDOW",
        type=int,
        default=4,
        help="steps of previous episode added in the rnn training horizon",
    )
    dqn_parser.add_argument(
        "--NUM_STEPS",
        type=int,
        default=128,
        help="steps per environment in each update",
    )
    dqn_parser.add_argument("--EPS_START", type=float, default=1, help="Epsilon start")
    dqn_parser.add_argument(
        "--EPS_FINISH", type=float, default=0.05, help="Epsilon finish"
    )
    dqn_parser.add_argument(
        "--EPS_DECAY", type=float, default=0.25, help="Epsilon decay"
    )
    dqn_parser.add_argument(
        "--NUM_MINIBATCHES", type=int, default=16, help="minibatches per epoch"
    )
    dqn_parser.add_argument(
        "--NUM_EPOCHS", type=int, default=4, help="minibatches per epoch"
    )
    dqn_parser.add_argument(
        "--NORM_INPUT", type=bool, default=False, help="Normalize input using LayerNorm"
    )
    dqn_parser.add_argument("--HIDDEN_SIZE", type=int, default=256, help="Hidden size")
    dqn_parser.add_argument(
        "--NUM_LAYERS", type=int, default=2, help="Number of layers"
    )
    dqn_parser.add_argument(
        "--NORM_TYPE", type=str, default="layer_norm", help="Normalization type"
    )
    dqn_parser.add_argument("--LR", type=float, default=0.00005, help="Learning rate")
    dqn_parser.add_argument(
        "--MAX_GRAD_NORM", type=float, default=0.5, help="Max gradient norm"
    )
    dqn_parser.add_argument(
        "--LR_LINEAR_DECAY", type=bool, default=True, help="Linear decay learning rate"
    )
    dqn_parser.add_argument("--REW_SCALE", type=float, default=1, help="Reward scale")
    dqn_parser.add_argument(
        "--GAMMA", type=float, default=0.99, help="Discount factor for rewards"
    )
    dqn_parser.add_argument("--LAMBDA", type=float, default=0.0, help="Lambda")
    dqn_parser.add_argument(
        "--HYP_TUNE", type=bool, default=False, help="Hyperparameter tuning"
    )
    dqn_parser.add_argument("--ENTITY", type=str, default="", help="Entity name")
    dqn_parser.add_argument(
        "--PROJECT", type=str, default="NavigatorEasy", help="WanDB Project name"
    )
    dqn_parser.add_argument(
        "--WANDB_MODE", type=str, default="online", help="WanDB mode"
    )
    dqn_parser.add_argument("--SEED", type=int, default=0, help="Random seed")
    dqn_parser.add_argument(
        "--NUM_SEEDS", type=int, default=1, help="Number of Random seeds"
    )
    dqn_parser.add_argument(
        "--PARTIAL", action="store_true", help="Partial Observations"
    )
    dqn_parser.add_argument(
        "--ENV_NAME", type=str, default="BattleShipEasy", help="Environment name"
    )
    dqn_parser.add_argument(
        "--ENV_KWARGS", type=dict, default={}, help="Environment kwargs"
    )
    dqn_parser.add_argument(
        "--TEST_DURING_TRAINING", type=bool, default=False, help="Test during training"
    )
    dqn_parser.add_argument(
        "--TEST_INTERVAL", type=float, default=0.05, help="In terms of total updatesl"
    )
    dqn_parser.add_argument(
        "--TEST_NUM_ENVS", type=int, default=128, help="Number of test environments"
    )
    dqn_parser.add_argument(
        "--EPS_TEST", type=float, default=0, help="0 for greedy policy"
    )
    dqn_parser.add_argument(
        "--ALG_NAME", type=str, default="DQN", help="Algorithm name"
    )
    dqn_parser.add_argument(
        "--OBS_SIZE", type=int, default=128, help="Observation size"
    )


    # DQN_RNN
    dqn_rnn_parser = subparsers.add_parser("DQN_RNN", help="Training with DQN_RNN")
    dqn_rnn_parser.add_argument(
        "--MEMORY_TYPE", type=str, default="MLP", help="Memory model type."
    )
    dqn_rnn_parser.add_argument(
        "--TOTAL_TIMESTEPS", type=int, default=2e7, help="Total timesteps"
    )
    dqn_rnn_parser.add_argument(
        "--TOTAL_TIMESTEPS_DECAY",
        type=int,
        default=2e6,
        help="Total timesteps decay will be used for decay functions, in case you want to test for less timesteps and keep decays same.",
    )
    dqn_rnn_parser.add_argument(
        "--NUM_ENVS", type=int, default=16, help="Parallel Environments"
    )
    dqn_rnn_parser.add_argument(
        "--MEMORY_WINDOW",
        type=int,
        default=4,
        help="steps of previous episode added in the rnn training horizon",
    )
    dqn_rnn_parser.add_argument(
        "--NUM_STEPS",
        type=int,
        default=128,
        help="steps per environment in each update",
    )
    dqn_rnn_parser.add_argument(
        "--EPS_START", type=float, default=1, help="Epsilon start"
    )
    dqn_rnn_parser.add_argument(
        "--EPS_FINISH", type=float, default=0.05, help="Epsilon finish"
    )
    dqn_rnn_parser.add_argument(
        "--EPS_DECAY", type=float, default=0.25, help="Epsilon decay"
    )
    dqn_rnn_parser.add_argument(
        "--NUM_MINIBATCHES", type=int, default=16, help="minibatches per epoch"
    )
    dqn_rnn_parser.add_argument(
        "--NUM_EPOCHS", type=int, default=4, help="minibatches per epoch"
    )
    dqn_rnn_parser.add_argument(
        "--NORM_INPUT", type=bool, default=False, help="Normalize input using LayerNorm"
    )
    dqn_rnn_parser.add_argument(
        "--HIDDEN_SIZE", type=int, default=256, help="Hidden size"
    )
    dqn_rnn_parser.add_argument(
        "--NUM_LAYERS", type=int, default=2, help="Number of layers"
    )
    dqn_rnn_parser.add_argument(
        "--NORM_TYPE", type=str, default="layer_norm", help="Normalization type"
    )
    dqn_rnn_parser.add_argument(
        "--LR", type=float, default=0.00005, help="Learning rate"
    )
    dqn_rnn_parser.add_argument(
        "--MAX_GRAD_NORM", type=float, default=0.5, help="Max gradient norm"
    )
    dqn_rnn_parser.add_argument(
        "--LR_LINEAR_DECAY", type=bool, default=True, help="Linear decay learning rate"
    )
    dqn_rnn_parser.add_argument(
        "--REW_SCALE", type=float, default=1, help="Reward scale"
    )
    dqn_rnn_parser.add_argument(
        "--GAMMA", type=float, default=0.99, help="Discount factor for rewards"
    )
    dqn_rnn_parser.add_argument("--LAMBDA", type=float, default=0.0, help="Lambda")
    dqn_rnn_parser.add_argument(
        "--HYP_TUNE", type=bool, default=False, help="Hyperparameter tuning"
    )
    dqn_rnn_parser.add_argument("--ENTITY", type=str, default="", help="Entity name")
    dqn_rnn_parser.add_argument(
        "--PROJECT", type=str, default="NavigatorEasy", help="WanDB Project name"
    )
    dqn_rnn_parser.add_argument(
        "--WANDB_MODE", type=str, default="online", help="WanDB mode"
    )
    dqn_rnn_parser.add_argument("--SEED", type=int, default=0, help="Random seed")
    dqn_rnn_parser.add_argument(
        "--NUM_SEEDS", type=int, default=1, help="Number of Random seeds"
    )
    dqn_rnn_parser.add_argument(
        "--PARTIAL", action="store_true", help="Partial Observations"
    )
    dqn_rnn_parser.add_argument(
        "--ENV_NAME", type=str, default="BattleShipEasy", help="Environment name"
    )
    dqn_rnn_parser.add_argument(
        "--ENV_KWARGS", type=dict, default={}, help="Environment kwargs"
    )
    dqn_rnn_parser.add_argument(
        "--TEST_DURING_TRAINING", type=bool, default=False, help="Test during training"
    )
    dqn_rnn_parser.add_argument(
        "--TEST_INTERVAL", type=float, default=0.05, help="In terms of total updatesl"
    )
    dqn_rnn_parser.add_argument(
        "--TEST_NUM_ENVS", type=int, default=128, help="Number of test environments"
    )
    dqn_rnn_parser.add_argument(
        "--EPS_TEST", type=float, default=0, help="0 for greedy policy"
    )
    dqn_rnn_parser.add_argument(
        "--ALG_NAME", type=str, default="DQN_RNN", help="Algorithm name"
    )
    dqn_rnn_parser.add_argument(
        "--OBS_SIZE", type=int, default=128, help="Observation size"
    )
    return parser.parse_args()


def get_local_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/cartpole/ppo_cartpole.json"
    )
    return parser.parse_args()


def main():
    args = get_args()
    args_dict = vars(args)

    if args.TRAIN_TYPE == "PPO":
        from popgym_arcade.baselines.ppo import ppo_run
        ppo_run(args_dict)
    elif args.TRAIN_TYPE == "PPO_RNN":
        from popgym_arcade.baselines.ppo_rnn import ppo_rnn_run
        ppo_rnn_run(args_dict)
    elif args.TRAIN_TYPE == "PQN":
        from popgym_arcade.baselines.pqn import pqn_run
        pqn_run(args_dict)
    elif args.TRAIN_TYPE == "PQN_RNN":
        from popgym_arcade.baselines.pqn_rnn import pqn_rnn_run
        pqn_rnn_run(args_dict)
    elif args.TRAIN_TYPE == "DQN_RNN":
        from popgym_arcade.baselines.dqn_rnn import dqn_rnn_run
    elif args.TRAIN_TYPE == "DQN":
        from popgym_arcade.baselines.dqn import dqn_run
        dqn_run(args_dict)


if __name__ == "__main__":
    main()
