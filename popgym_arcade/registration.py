from popgym_arcade.environments import (
    AutoEncodeEasy,
    AutoEncodeHard,
    AutoEncodeMedium,
    BattleShipEasy,
    BattleShipHard,
    BattleShipMedium,
    CartPoleEasy,
    CartPoleHard,
    CartPoleMedium,
    CountRecallEasy,
    CountRecallHard,
    CountRecallMedium,
    MineSweeperEasy,
    MineSweeperHard,
    MineSweeperMedium,
    NavigatorEasy,
    NavigatorHard,
    NavigatorMedium,
    NoisyCartPoleEasy,
    NoisyCartPoleHard,
    NoisyCartPoleMedium,
    SkittlesEasy,
    SkittlesHard,
    SkittlesMedium,
    BreakoutEasy,
    BreakoutMedium,
    BreakoutHard,
    TetrisEasy,
    TetrisMedium,
    TetrisHard,
)


def make(env_id: str, **env_kwargs):
    if env_id == "CartPoleEasy":
        env = CartPoleEasy(**env_kwargs)
    elif env_id == "CartPoleMedium":
        env = CartPoleMedium(**env_kwargs)
    elif env_id == "CartPoleHard":
        env = CartPoleHard(**env_kwargs)
    elif env_id == "NoisyCartPoleEasy":
        env = NoisyCartPoleEasy(**env_kwargs)
    elif env_id == "NoisyCartPoleMedium":
        env = NoisyCartPoleMedium(**env_kwargs)
    elif env_id == "NoisyCartPoleHard":
        env = NoisyCartPoleHard(**env_kwargs)
    elif env_id == "CountRecallEasy":
        env = CountRecallEasy(**env_kwargs)
    elif env_id == "CountRecallMedium":
        env = CountRecallMedium(**env_kwargs)
    elif env_id == "CountRecallHard":
        env = CountRecallHard(**env_kwargs)
    elif env_id == "BattleShipEasy":
        env = BattleShipEasy(**env_kwargs)
    elif env_id == "BattleShipMedium":
        env = BattleShipMedium(**env_kwargs)
    elif env_id == "BattleShipHard":
        env = BattleShipHard(**env_kwargs)
    elif env_id == "MineSweeperEasy":
        env = MineSweeperEasy(**env_kwargs)
    elif env_id == "MineSweeperMedium":
        env = MineSweeperMedium(**env_kwargs)
    elif env_id == "MineSweeperHard":
        env = MineSweeperHard(**env_kwargs)
    elif env_id == "AutoEncodeEasy":
        env = AutoEncodeEasy(**env_kwargs)
    elif env_id == "AutoEncodeMedium":
        env = AutoEncodeMedium(**env_kwargs)
    elif env_id == "AutoEncodeHard":
        env = AutoEncodeHard(**env_kwargs)
    elif env_id == "NavigatorEasy":
        env = NavigatorEasy(**env_kwargs)
    elif env_id == "NavigatorMedium":
        env = NavigatorMedium(**env_kwargs)
    elif env_id == "NavigatorHard":
        env = NavigatorHard(**env_kwargs)
    elif env_id == "SkittlesEasy":
        env = SkittlesEasy(**env_kwargs)
    elif env_id == "SkittlesMedium":
        env = SkittlesMedium(**env_kwargs)
    elif env_id == "SkittlesHard":
        env = SkittlesHard(**env_kwargs)
    elif env_id == "BreakoutEasy":
        env = BreakoutEasy(**env_kwargs)
    elif env_id == "BreakoutMedium":
        env = BreakoutMedium(**env_kwargs)
    elif env_id == "BreakoutHard":
        env = BreakoutHard(**env_kwargs)
    elif env_id == "TetrisEasy":
        env = TetrisEasy(**env_kwargs)
    elif env_id == "TetrisMedium":
        env = TetrisMedium(**env_kwargs)
    elif env_id == "TetrisHard":
        env = TetrisHard(**env_kwargs)
    else:
        raise ValueError("Environment ID is not registered")

    return env, env.default_params


REGISTERED_ENVIRONMENTS = [
    "CartPoleEasy",
    "CartPoleMedium",
    "CartPoleHard",
    "NoisyCartPoleEasy",
    "NoisyCartPoleMedium",
    "NoisyCartPoleHard",
    "CountRecallEasy",
    "CountRecallMedium",
    "CountRecallHard",
    "BattleShipEasy",
    "BattleShipMedium",
    "BattleShipHard",
    "AutoEncodeEasy",
    "AutoEncodeMedium",
    "AutoEncodeHard",
    "NavigatorEasy",
    "NavigatorMedium",
    "NavigatorHard",
    "MineSweeperEasy",
    "MineSweeperMedium",
    "MineSweeperHard",
    "SkittlesEasy",
    "SkittlesMedium",
    "SkittlesHard",
    "BreakoutEasy",
    "BreakoutMedium",
    "BreakoutHard",
    "TetrisEasy",
    "TetrisMedium",
    "TetrisHard"
]
