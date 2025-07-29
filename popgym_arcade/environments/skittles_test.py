# 导入 functools 模块，它提供了高阶函数和可调用对象的操作。
# 在这里，它主要用于 jax.jit 的 partial 应用，可以将函数的某些参数预先固定。
import functools
# 导入 typing 模块，用于提供类型提示，增强代码的可读性和健壮性。
# Any: 表示任何类型。
# Dict: 表示字典类型。
# Optional: 表示一个可选类型，可以是指定类型或 None。
# Tuple: 表示元组类型。
# Union: 表示联合类型，可以是多种指定类型之一。
from typing import Any, Dict, Optional, Tuple, Union

# 导入 chex 模块，这是一个由 DeepMind 开发的库，用于在 JAX 中编写可靠、可复用的代码。
# 它提供了额外的断言、测试工具和与JAX兼容的数据类。
import chex
# 导入 jax 模块，这是一个用于高性能机器学习研究的库，核心是 NumPy 和自动微分的结合。
import jax
# 导入 jax.numpy，它是 JAX 对 NumPy API 的实现。
# 它的接口和 numpy 几乎一样，但它支持在 GPU/TPU 上运行、自动微分和即时(JIT)编译。
import jax.numpy as jnp
# 从 chex 导入 dataclass，它是一个对 Python 原生 dataclass 的封装，使其与 JAX 的转换（如 jit, vmap）兼容。
from chex import dataclass
# 从 gymnax.environments 导入 environment 和 spaces。
# gymnax 是一个用 JAX 实现的 Gym API，用于创建可 JIT 编译的强化学习环境。
# environment: 定义了 RL 环境的基本接口。
# spaces: 定义了动作空间和观察空间的数据结构（如 Discrete, Box）。
from gymnax.environments import environment, spaces
# 从 jax 导入 lax 模块。
# lax (Linear Algebra Xpress) 是 JAX 的底层操作库，提供了许多高效的原始操作，如控制流（cond, fori_loop）。
from jax import lax

# 从项目内部的 draw_utils 模块导入一系列绘图函数。
# 这些函数用于在 JAX 环境中绘制图形元素，如网格、数字、矩形等，最终生成环境的视觉观察。
from popgym_arcade.environments.draw_utils import (
    draw_grid,
    draw_number,
    draw_rectangle,
    draw_str,
    draw_sub_canvas,
)


# 使用 chex.dataclass 定义环境状态类 EnvState。
# dataclass 会自动生成 __init__, __repr__ 等方法。
# frozen=True 使其实例不可变，这对于在 JAX 中进行函数式编程非常重要，可以避免意外的副作用。
@dataclass(frozen=True)
class EnvState(environment.EnvState):
    """
    环境状态的数据类。

    Attributes:
        matrix_state (chex.Array): 存储游戏棋盘状态的二维数组，0 代表空格，1 代表敌人。
        color_indexes (chex.Array): 存储每一行敌人颜色的索引，用于实现彩虹效果。
        x (int): 智能体当前在最后一行的列坐标。
        xp (int): 智能体当前位置下方是否有敌人（1 表示有，0 表示没有），用于碰撞检测。
        over (int): 标志位，记录智能体是否在之前的步骤中被击中。
        time (int): 当前 episode 已经进行的时间步数。
        score (int): 当前得分，通常等于存活的时间步数。
    """
    matrix_state: chex.Array  # 游戏棋盘，一个二维数组
    color_indexes: chex.Array # 每一行敌人的颜色索引
    x: int                    # 智能体的 x 坐标（列）
    xp: int                   # 智能体当前位置的棋盘值（用于检测碰撞）
    over: int                 # 游戏是否结束的标志
    time: int                 # 当前时间步
    score: int                # 当前得分


# 使用 chex.dataclass 定义环境参数类 EnvParams。
# 这个类用于存储环境的配置参数，这些参数在整个 episode 中通常保持不变。
# 在这个特定的环境中，目前没有定义额外的参数，但保留了这个结构以便未来扩展。
@dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    pass


# 定义 Skittles 环境主类，继承自 gymnax 的 Environment 基类。
# 这个类封装了游戏的所有逻辑，包括状态转换、奖励计算、终止判断和渲染。
class Skittles(environment.Environment[EnvState, EnvParams]):
    """
    Jax 可编译的 Skittles 环境。

    ### 描述
    在 Skittles 游戏中，智能体的任务是躲避从屏幕顶部落下的敌人。
    智能体可以向左或向右移动来躲避敌人。目标是尽可能长时间地存活而不被敌人击中。
    游戏有三种难度：简单、中等和困难。每种难度的网格大小和最大步数都不同。
    - 简单：8x8 网格，目标存活 200 步，每行有 1 个敌人。
    - 中等：10x10 网格，目标存活 400 步，每行有 2 个敌人。
    - 困难：12x12 网格，目标存活 600 步，每行有 2 个敌人。
    当智能体被敌人击中或达到最大步数时，episode 结束。

    ### 棋盘元素
    - 0: 空格
    - 1: 敌人
    玩家只能在棋盘的最后一行移动，其位置由列索引表示。

    ### 动作空间
    | 动作 | 描述                         |
    |--------|-------------------------------------|
    | 0      | 上 (无操作)                      |
    | 1      | 下 (无操作)                      |
    | 2      | 左                                |
    | 3      | 右                               |
    | 4      | 开火 (无操作)                    |

    ### 观察空间
    OBS_SIZE 可以是 128 或 256。观察是一个渲染后的图像，形状为 (OBS_SIZE, OBS_SIZE, 3)。
    图像包含：
        - 智能体的当前位置（在最后一行，白色）。
        - 从屏幕顶部掉落的敌人（彩虹色）。
        - 网格线（白色）。
        - 分数（绿色）。
        - 环境名称（黄色）。

    ### 奖励
    - 奖励范围：1.0 / 每个 episode 的最大步数。每存活一步，获得一个小的固定奖励。

    ### 终止与截断
    当智能体被敌人击中或达到最大步数时，episode 结束。

    ### 参数
    - max_steps_in_episode: 每个 episode 的最大步数。
    - grid_size: 网格的大小（行数和列数）。
    - obs_size: 观察空间的大小，可选 128 或 256。
    - partial_obs: 是否使用部分观察。
    - enemy_num: 该难度级别下的敌人数量。

    """

    # render_common 是一个字典，存储了渲染时通用的颜色和参数。
    # 这种方式将配置和代码逻辑分离，方便修改。
    render_common = {
        # parameters for rendering (256, 256, 3) canvas
        # 定义画布背景色 (黑色)
        "clr": jnp.array([0, 0, 0], dtype=jnp.uint8),
        # parameters for rendering sub canvas
        # 定义子画布背景色 (黑色)
        "sub_clr": jnp.array([0, 0, 0], dtype=jnp.uint8),
        # parameters for current action position
        # 定义智能体颜色 (白色)
        "action_clr": jnp.array([255, 255, 255], dtype=jnp.uint8),
        # parameters for rendering enemy
        # 定义彩虹色，用于渲染不同行的敌人
        "red": jnp.array([255, 0, 0], dtype=jnp.uint8),
        "orange": jnp.array([255, 128, 0], dtype=jnp.uint8),
        "yellow": jnp.array([255, 255, 0], dtype=jnp.uint8),
        "green": jnp.array([0, 255, 0], dtype=jnp.uint8),
        "blue": jnp.array([0, 0, 255], dtype=jnp.uint8),
        "indigo": jnp.array([74, 214, 247], dtype=jnp.uint8),
        "violet": jnp.array([125, 0, 235], dtype=jnp.uint8),
        # parameters for rendering grids
        # 定义网格线颜色 (白色)
        "grid_clr": jnp.array([255, 255, 255], dtype=jnp.uint8),
        # parameters for rendering score
        # 定义分数文本颜色 (绿色)
        "sc_clr": jnp.array([0, 255, 128], dtype=jnp.uint8),
        # parameters for rendering env name
        # 定义环境名称文本颜色 (黄色)
        "env_clr": jnp.array([255, 245, 0], dtype=jnp.uint8),
    }
    # render_256x 字典存储了渲染 256x256 画布时的特定参数。
    # 它通过 `**render_common` 语法继承了通用参数，并添加或覆盖了特定尺寸的参数。
    render_256x = {
        **render_common,
        # parameters for rendering (256, 256, 3) canvas
        "size": 256,
        # 子画布大小，根据棋盘大小(8, 10, 12)有不同取值
        "sub_size": {
            8: 186,
            10: 192,
            12: 182,
        },
        # parameters for rendering grids
        "grid_px": 2, # 网格线宽度
        # parameters for rendering score
        "sc_t_l": (86, 2),    # 分数显示的左上角坐标
        "sc_b_r": (171, 30),  # 分数显示的右下角坐标
        # parameters for rendering env name
        "env_t_l": (0, 231),  # 环境名称显示的左上角坐标
        "env_b_r": (256, 256),# 环境名称显示的右下角坐标
    }

    # render_128x 字典存储了渲染 128x128 画布时的特定参数。
    render_128x = {
        **render_common,
        # parameters for rendering (128, 128, 3) canvas
        "size": 128,
        # 子画布大小
        "sub_size": {
            8: 90,
            10: 92,
            12: 98,
        },
        # parameters for rendering grids
        "grid_px": 2, # 网格线宽度
        # parameters for rendering score
        "sc_t_l": (43, 1),   # 分数显示的左上角坐标
        "sc_b_r": (85, 15),  # 分数显示的右下角坐标
        # parameters for rendering env name
        "env_t_l": (0, 115), # 环境名称显示的左上角坐标
        "env_b_r": (128, 128),# 环境名称显示的右下角坐标
    }
    # render_mode 将观察尺寸映射到对应的渲染配置字典。
    # 这样在渲染时可以根据 `obs_size` 动态选择配置。
    render_mode = {
        256: render_256x,
        128: render_128x,
    }

    # 环境的构造函数 __init__。
    # 它在创建环境实例时被调用，用于初始化环境的各种参数。
    def __init__(
        self,
        max_steps_in_episode: int,  # 一个 episode 的最大步数
        grid_size: int,             # 棋盘大小
        obs_size: int = 128,        # 观察图像的大小，默认为 128
        partial_obs=False,          # 是否为部分可观察，默认为 False
        enemy_num: int = 2,         # 每行生成的敌人数量，默认为 2
    ):
        # 调用父类 Environment 的构造函数。
        super().__init__()
        # 将传入的参数保存为实例的属性。
        self.obs_size = obs_size
        self.max_steps_in_episode = max_steps_in_episode
        # 计算每一步的奖励值，使总奖励为1。
        self.reward_scale = 1.0 / max_steps_in_episode
        self.grid_size = grid_size
        self.partial_obs = partial_obs
        self.enemy_num = enemy_num

    # `default_params` 是一个属性方法（property）。
    # 它返回环境的默认参数实例。因为 `EnvParams` 是空的，所以直接返回一个实例。
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    # `step_env` 是环境的核心方法之一，执行一个时间步的逻辑。
    # 它接收当前状态、动作和参数，并返回新的观察、新状态、奖励、完成标志和额外信息。
    def step_env(
        self,
        key: chex.PRNGKey,  # JAX 的随机数生成器 key
        state: EnvState,    # 当前环境状态
        action: int,        # 智能体执行的动作
        params: EnvParams,  # 环境参数
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """在环境中执行一个步骤。"""
        # JAX 的随机数生成是显式的。每次使用后，都需要分割 key 以得到新的 key 供后续使用。
        # 这种设计保证了随机数生成的可复现性。
        key, newkey = jax.random.split(key)
        # 从状态中获取智能体上一位置的棋盘值和游戏结束标志。
        xp = state.xp
        over = state.over
        x = state.x
        # 根据动作更新智能体的位置 `x`。
        # `action == 2` 是向左，`x - 1`。
        # `action == 3` 是向右，`x + 1`。
        # `jnp.where(condition, a, b)` 是 JAX 的条件选择，类似三元运算符。
        # `jnp.clip` 用于将 `x` 的值限制在 `[0, self.grid_size - 1]` 的有效范围内。
        x = jnp.clip(jnp.where(action == 2, x - 1, x), 0, self.grid_size - 1)
        x = jnp.clip(jnp.where(action == 3, x + 1, x), 0, self.grid_size - 1)

        # 获取当前的棋盘状态。
        matrix_state = state.matrix_state
        # 检查智能体移动前的位置是否安全（即是否已经在一个有敌人的格子上）。
        # 这用于判断游戏是否因之前的状态而结束。
        xp = matrix_state[self.grid_size - 1, x]
        over = xp
        # 更新棋盘状态，让所有敌人向下移动一行。
        # 通过数组切片实现：将第 0 到 grid_size-2 行的内容，设置到第 1 到 grid_size-1 行。
        # `.at[...].set(...)` 是 JAX 中用于更新数组特定部分（而不修改原数组）的推荐方法，符合函数式编程范式。
        matrix_state = matrix_state.at[1 : self.grid_size, :].set(
            matrix_state[0 : self.grid_size - 1, :]
        )

        # 分割随机 key，用于生成新的一行敌人。
        newkey, enemy_key_test = jax.random.split(newkey)



        def remove_invincible_enemy(ok_key):
            matrix_twin = matrix_state
            enemy_tmp = self.random_enemy(ok_key)
            matrix_twin = matrix_state.at[0, :].set(enemy_tmp)
            building_block = jnp.ones((self.grid_size, 2), dtype = matrix_twin.dtype)
            matrix_fat = jnp.concatenate([building_block, matrix_twin, building_block], axis=1)
            x_twin = x
            test_value = 0
            judge1 = matrix_fat[self.grid_size-1, x_twin-2:x_twin+2]
            judge2 = matrix_fat[self.grid_size-2, x_twin-2:x_twin+2]
            chief = jnp.logical_or(judge1, judge2)
            def x_change(i, test_value):
                x_twin = jnp.where(chief == jnp.array([0, 0, 0, 1, 1]), x_twin - 1, x_twin)
                x_twin = jnp.where(chief == jnp.array([0, 0, 0, 1, 0]), x_twin - 1, x_twin)
                x_twin = jnp.where(chief == jnp.array([0, 0, 0, 1, 1]), x_twin - 1, x_twin)
                x_twin = jnp.where(chief == jnp.array([1, 0, 0, 1, 1]), x_twin - 1, x_twin)

                
                x_twin = jnp.where(chief == jnp.array([1, 1, 0, 0, 0]), x_twin + 1, x_twin)
                x_twin = jnp.where(chief == jnp.array([0, 1, 0, 0, 0]), x_twin + 1, x_twin)
                x_twin = jnp.where(chief == jnp.array([0, 1, 0, 0, 1]), x_twin + 1, x_twin)
                x_twin = jnp.where(chief == jnp.array([1, 1, 0, 0, 1]), x_twin + 1, x_twin)

                test_value = jnp.where(chief == jnp.array([0, 1, 1, 1, 0]), 1, 0)
                test_value = jnp.where(chief == jnp.array([1, 1, 1, 1, 0]), 1, 0)
                test_value = jnp.where(chief == jnp.array([0, 1, 1, 1, 1]), 1, 0)
                test_value = jnp.where(chief == jnp.array([1, 1, 1, 1, 1]), 1, 0)

                judge1 = matrix_fat[self.grid_size-1, x_twin-2:x_twin+2]
                judge2 = matrix_fat[self.grid_size-2, x_twin-2:x_twin+2]
                chief = jnp.logical_or(judge1, judge2)

                
                
                return chief 
            
            x_final = jax.lax.fori_loop(0, self.grid_size - 2, x_change, chief)

            return 
    

        def ok_game_key(ok_key):
            key,ok_key = jax.random.split(key)
            return ok_key
        

        enemy_key = jax.lax.while_loop(remove_invincible_enemy, ok_game_key, enemy_key_test)
        # 调用 `random_enemy` 方法生成新的一行敌人。
        enemy_new = self.random_enemy(enemy_key)
        # `jnp.squeeze` 去除多余的维度，例如从 (1, N) 变为 (N,)。
        enemy_new = jnp.squeeze(enemy_new)


        
        # 将新生成的敌人行设置到棋盘的第一行。
        matrix_state = matrix_state.at[0, :].set(enemy_new)
        # 更新 `xp`，检查智能体当前新位置下方是否有敌人，用于即时碰撞检测。
        xp = matrix_state[self.grid_size - 1, x]

        # 更新颜色索引，实现敌人下落时的彩虹效果。
        # 每个时间步，第一行的颜色索引加一，并对 7 取模，实现循环。
        new_color_idx = (state.color_indexes[0] + 1) % 7

        # 将整个颜色索引数组向下滚动，旧的颜色跟着行一起下移。
        new_color_indexes = jnp.roll(state.color_indexes, shift=1)
        # 将新的颜色索引设置到第一行的位置。
        new_color_indexes = new_color_indexes.at[0].set(new_color_idx)
        # 创建新的状态对象 `EnvState`。
        # JAX 是函数式的，我们不修改旧的 `state`，而是返回一个新的 `state` 实例。
        state = EnvState(
            matrix_state=matrix_state,
            x=x,
            xp=matrix_state[self.grid_size - 1, x],  # 新的碰撞检测值
            over=over,  # 保留上一步的碰撞状态
            time=state.time + 1,  # 时间步加一
            score=state.score + 1,  # 分数加一
            color_indexes=new_color_indexes,
        )

        # 检查 episode 是否终止。
        done = self.is_terminal(state, params)
        # 构造一个包含额外信息的字典 `infos`。
        # `terminated`: 因碰撞而结束。
        # `truncated`: 因达到最大步数而结束。
        # `discount`: 折扣因子。
        infos = {
            "terminated": state.xp + state.over,  # xp 或 over 大于0都意味着碰撞
            "truncated": state.time >= self.max_steps_in_episode,
            "discount": self.discount(state, params),
        }
        # `lax.stop_gradient` 阻止梯度流过观察和状态，因为它们通常不参与梯度计算。
        # 返回 (观察, 状态, 奖励, 终止标志, 信息)。
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            jnp.array(self.reward_scale),
            done,
            infos,
        )

    # `reset_env` 方法用于在每个 episode 开始时重置环境。
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """将环境重置为初始状态。"""
        # 分割随机 key，用于初始化智能体位置。
        key, subkey1 = jax.random.split(key)
        # 初始化一个全为 0 的棋盘。
        matrix_state = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        # 在最后一行的有效范围内，随机生成智能体的初始位置 `x`。
        x = jax.random.randint(
            subkey1, shape=(), minval=0, maxval=self.grid_size, dtype=jnp.int32
        )

        # 创建初始的 `EnvState` 对象。
        state = EnvState(
            matrix_state=matrix_state,
            # 初始化颜色索引数组，全为0。
            color_indexes=jnp.zeros(self.grid_size).at[0].set(0),
            x=x,
            # 初始时，智能体位置没有敌人。
            xp=matrix_state[self.grid_size - 1, x],
            time=0,  # 时间步为 0
            score=0,  # 分数为 0
            over=0,  # 游戏未结束
        )
        # 返回初始观察和初始状态。
        return self.get_obs(state), state
    
    # def remove_invincible_enemy(self, state: EnvState, params: EnvParams):
    #     invincible_value = jnp.array(1)
    
    
    # `random_enemy` 方法用于生成新的一行敌人。
    def random_enemy(self, key) -> jnp.ndarray:
        """生成一个随机的敌人行。"""
        # 分割随机 key。
        key, subkey2 = jax.random.split(key)
        # 创建一个全为 0 的行向量。
        enemy_row = jnp.zeros(self.grid_size, dtype=jnp.int32)
        # 从 `[0, grid_size-1]` 中不重复地选择 `enemy_num` 个索引。
        indices = jax.random.choice(
            subkey2, jnp.arange(self.grid_size), shape=(self.enemy_num,), replace=False
        )
        # 在选中的索引位置上，将值设为 1，表示敌人。
        enemy_row = enemy_row.at[indices].set(1)
        # 将行向量 reshape 成 (1, grid_size) 的二维数组，方便后续操作。
        enemy_row = enemy_row.reshape(1, -1)
        return enemy_row

    # `get_obs` 方法根据当前状态生成观察。
    # 在这个环境中，观察是渲染后的图像。
    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        # 直接调用 render 方法来获取图像作为观察。
        return self.render(state)
        # 也可以直接返回棋盘状态矩阵作为观察（如果不需要图像输入）。
        # return state.matrix_state

    # `is_terminal` 方法判断 episode 是否结束。
    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """检查 episode 是否结束。"""
        # 结束条件1：发生碰撞。`state.xp` 是当前位置碰撞，`state.over` 是上一步位置碰撞。
        # 任何一个大于0都表示碰撞。
        done_crash = state.xp + state.over
        # 结束条件2：达到最大步数。
        done_steps = state.time >= self.max_steps_in_episode
        # 两个条件满足任意一个，`done` 就为 True。`jnp.logical_or` 用于执行逻辑或操作。
        done = jnp.logical_or(done_crash, done_steps)
        return done

    # `render` 方法将环境状态可视化为一张图像。
    # 使用 `@functools.partial(jax.jit, static_argnums=(0,))` 对此方法进行 JIT 编译。
    # `static_argnums=(0,)` 告诉 JIT 编译器，第一个参数 `self` 是静态的，
    # 如果 `self` 的属性发生变化（例如 `grid_size`），JAX 会重新编译此函数。
    @functools.partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnvState) -> chex.Array:
        """渲染环境的当前状态。"""
        # 根据观察尺寸 `self.obs_size` 从 `render_mode` 中选择渲染配置。
        render_config = self.render_mode[self.obs_size]
        board_size = self.grid_size

        # 从配置中获取网格线宽度、子画布大小和每个格子的像素大小。
        grid_px = render_config["grid_px"]
        sub_size = render_config["sub_size"][board_size]
        square_size = (sub_size - (board_size + 1) * grid_px) // board_size

        # 使用 `jnp.meshgrid` 生成网格中每个单元格的坐标。这比循环更高效。
        x_coords, y_coords = jnp.arange(board_size), jnp.arange(board_size)
        xx, yy = jnp.meshgrid(x_coords, y_coords, indexing="ij")
        # 计算每个单元格左上角的像素坐标。
        # top_left_x = grid_px + xx * (square_size + grid_px)
        # top_left_y = grid_px + yy * (square_size + grid_px)
        top_left_x = grid_px + yy * (square_size + grid_px)
        top_left_y = grid_px + xx * (square_size + grid_px)
        # 将 x, y 坐标堆叠成一个形状为 (board_size, board_size, 2) 的数组。
        all_top_left = jnp.stack([top_left_x, top_left_y], axis=-1)
        # 计算每个单元格右下角的坐标。
        all_bottom_right = all_top_left + square_size

        # 初始化主画布和子画布（用于绘制游戏棋盘）。
        # `jnp.full` 创建一个用指定颜色填充的数组。
        canvas = jnp.full(
            (render_config["size"],) * 2 + (3,), render_config["clr"], dtype=jnp.uint8
        )
        sub_canvas = jnp.full(
            (sub_size, sub_size, 3), render_config["sub_clr"], dtype=jnp.uint8
        )

        # 获取智能体的位置。
        action_x, action_y = board_size - 1, state.x

        # 将二维棋盘状态展平成一维，方便 `jax.vmap` 或 `fori_loop` 处理。
        board_flat = state.matrix_state.flatten()

        # 定义一个内部函数 `render_cell`，用于渲染单个单元格。
        def render_cell(pos, canvas):
            # 从一维索引 `pos` 计算出二维坐标 `x, y`。
            x = pos // board_size
            y = pos % board_size
            # 获取该单元格的左上角和右下角坐标。
            tl = all_top_left[x, y]
            br = all_bottom_right[x, y]
            # 获取单元格的值 (0 或 1)。
            cell_val = board_flat[pos]

            # 定义彩虹颜色列表。
            rainbow_colors = jnp.array(
                [
                    render_config["red"],
                    render_config["orange"],
                    render_config["yellow"],
                    render_config["green"],
                    render_config["blue"],
                    render_config["indigo"],
                    render_config["violet"],
                ]
            )
            # 根据行号和颜色索引，获取当前敌人的颜色。
            color_idx = jnp.int32(state.color_indexes[x])
            enemy_color = rainbow_colors[color_idx % len(rainbow_colors)]
            # 使用 `lax.cond`，这是一个可 JIT 编译的 if/else 语句。
            # 如果 `cell_val` 为 1 (敌人)，则调用 `draw_rectangle` 绘制敌人方块。
            # 否则，返回原始画布。
            canvas = lax.cond(
                cell_val == 1,
                lambda: draw_rectangle(tl, br, enemy_color, canvas),
                lambda: canvas,
            )

            return canvas

        # 定义部分渲染逻辑：只渲染智能体所在的单元格和第一行敌人。
        # 这在 `partial_obs=True` 时使用，可以加快渲染速度。
        def _render_partial(sub_canvas):
            pos = action_x * board_size + action_y
            sub_canvas = render_cell(pos, sub_canvas)

            # 获取第一行的所有索引。
            first_row_indices = jnp.arange(board_size)

            # 定义一个用于循环体的函数。
            def render_first_row_cell(idx, canvas):
                return render_cell(idx, canvas)

            # 使用 `jax.lax.fori_loop` 循环渲染第一行的每个单元格。
            # fori_loop 是可 JIT 编译的 for 循环，比 Python 原生 for 循环在 JAX 中高效得多。
            sub_canvas = jax.lax.fori_loop(
                0,
                board_size,
                lambda i, c: render_first_row_cell(first_row_indices[i], c),
                sub_canvas,
            )
            return sub_canvas

        # 定义完整渲染逻辑：渲染棋盘上的所有单元格。
        def _render_full(sub_canvas):
            cell_indices = jnp.arange(board_size**2)
            # 使用 `jax.vmap` 将 `render_cell` 函数向量化。
            # `vmap` 可以将一个函数应用于数组的每个元素，实现并行化，效率远高于 for 循环。
            # `in_axes=(0, None)` 表示对第一个参数 `pos` 进行向量化，第二个参数 `canvas` 保持不变。
            updated = jax.vmap(render_cell, in_axes=(0, None))(cell_indices, sub_canvas)
            # `vmap` 会返回一个 (N, H, W, C) 的画布堆栈，使用 `jnp.max` 将它们合并成一张。
            # 因为只有被绘制的像素非零，所以 max 操作可以正确地合并所有绘制结果。
            return jnp.max(updated, axis=0)

        # 条件渲染逻辑：
        # `lax.cond` 用于选择执行哪种渲染模式。
        # - 如果是第 0 步 (time == 0)，总是执行完整渲染，以显示初始棋盘。
        # - 否则，根据 `self.partial_obs` 标志选择部分渲染或完整渲染。
        sub_canvas = lax.cond(
            state.time == 0,
            lambda: _render_full(sub_canvas),
            lambda: lax.cond(
                self.partial_obs,
                lambda: _render_partial(sub_canvas),
                lambda: _render_full(sub_canvas),
            ),
        )

        # 获取智能体位置的坐标，并用指定颜色绘制矩形来表示智能体。
        action_tl = all_top_left[action_x, action_y]
        action_br = all_bottom_right[action_x, action_y]
        sub_canvas = draw_rectangle(
            action_tl, action_br, render_config["action_clr"], sub_canvas
        )

        # 调用 `draw_grid` 在子画布上绘制网格线。
        sub_canvas = draw_grid(
            square_size, grid_px, render_config["grid_clr"], sub_canvas
        )

        # 在主画布上绘制分数。
        canvas = draw_number(
            render_config["sc_t_l"],
            render_config["sc_b_r"],
            render_config["sc_clr"],
            canvas,
            state.score,
        )

        # 在主画布上绘制环境名称。
        canvas = draw_str(
            render_config["env_t_l"],
            render_config["env_b_r"],
            render_config["env_clr"],
            canvas,
            self.name,
        )

        # 将绘制好的子画布（游戏区域）合并到主画布的中央。
        return draw_sub_canvas(sub_canvas, canvas)

    # `name` 属性方法，返回环境的名称。
    @property
    def name(self) -> str:
        return "Skittles"

    # `action_space` 方法定义了智能体可以采取的动作空间。
    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """环境的动作空间。"""
        # 返回一个离散动作空间，包含 5 个动作 (0-4)。
        return spaces.Discrete(5)

    # `observation_space` 方法定义了环境的观察空间。
    def observation_space(self, params: EnvParams) -> spaces.Box:
        """环境的观察空间。"""
        # 返回一个 Box 空间，代表一个图像。
        # 形状为 (obs_size, obs_size, 3)，像素值范围从 0 到 255，数据类型为 uint8。
        return spaces.Box(0, 255, (self.obs_size, self.obs_size, 3), dtype=jnp.uint8)


# 定义简单难度的 Skittles 类。
# 它继承自 Skittles 基类，并通过 `super().__init__` 传入特定于简单难度的参数。
class SkittlesEasy(Skittles):
    def __init__(self, **kwargs):
        # `**kwargs` 语法允许在创建实例时传递任意数量的关键字参数（如此处的 obs_size），
        # 并将它们传递给父类的构造函数。
        super().__init__(max_steps_in_episode=200, grid_size=10, enemy_num=1, **kwargs)


# 定义中等难度的 Skittles 类。
class SkittlesMedium(Skittles):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=400, grid_size=8, enemy_num=1, **kwargs)


# 定义困难难度的 Skittles 类。
class SkittlesHard(Skittles):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=600, grid_size=6, enemy_num=1, **kwargs)
