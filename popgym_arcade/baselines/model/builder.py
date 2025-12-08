from typing import Tuple

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from distreqx import distributions
from jaxtyping import Array, PRNGKeyArray

from memax.equinox.train_utils import get_residual_memory_models


class ActorCritic(eqx.Module):
    action_dim: int = 5
    actor_cnn: nn.Sequential
    actor_trunk: nn.Sequential
    critic_cnn: nn.Sequential
    critic_trunk: nn.Sequential

    def __init__(self, key: PRNGKeyArray, obs_size: int):
        key_array = jax.random.split(key, 14)
        if obs_size == 256:
            self.actor_cnn = nn.Sequential(
                [
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=64,
                        kernel_size=7,
                        stride=2,
                        key=key_array[0],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        key=key_array[1],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        key=key_array[2],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=3,
                        stride=2,
                        key=key_array[3],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )
            self.critic_cnn = nn.Sequential(
                [
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=64,
                        kernel_size=7,
                        stride=2,
                        key=key_array[7],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        key=key_array[8],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        key=key_array[9],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=3,
                        stride=2,
                        key=key_array[10],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )
        else:
            self.actor_cnn = nn.Sequential(
                [
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=64,
                        kernel_size=5,
                        stride=2,
                        key=key_array[0],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        key=key_array[1],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        key=key_array[2],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=3, stride=1),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=1,
                        stride=1,
                        key=key_array[3],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )
            self.critic_cnn = nn.Sequential(
                [
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=64,
                        kernel_size=5,
                        stride=2,
                        key=key_array[0],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        key=key_array[1],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        key=key_array[2],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=3, stride=1),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=1,
                        stride=1,
                        key=key_array[3],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )
        self.actor_trunk = nn.Sequential(
            [
                nn.Linear(in_features=512, out_features=256, key=key_array[4]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(in_features=256, out_features=256, key=key_array[5]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(
                    in_features=256, out_features=self.action_dim, key=key_array[6]
                ),
            ]
        )

        self.critic_trunk = nn.Sequential(
            [
                nn.Linear(in_features=512, out_features=256, key=key_array[11]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(in_features=256, out_features=256, key=key_array[12]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(in_features=256, out_features=1, key=key_array[13]),
            ]
        )

    def __call__(self, x: Array) -> Tuple:
        """Expects image in [0, 255]"""
        x = x.transpose((0, 3, 1, 2)) / 255.0
        actor_embedding = eqx.filter_vmap(self.actor_cnn)(x)
        critic_embedding = eqx.filter_vmap(self.critic_cnn)(x)

        actor_embedding = actor_embedding.reshape(actor_embedding.shape[0], -1)
        critic_embedding = critic_embedding.reshape(critic_embedding.shape[0], -1)

        actor_mean = eqx.filter_vmap(self.actor_trunk)(actor_embedding)
        critic = eqx.filter_vmap(self.critic_trunk)(critic_embedding)
        pi = distributions.Categorical(logits=actor_mean)
        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticRNN(eqx.Module):
    action_dim: int = 5
    actor_cnn: nn.Sequential
    actor_rnn: eqx.Module
    actor_trunk: nn.Sequential
    critic_cnn: nn.Sequential
    critic_rnn: eqx.Module
    critic_trunk: nn.Sequential

    def __init__(self, key: PRNGKeyArray, obs_size: int, rnn_type: str = "lru"):
        key_array = jax.random.split(key, 14)
        if obs_size == 256:
            self.actor_cnn = nn.Sequential(
                [
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=64,
                        kernel_size=7,
                        stride=2,
                        key=key_array[0],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        key=key_array[1],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        key=key_array[2],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=3,
                        stride=2,
                        key=key_array[3],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )
            self.critic_cnn = nn.Sequential(
                [
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=64,
                        kernel_size=7,
                        stride=2,
                        key=key_array[6],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        key=key_array[7],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        key=key_array[8],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=3,
                        stride=2,
                        key=key_array[9],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )
        else:
            self.actor_cnn = nn.Sequential(
                [
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=64,
                        kernel_size=5,
                        stride=2,
                        key=key_array[0],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        key=key_array[1],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        key=key_array[2],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=3, stride=1),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=1,
                        stride=1,
                        key=key_array[3],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )
            self.critic_cnn = nn.Sequential(
                [
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=64,
                        kernel_size=5,
                        stride=2,
                        key=key_array[0],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        key=key_array[1],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        key=key_array[2],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=3, stride=1),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=1,
                        stride=1,
                        key=key_array[3],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )
        self.actor_rnn = get_residual_memory_models(
            input=512,
            hidden=512,
            output=256,
            num_layers=2,
            models=[rnn_type],
            layer_kwargs={
                "Attention": {"window_size": 128},
                "Attention-RoPE": {"window_size": 128},
                "Attention-ALiBi": {"window_size": 128},
            },
            key=key_array[4],
        )[rnn_type]
        self.actor_trunk = nn.Sequential(
            [
                nn.Linear(in_features=256, out_features=256, key=key_array[5]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(
                    in_features=256, out_features=self.action_dim, key=key_array[12]
                ),
            ]
        )

        self.critic_rnn = get_residual_memory_models(
            input=512,
            hidden=512,
            output=256,
            num_layers=2,
            models=[rnn_type],
            layer_kwargs={
                "Attention": {"window_size": 128},
                "Attention-RoPE": {"window_size": 128},
                "Attention-ALiBi": {"window_size": 128},
            },
            key=key_array[10],
        )[rnn_type]
        self.critic_trunk = nn.Sequential(
            [
                nn.Linear(in_features=256, out_features=256, key=key_array[5]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(in_features=256, out_features=1, key=key_array[13]),
            ]
        )

    def __call__(self, actor_state, critic_state, x):
        """Expects image in [0, 255]"""
        inputs, dones = x
        inputs = inputs.transpose((0, 1, 4, 2, 3)) / 255.0
        actor_embedding = eqx.filter_vmap(eqx.filter_vmap(self.actor_cnn))(inputs)
        critic_embedding = eqx.filter_vmap(eqx.filter_vmap(self.critic_cnn))(inputs)

        actor_embedding = actor_embedding.reshape(
            (actor_embedding.shape[0], actor_embedding.shape[1], -1)
        )
        critic_embedding = critic_embedding.reshape(
            (critic_embedding.shape[0], critic_embedding.shape[1], -1)
        )
        actor_rnn_in = (actor_embedding, dones)
        critic_rnn_in = (critic_embedding, dones)
        actor_state, actor_embedding = eqx.filter_vmap(
            self.actor_rnn, in_axes=(0, 1), out_axes=(0, 1)
        )(actor_state, actor_rnn_in)
        actor_state = eqx.filter_vmap(self.actor_rnn.latest_recurrent_state, in_axes=0)(
            actor_state
        )
        critic_state, critic_embedding = eqx.filter_vmap(
            self.critic_rnn, in_axes=(0, 1), out_axes=(0, 1)
        )(critic_state, critic_rnn_in)
        critic_state = eqx.filter_vmap(
            self.critic_rnn.latest_recurrent_state, in_axes=0
        )(critic_state)
        actor_mean = eqx.filter_vmap(eqx.filter_vmap(self.actor_trunk))(actor_embedding)
        pi = distributions.Categorical(logits=actor_mean)
        # pi = distrax.Categorical(logits=actor_mean)
        critic = eqx.filter_vmap(eqx.filter_vmap(self.critic_trunk))(critic_embedding)
        return actor_state, critic_state, pi, jnp.squeeze(critic, axis=-1)

    def initialize_carry(self, key: PRNGKeyArray):
        key_init = jax.random.split(key, 2)
        actor_state = eqx.filter_jit(self.actor_rnn.initialize_carry)(key=key_init[0])
        critic_state = eqx.filter_jit(self.critic_rnn.initialize_carry)(key=key_init[1])
        return actor_state, critic_state


class QNetwork(eqx.Module):
    """CNN + MLP"""

    action_dim: int = 5
    cnn: nn.Sequential
    trunk: nn.Sequential

    def __init__(self, key: PRNGKeyArray, obs_size: int):
        keys = jax.random.split(key, 9)
        if obs_size == 256:
            self.cnn = nn.Sequential(
                [
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=64,
                        kernel_size=7,
                        stride=2,
                        key=keys[0],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        key=keys[1],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        key=keys[2],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=3,
                        stride=2,
                        key=keys[3],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )
        else:
            self.cnn = nn.Sequential(
                [
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=64,
                        kernel_size=5,
                        stride=2,
                        key=keys[0],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        key=keys[1],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        key=keys[2],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=3, stride=1),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=1,
                        stride=1,
                        key=keys[3],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )

        self.trunk = nn.Sequential(
            [
                nn.Linear(in_features=512, out_features=256, key=keys[4]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(in_features=256, out_features=256, key=keys[5]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(in_features=256, out_features=self.action_dim, key=keys[6]),
            ]
        )

    def __call__(self, x: jax.Array):
        """Expects image in [0, 255]"""
        x = x.transpose((0, 3, 1, 2)) / 255.0
        x = eqx.filter_vmap(self.cnn)(x)
        x = x.reshape(x.shape[0], -1)
        x = eqx.filter_vmap(self.trunk)(x)
        return x


class QNetworkRNN(eqx.Module):
    """CNN + MLP"""

    action_dim: int = 5
    cnn: nn.Sequential
    rnn: eqx.Module
    trunk: nn.Sequential

    def __init__(self, key: PRNGKeyArray, obs_size: int, rnn_type: str = "lru"):
        keys = jax.random.split(key, 8)
        if obs_size == 256:
            self.cnn = nn.Sequential(
                [
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=64,
                        kernel_size=7,
                        stride=2,
                        key=keys[0],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        key=keys[1],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        key=keys[2],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=3,
                        stride=2,
                        key=keys[3],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )
        else:
            self.cnn = nn.Sequential(
                [
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=64,
                        kernel_size=5,
                        stride=2,
                        key=keys[0],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        key=keys[1],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        key=keys[2],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                    nn.MaxPool2d(kernel_size=3, stride=1),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=1,
                        stride=1,
                        key=keys[3],
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )
        self.rnn = get_residual_memory_models(
            input=517,
            hidden=512,
            output=256,
            num_layers=2,
            layer_kwargs={
                "Attention": {"window_size": 128},
                "Attention-RoPE": {"window_size": 128},
                "Attention-ALiBi": {"window_size": 128},
            },
            models=[rnn_type],
            key=keys[4],
        )[rnn_type]
        self.trunk = nn.Sequential(
            [nn.Linear(in_features=256, out_features=self.action_dim, key=keys[7])]
        )

    def __call__(self, hidden_state, x, done, last_action):
        """Expects image in [0, 255]"""
        x = x.transpose((0, 1, 4, 2, 3)) / 255.0
        x = eqx.filter_vmap(eqx.filter_vmap(self.cnn))(x)

        x = x.reshape((x.shape[0], x.shape[1], -1))

        last_action = jax.nn.one_hot(last_action, self.action_dim)
        x = jnp.concatenate([x, last_action], axis=-1)
        rnn_in = (x, done)

        hidden_state, x = eqx.filter_vmap(self.rnn, in_axes=(0, 1), out_axes=(0, 1))(
            hidden_state, rnn_in
        )
        hidden_state = eqx.filter_vmap(self.rnn.latest_recurrent_state, in_axes=0)(
            hidden_state
        )

        q_vals = eqx.filter_vmap(eqx.filter_vmap(self.trunk))(x)

        return hidden_state, q_vals

    def initialize_carry(self, key: PRNGKeyArray):
        key_init = jax.random.split(key, 1)
        hidden_state = eqx.filter_jit(self.rnn.initialize_carry)(key=key_init[0])
        return hidden_state