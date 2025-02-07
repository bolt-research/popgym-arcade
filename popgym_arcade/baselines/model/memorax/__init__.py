from popgym_arcade.baselines.model.memorax.semigroups.fart import FART, FARTSemigroup
from popgym_arcade.baselines.model.memorax.semigroups.ffm import FFM, FFMSemigroup
from popgym_arcade.baselines.model.memorax.semigroups.gilr import GILR, GILRSemigroup

from popgym_arcade.baselines.model.memorax.train_utils import get_residual_memory_model, add_batch_dim

MONOIDS = [FARTSemigroup, FFMSemigroup, GILRSemigroup]
