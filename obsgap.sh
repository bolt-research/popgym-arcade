# Sweep one memory model to test obs gap and memory bias.
# For obs gap, we need 1 and 2 layers; 
# obs gap = J(f,pi,P) - J(f,pi,M), memory model cannot recover latent Markov state. So increasing layer will help;
# For memory bias, we need 6 and 8 layers. memory bias = J(f,pi,M) - J(pi,M). Larger layers, bigger bias.

# conda activate jaxenv
python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "Attention" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 1
python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "Attention" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 2
python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "Attention" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 1 --PARTIAL
python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "Attention" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 2 --PARTIAL

python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "Attention" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 6
python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "Attention" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 8
python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "Attention" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 6 --PARTIAL
python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "Attention" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 8 --PARTIAL
python popgym_arcade/train.py PQN --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 0
python popgym_arcade/train.py PQN --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 0 --PARTIAL


# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 1
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 2
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 1 --PARTIAL
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 2 --PARTIAL

# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 6
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 8
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 6 --PARTIAL
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 8 --PARTIAL
# python popgym_arcade/train.py PQN --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 0
# python popgym_arcade/train.py PQN --ENV_NAME "MineSweeperEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 0 --PARTIAL

# python popgym_arcade/train.py PQN --ENV_NAME "CartPoleEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 0
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "CartPoleEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 1
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "CartPoleEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 2
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "CartPoleEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 1 --PARTIAL
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "CartPoleEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 2 --PARTIAL

# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "CartPoleEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 6
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "CartPoleEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 8
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "CartPoleEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 6 --PARTIAL
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "mingru" --ENV_NAME "CartPoleEasy" --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --NUM_LAYERS 8 --PARTIAL
