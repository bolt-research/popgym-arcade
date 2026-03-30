# Memory Tools

# Observability Gap: J(f, pi, M) - J(f, pi, P)
# Memory Bias: J(f, pi, M) - J(pi, M)
# run plottable to get the data, then run return_gap_bias to plot the obs gap and memory bias
python plottable.py
python return_gap_bias.py

# Pixel Visualizations
python analysis.py
# Recall Density
python run_multi_seed_analysis_pqn.py
python run_multi_seed_analysis_ppo.py
python plot_saliency_summary.py
python plot_saliency_by_env_models.py
# OOD Noise injection
python noiseva.py