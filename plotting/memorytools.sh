# Memory Tools

# Observability Gap: J(f, pi, M) - J(f, pi, P)
# Memory Bias: J(f, pi, M) - J(pi, M)
# run plottable to get the data, then run return_gap_bias to plot the obs gap and memory bias
python plotting/download_csv.py \
    --entity your_wandb_entity \
    --project=your_wandb_project_name \
    --model-group-csv my_rnn.csv 
python plotting/return_gap_bias.py \
    --input-csv my_rnn.csv \
    --output gap_bias_plot.pdf 
# Pixel Visualizations
python plotting/pixel_vis_pqn.py \
    --model-path my_rnn_weight.pkl \
    --env-name CartPoleEasy \
    --memory_type my_rnn \
    --output pixel_vis.pdf 
# recall density 
python plotting/density_analysis_pqn.py \
    --pkls_dir model_weights_dir \
    --out_dir your_recall_density_dir 
python plotting/plot_density_summary.py \
    --recall_density_dir your_recall_density_dir \
    --output density_plot.pdf 
# OOD Noise injection
python plotting/noiseva.py \
    --model-dir /path/to/checkpoints \
    --memory-types lru \
    --env-names CartPoleEasy \

