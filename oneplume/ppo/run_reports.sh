MODEL="/Users/aaravsinha/manyplume/oneplume/ppo/trained_models/ExptMemory20250720/plume_20250720_VRNN_constantx5b5noisy3x5b5_stepoob_bx0.30.8_t10000004000000_q2.00.5_dmx0.80.8_dmn0.70.4_h64_wd0.0001_n1_codeVRNN_seed3031937.pt"
OUTDIR="$(dirname $MODEL)/viz_output"
mkdir -p $OUTDIR

# First, generate episode data if you don't have it yet
python -u evalCli.py --model_fname $MODEL --dataset constantx5b5 --fixed_eval --n_episodes 20 --save_logs

# Then create a simple script to generate the regime-colored visualization
cat > regime_viz.py << 'EOF'
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
sys.path.append('../')
import log_analysis
import agent_analysis
import config

# Load model and episode data
model_path = sys.argv[1]
outdir = sys.argv[2]

model_dir = model_path.replace('.pt', '/')
logfiles = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
if not logfiles:
    print("No episode logs found. Run evalCli.py first.")
    sys.exit(1)

with open(f"{model_dir}/{logfiles[0]}", 'rb') as f:
    episode_logs = pickle.load(f)

# Find an episode that shows all three regimes
for i, log in enumerate(episode_logs[:10]):
    traj_df = log_analysis.get_traj_df(log, extended_metadata=True, squash_action=True)
    regimes = traj_df['regime'].unique()
    if len(regimes) >= 3 and 'TRACK' in regimes and 'RECOVER' in regimes and 'SEARCH' in regimes:
        print(f"Found episode {i} with all three regimes")
        
        # Get regime colors
        regime_colorby = log_analysis.regime_to_colors(traj_df['regime'].to_list())
        
        # Create visualization
        fig, ax = agent_analysis.visualize_episodes([log], 
                                      zoom=-1, 
                                      dataset='constantx5b5',
                                      animate=False,
                                      plotsize=(7,7), 
                                      birthx=1.0,
                                      colorby=regime_colorby)
        
        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()
        patch1 = mpatches.Patch(color=config.regime_colormap['TRACK'], label='Track')   
        patch2 = mpatches.Patch(color=config.regime_colormap['RECOVER'], label='Recover')   
        patch3 = mpatches.Patch(color=config.regime_colormap['SEARCH'], label='Lost')   
        handles.extend([patch1, patch2, patch3])
        plt.legend(handles=handles, loc='best')
        
        # Save figure
        plt.savefig(f"{outdir}/regime_visualization_ep{i}.png", dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {outdir}/regime_visualization_ep{i}.png")
        plt.close()
        break
EOF

# Run the script to generate the visualization
python -u regime_viz.py $MODEL $OUTDIR