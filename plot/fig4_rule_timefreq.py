"""
Figure 4
Plot time-freq decoding performance for cue side and
type computed from run_decoding_WM_timefreq.py and
run_decoding_locacue_timefreq.py
"""

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from config import path_data, subjects, subjects_con
from base import gat_stats
from copy import deepcopy
from webcolors import hex_to_rgb

title_size = 12
legend_size = 16
ticks_size = 12
asterisk_size = 24
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.labelsize'] = ticks_size
plt.rcParams['ytick.labelsize'] = ticks_size

# Define analyses
analyses = ['cue_side', 'cue_type']
# Define results to plot
results_folders = ['time_frequency', 'locacue_timefreq']
# Define subjects
subjectss = [subjects, subjects_con]
# Define colors
colors = ['#1f77b4', '#d62728', '#ff7f0e']
# Define frequencies
freqs = np.arange(2, 60, 2)
# Define epoch to plot and times for x-axis
sfreq = 120
tmin = -.2
tmax = 1.5
sample_times = np.linspace(0, (tmax-tmin)*sfreq,
                              (tmax-tmin)*sfreq + 1)
times = sample_times/sfreq + tmin
# Loop across each analysis
for results_folder, subjects in zip(results_folders, subjectss):
    for analysis in analyses:
        chance = 0.5
        all_scores = list()
        all_decod_p_values = list()
        all_sig = list()
        for subject in subjects:
            fname_scores = '%s_tf_scores_%s_%s.npy' % (
                subject, 'Cue', analysis)
            scores = np.load(op.join(path_data, 'results/', subject,
                                     results_folder, fname_scores))
            all_scores.append(scores)
        all_scores = np.array(all_scores)  # all subjects scores diag for
        # Permutation test with cluster on 2 dimensions
        all_gat_p_values = gat_stats(np.array(all_scores) - chance)
        all_sig = np.array(all_gat_p_values < 0.05)

        # Define name, ymax, ylim and colors for each analysis
        fig_param = {'cue_side': ['Cue Side (without associated rule)', 0.57, colors[0]],
                     'cue_type': ['Cue Type (without associated rule)', 0.57, colors[0]]}
        title = fig_param[analysis][0]
        ymax = fig_param[analysis][1]
        ymin = 0.5
        # Define colormap
        if 'time_frequency' in results_folder:
            color = np.array(hex_to_rgb(fig_param[analysis][2]))/255.
            color = np.concatenate((color, [1]), axis=0)
            cmap = deepcopy(plt.get_cmap('magma_r'))
            cmap.colors = np.c_[np.linspace(1, color[0], 256),
                                np.linspace(1, color[1], 256),
                                np.linspace(1, color[2], 256)]
        else:
            cmap = 'gray_r'
        # Open time-freq figure with 1 subplot for cue epoch
        fig, ax1 = plt.subplots()
        fig.set_size_inches(4.5, 2.75)
        ax1.set_ylabel('Frequencies', fontsize=legend_size)
        ax1.get_xaxis().tick_bottom()
        ax1.set_xticks(np.linspace(0, 1.5, 4))
        ax1.set_xlabel('Times (s)', fontsize=legend_size)
        axes1 = ax1.imshow(np.mean((all_scores), axis=0), aspect='auto',
                           origin='lower', cmap=cmap,
                           extent=[tmin, tmax, freqs[0], freqs[-1]],
                           vmin=ymin, vmax=ymax)
        xx, yy = np.meshgrid(times, freqs, copy=False, indexing='xy')
        ax1.contour(xx, yy, all_sig, colors='Gray', levels=[0],
                    linestyles='solid', linewidths=1)
        ax1.fill_between(times, freqs[0], freqs[-1],
                         where=(times >= 0) & (times <= 0.1),
                         alpha=0.2,
                         color='gray',
                         interpolate=True)
        cbar = fig.colorbar(axes1, ticks=[ymin, ymax])
        cbar.set_label('AUC', rotation=270, fontsize=legend_size)
        # Save time-freq figure
        fname = op.join(path_data, 'fig4', '%s_%s.png' % (analysis,
                                                          results_folder))
        plt.tight_layout()
        plt.savefig(fname, transparent=True)
