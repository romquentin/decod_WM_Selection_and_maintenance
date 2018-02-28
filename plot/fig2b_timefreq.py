"""
Figure 2 - Right
Plot time-freq decoding performance computed from
run_decoding_WM_timefreq.py
"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from config import subjects, path_data
from base import gat_stats
from webcolors import hex_to_rgb
from copy import deepcopy

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26
title_size = 26

# Define analyses
analyses = ['left_angle', 'right_angle', 'left_sfreq', 'right_sfreq',
            'target_angle_cue_angle', 'target_sfreq_cue_sfreq',
            'cue_side', 'cue_type',
            'probe_angle', 'probe_sfreq']

# Define results to plot
results_folder = 'time_frequency'

# Define colors
colors = ['#1f77b4', '#d62728', '#ff7f0e']
# Define frequencies
freqs = np.arange(2, 60, 2)

# Define times for x-axis
sfreq = 120
tmin = -.2
tmax = .9
tmax_cue = 1.5
sample_times = np.linspace(0, (tmax-tmin)*sfreq, (tmax-tmin)*sfreq + 1)
sample_times_cue = np.linspace(0, (tmax_cue-tmin)*sfreq,
                               (tmax_cue-tmin)*sfreq + 1)
times = sample_times/sfreq + tmin
times_cue = sample_times_cue/sfreq + tmin

# Loop across each analysis
for analysis in analyses:
    if ('cue_type' in analysis) or ('cue_side' in analysis):
        chance = .5
    else:
        chance = 0
    full_scores = list()
    full_sig = list()
    for epoch_type in ['Target', 'Cue', 'Probe']:  # Loop accross each epoch
        all_scores = list()
        all_decod_p_values = list()
        all_sig = list()
        for subject in subjects:
            if 'target' in analysis:
                fname_scores_cued = '%s_tf_scores_%s_%s.npy' % (
                    subject, epoch_type, analysis)
                scores_cued = np.load(op.join(path_data, 'results/', subject,
                                      results_folder, fname_scores_cued))
                if 'angle' in analysis:
                    fname_scores_uncued = '%s_tf_scores_%s_distr_angle.npy' % (
                        subject, epoch_type)
                    scores_uncued = np.load(op.join(path_data,
                                                    'results/', subject,
                                                    results_folder,
                                                    fname_scores_uncued))
                elif 'sfreq' in analysis:
                    fname_scores_uncued = '%s_tf_scores_%s_distr_sfreq.npy' % (
                        subject, epoch_type)
                    scores_uncued = np.load(op.join(path_data,
                                                    'results/', subject,
                                                    results_folder,
                                                    fname_scores_uncued))
                scores = scores_cued - scores_uncued
            else:
                fname_scores = '%s_tf_scores_%s_%s.npy' % (
                    subject, epoch_type, analysis)
                scores = np.load(op.join(path_data, 'results/', subject,
                                         results_folder, fname_scores))
            all_scores.append(scores)
        all_scores = np.array(all_scores)  # all subjects scores for one epoch
        # Permutation test with cluster on 2 dimensions
        all_gat_p_values = gat_stats(np.array(all_scores) - chance)
        all_sig = np.array(all_gat_p_values < 0.05)
        full_scores.append(all_scores)  # all subjects scores for all epochs
        full_sig.append(all_sig)  # significance of permutation test < 0.05

    # Plot 2D time_freq
    # Define name, ymax and colors for each analysis
    fig_param = {'left_angle': ['Left Angle', 0.1, colors[2]],
                 'right_angle': ['Right Angle', 0.1, colors[2]],
                 'left_sfreq': ['Left Spatial Frequency', 0.3, colors[2]],
                 'right_sfreq': ['Right Spatial Frequency', 0.3, colors[2]],
                 'target_angle_cue_angle': ['Cued - Uncued Angle', 0.1,
                                            colors[1]],
                 'target_sfreq_cue_sfreq': ['Cued - Uncued Spatial Frequency',
                                            0.1, colors[1]],
                 'cue_side': ['Cue Side (Spatial Rule)', 0.57, colors[0]],
                 'cue_type': ['Cue Type (Object Rule)', 0.57, colors[0]],
                 'probe_angle': ['Probe Angle', 0.1, colors[2]],
                 'probe_sfreq': ['Probe Spatial Frequency', 0.3, colors[2]]}
    title = fig_param[analysis][0]
    ymax = fig_param[analysis][1]
    # create colormap
    color = np.array(hex_to_rgb(fig_param[analysis][2]))/255.
    color = np.concatenate((color, [1]), axis=0)
    cmap = deepcopy(plt.get_cmap('magma_r'))
    cmap.colors = np.c_[np.linspace(1, color[0], 256),
                        np.linspace(1, color[1], 256),
                        np.linspace(1, color[2], 256)]

    #  Define ymin
    if ('cue_type' in analysis) or ('cue_side' in analysis):
        ymin = 0.5
    else:
        ymin = 0

    # Open figure with 3 subplots (for stim, cue and probe epoch)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(16, 3.6),
                                        gridspec_kw={'width_ratios':
                                                     [132, 177, 48]})
    # plot time-freq decoding performance during stimulus epoch on ax1
    ax1.set_ylabel('Frequencies', fontsize=title_size)
    if analysis == 'probe_angle':
        ax1.get_xaxis().tick_bottom()
        ax1.set_xticks(np.linspace(0, .8, 3))
    else:
        ax1.tick_params(
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off')
    ax1.set_yticks([10, 50])
    ax1.imshow(np.mean((full_scores[0]), axis=0), aspect='auto',
               origin='lower', cmap=cmap,
               extent=[tmin, tmax, freqs[0], freqs[-1]],
               vmin=ymin, vmax=ymax)
    xx, yy = np.meshgrid(times, freqs, copy=False, indexing='xy')
    ax1.contour(xx, yy, full_sig[0], colors='k', levels=[0],
                linestyles='dashed', linewidths=3)
    ax1.fill_between(times, freqs[0], freqs[-1],
                     where=(times >= 0) & (times <= 0.1),
                     alpha=0.2,
                     color='gray',
                     interpolate=True)
    # plot time-freq decoding performance during cue epoch on ax2
    wh = np.where(times_cue >= 0)  # start at cue onset
    if analysis == 'probe_angle':
        ax2.get_xaxis().tick_bottom()
        ax2.set_xticks(np.linspace(0, 1.2, 4))
    else:
        ax2.tick_params(
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off')
    ax2.imshow(np.mean((full_scores[1]), axis=0), aspect='auto',
               origin='lower', cmap=cmap,
               extent=[0, tmax_cue, freqs[0], freqs[-1]],
               vmin=ymin, vmax=ymax)
    xx, yy = np.meshgrid(times_cue[wh], freqs, copy=False, indexing='xy')
    ax2.contour(xx, yy,
                np.reshape(full_sig[1][:, wh], (len(freqs), len(wh[0]))),
                colors='k', levels=[0],
                linestyles='dashed', linewidths=3)
    ax2.fill_between(times_cue[wh], freqs[0], freqs[-1],
                     where=(times_cue[wh] >= 0) & (times_cue[wh] <= 0.1),
                     alpha=0.2,
                     color='gray',
                     interpolate=True)

    # plot time-freq decoding performance during probe epoch on ax3
    # start only at cue onset and stop at times=0.4s)
    wh = np.where((times_cue >= 0) & (times_cue < 0.4))
    if analysis == 'probe_angle':
        ax3.get_xaxis().tick_bottom()
        ax3.set_xticks(np.linspace(-0, 0.4, 2))
    else:
        ax3.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
    axes3 = ax3.imshow(np.mean((full_scores[2]), axis=0), aspect='auto',
                       origin='lower', cmap=cmap,
                       extent=[0, 0.4, freqs[0], freqs[-1]],
                       vmin=ymin, vmax=ymax)
    xx, yy = np.meshgrid(times[wh],
                         freqs, copy=False, indexing='xy')
    ax3.contour(xx, yy,
                np.reshape(full_sig[2][:, wh], (len(freqs), len(wh[0]))),
                colors='k', levels=[0],
                linestyles='dashed', linewidths=3)
    ax3.fill_between(times[wh], freqs[0], freqs[-1],
                     where=(times[wh] >= 0) & (times[wh] <= 0.4),
                     alpha=0.2,
                     color='gray',
                     interpolate=True)
    cbar = fig.colorbar(axes3, ticks=[ymax])
    cbar.set_label('r', rotation=270, fontsize=title_size)
    if ('cue_type' in analysis) or ('cue_side' in analysis):
        cbar.set_label('AUC', rotation=270, fontsize=title_size)
    # Save figure
    fname = op.join(path_data, 'fig1_22sub/fig_timefreq_diff', analysis + '.png')
    plt.savefig(fname, transparent=True)
