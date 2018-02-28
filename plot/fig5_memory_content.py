"""
Figure 2
Plot diagonal decoding performance and time-generalization for cued item
and uncued item computed fromrun_decoding_WM.py
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
from copy import deepcopy
from scipy.stats import ttest_1samp
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
analyses = ['target_angle_cue_angle', 'target_sfreq_cue_sfreq',
            'distr_angle', 'distr_sfreq']

# Define results to plot
results_folder = 'sensors_patterns'
# Define colors
colors = ['#1f77b4', '#d62728', '#ff7f0e']
# Define epoch to plot and times for x-axis
epoch_type = 'Cue'
sfreq = 120
tmin = -.2
tmax = 1.5
sample_times = np.linspace(0, (tmax-tmin)*sfreq, (tmax-tmin)*sfreq + 1)
times = sample_times/sfreq + tmin
# Loop across each analysis
for analysis in analyses:
    all_scores = list()
    all_diag = list()
    for subject in subjects:
        fname_scores = '%s_scores_%s_%s.npy' % (
            subject, epoch_type, analysis)
        fname_patterns = '%s_patterns_%s_%s.npy' % (
            subject, epoch_type, analysis)
        fname_filters = '%s_filters_%s_%s.npy' % (
            subject, epoch_type, analysis)
        scores = np.load(op.join(path_data, 'results/', subject,
                                 results_folder, fname_scores))
        diag = np.diag(scores)
        all_scores.append(scores)
        all_diag.append(diag)
    all_scores = np.array(all_scores)   # all subjects tg scores for one epoch
    all_diag = np.array(all_diag)  # all subjects scores diag for one epoch
    # Define chance level
    if ('cue_type' in analysis) or ('cue_side' in analysis):
        chance = .5
    else:
        chance = 0
    # stats on TG matrix at each train/test sample (cluster permutation test)
    gat_p_values = gat_stats(np.array(all_scores) - chance)
    sig = np.array(gat_p_values < 0.05)  # permutation test signif < 0.05
    # stats on average decoding
    where = (times >= 0)
    _, p = ttest_1samp(all_diag[:, where].mean(axis=1), chance)

    # Define name, ymax, ylim and colors for each analysis
    fig_param = {'target_angle_cue_angle': ['Cued Angle', 0.1, colors[1]],
                 'target_sfreq_cue_sfreq': ['Cued Spatial Frequency', 0.1, colors[1]], # noqa
                 'distr_angle': ['Uncued Angle', 0.1, colors[1]],
                 'distr_sfreq': ['Uncued Spatial Frequency', 0.1, colors[1]]}
    title = fig_param[analysis][0]
    ymax = fig_param[analysis][1]
    ymin = -ymax/10.
    color = fig_param[analysis][2]

    # Open diagonal figure with 1 subplot for cue epoch
    fig_diag, axes = plt.subplots()
    fig_diag.set_size_inches(4, 3)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.get_xaxis().tick_bottom()
    axes.get_yaxis().tick_left()
    plt.xticks(np.linspace(-.2, 1.4, 9))
    plt.ylim(ymin, ymax)
    plt.xlabel('Time (s)', fontsize=legend_size)
    plt.ylabel('Decoding Performance (r)', fontsize=legend_size)
    plt.xticks(np.linspace(-.2, 1.4, 9))
    axes.axhline(y=chance, linewidth=0.5, color='gray')
    axes.fill_between(times, ymin, ymax,
                      where=(times >= 0) & (times <= 0.1),
                      alpha=0.2,
                      color='gray',
                      interpolate=True)
    sem = np.std(all_diag, axis=0)/np.sqrt(len(subjects))
    wh = np.where(times)
    mean1 = np.array(np.mean(all_diag, axis=0)) + (np.array(sem))
    mean2 = np.array(np.mean(all_diag, axis=0))-(np.array(sem))
    plt.fill_between(times[wh], mean1[wh], mean2[wh], color='0.6')
    plt.fill_between(times[wh], mean1[wh], mean2[wh], where=np.diag(sig[wh]),
                     color=color, alpha=1)
    plt.fill_between(times[wh], mean1[wh], chance,
                     where=np.diag(sig[wh]), color=color, alpha=0.7)
    if p <= 0.001:
        plt.text(0.5, 0.9, '***', fontsize=32, horizontalalignment='center',
                 verticalalignment='center', transform=axes.transAxes)
    elif (p <= 0.01) & (p > 0.001):
        plt.text(0.5, 0.9, '**', fontsize=32, horizontalalignment='center',
                 verticalalignment='center', transform=axes.transAxes)
    elif (p <= 0.05) & (p > 0.01):
        plt.text(0.5, 0.9, '*', fontsize=32, horizontalalignment='center',
                 verticalalignment='center', transform=axes.transAxes)
    # Save diagonal figure
    fname = op.join(path_data, 'fig2', analysis + '.png')
    plt.tight_layout()
    plt.savefig(fname, transparent=True)

    # Open TG figure with 1 subplot for cue epoch
    # Create colormap
    color = np.array(hex_to_rgb(fig_param[analysis][2]))/255.
    color = np.concatenate((color, [1]), axis=0)
    cmap = deepcopy(plt.get_cmap('magma_r'))
    cmap.colors = np.c_[np.linspace(1, color[0], 256),
                        np.linspace(1, color[1], 256),
                        np.linspace(1, color[2], 256)]

    fig_mean, axes = plt.subplots()
    fig_mean.set_size_inches(4, 3)
    imshow = axes.imshow(np.mean((all_scores), axis=0), origin='lower',
                         cmap='Reds',
                         extent=[tmin, tmax, tmin, tmax], vmin=0, vmax=ymax)
    cbar = plt.colorbar(imshow, ax=axes, ticks=[0, 0.1])
    cbar.ax.set_yticklabels(['0', '0.1'])
    cbar.set_label('r', rotation=270, fontsize=legend_size)
    axes.fill_between(times, times[0], times[-1],
                      where=(times >= 0) & (times <= 0.1),
                      alpha=0.2,
                      color='gray',
                      interpolate=True)
    axes.fill_betweenx(times, times[0], times[-1],
                       where=(times >= 0) & (times <= 0.1),
                       alpha=0.2,
                       color='gray')

    axes.set_xticks(np.linspace(0, 1.5, 4))
    axes.set_yticks(np.linspace(0, 1.5, 4))
    axes.set_xlabel('Test Times (s)', fontsize=legend_size)
    axes.set_ylabel('Train Times (s)', fontsize=legend_size)
    axes.xaxis.set_ticks_position('bottom')
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    plt.contour(xx, yy, sig, colors='Gray', levels=[0],
                linestyles='solid', linewidths=1)
    # Save TG figure
    fname = op.join(path_data, 'fig2', analysis + 'GAT.png')
    plt.tight_layout()
    plt.savefig(fname, transparent=True)
