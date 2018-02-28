"""
Figure 4
Plot diagonal decoding performance and time-generalization for cue side and
type computed from run_decoding_WM.py and run_decoding_locacue.py
"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from config import path_data
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
analyses = ['cue_side', 'cue_type']
# Define results to plot
results_folder = 'sensors_patterns'
results_folder_con = 'locacue'
# Define subjects
subjectss = [subjects, subjects_con]

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
    all_scores_con = list()
    all_diag_con = list()
    for subject in subjects:  # during WM task
        fname_scores = '%s_scores_%s_%s.npy' % (
            subject, epoch_type, analysis)
        scores = np.load(op.join(path_data, 'results/', subject,
                                 results_folder, fname_scores))
        diag = np.diag(scores)
        all_scores.append(scores)
        all_diag.append(diag)
    all_scores = np.array(all_scores)  # all subjects tg scores for one epoch
    all_diag = np.array(all_diag)  # all subjects scores diag for one epoch
    for subject in subjects_con:  # during control task (localizer)
        fname_scores = '%s_scores_%s_%s.npy' % (
            subject, epoch_type, analysis)
        scores_con = np.load(op.join(path_data, 'results/', subject,
                             results_folder_con, fname_scores))
        diag_con = np.diag(scores_con)
        all_scores_con.append(scores_con)
        all_diag_con.append(diag_con)
    all_scores_con = np.array(all_scores_con)  # all subjects tg scores
    all_diag_con = np.array(all_diag_con)  # all subjects scores diag

    # Define chance level
    chance = .5
    # stats on TG matrix at each train/test sample (cluster permutation test)
    gat_p_values = gat_stats(np.array(all_scores) - chance)
    sig = np.array(gat_p_values < 0.05)
    gat_p_values = gat_stats(np.array(all_scores_con) - chance)
    sig_con = np.array(gat_p_values < 0.05)
    # stats on average decoding
    where = (times >= 0)
    _, p = ttest_1samp(all_diag[:, where].mean(axis=1), chance)

    # Define name, ymax, ylim and colors for each analysis
    fig_param = {'sensors_patterns_cue_side': ['Cue Side (Spatial Rule)', 0.9,
                                               colors[0]],
                 'sensors_patterns_cue_type': ['Cue Type (Feature Rule)', 0.9,
                                               colors[0]],
                 'locacue_cue_side': ['Cue Side (without associated rule)',
                                      0.9, 'gray'],
                 'locacue_cue_type': ['Cue Type (without associated rule)',
                                      0.9, 'gray']}
    title = fig_param[results_folder + '_' + analysis][0]
    ymax = fig_param[results_folder + '_' + analysis][1]
    ymin = chance - (ymax/10.)/2.
    color = fig_param[results_folder + '_' + analysis][2]
    # Open diagonal figure with 1 subplot for cue epoch
    fig_diag, axes = plt.subplots()
    fig_diag.set_size_inches(6.9, 5.2)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.get_xaxis().tick_bottom()
    axes.get_yaxis().tick_left()
    plt.ylim(ymin, ymax)
    plt.yticks(np.linspace(0.5, 0.9, 3))
    plt.ylabel('Decoding Performance (AUC)', fontsize=legend_size)
    plt.xlabel('Time (s)', fontsize=legend_size)
    plt.xticks(np.linspace(-.2, 1.4, 9))
    axes.axhline(y=chance, linewidth=0.5, color='gray')
    axes.fill_between(times, ymin, ymax,
                      where=(times >= 0) & (times <= 0.1),
                      alpha=0.2,
                      color='gray',
                      interpolate=True)
    sem = np.std(all_diag, axis=0)/np.sqrt(len(subjects))
    sem_con = np.std(all_diag_con, axis=0)/np.sqrt(len(subjects))
    wh = np.where(times)
    mean1 = np.array(np.mean(all_diag, axis=0)) + (np.array(sem))
    mean2 = np.array(np.mean(all_diag, axis=0))-(np.array(sem))
    mean1_con = np.array(np.mean(all_diag_con, axis=0)) + (np.array(sem))
    mean2_con = np.array(np.mean(all_diag_con, axis=0))-(np.array(sem))

    plt.fill_between(times[wh], mean1[wh], mean2[wh], color='0.6')
    plt.fill_between(times[wh], mean1[wh], mean2[wh],
                     where=np.diag(sig[wh]),
                     color=color, alpha=1)
    plt.fill_between(times[wh], mean1[wh], chance,
                     where=np.diag(sig[wh]), color=color, alpha=0.7)
    plt.fill_between(times[wh], mean1_con[wh], mean2_con[wh], color='0.5')
    plt.fill_between(times[wh], mean1_con[wh], mean2_con[wh],
                     where=np.diag(sig_con[wh]),
                     color='gray', alpha=0.3)
    plt.fill_between(times[wh], mean1_con[wh], chance,
                     where=np.diag(sig[wh]), color='gray', alpha=0.3)
    if p <= 0.001:
        plt.text(0.5, 0.9, '***', fontsize=asterisk_size,
                 horizontalalignment='center', verticalalignment='center',
                 transform=axes.transAxes)
    elif (p <= 0.01) & (p > 0.001):
        plt.text(0.5, 0.9, '**', fontsize=asterisk_size,
                 horizontalalignment='center', verticalalignment='center',
                 transform=axes.transAxes)
    elif (p <= 0.05) & (p > 0.01):
        plt.text(0.5, 0.9, '*', fontsize=asterisk_size,
                 horizontalalignment='center', verticalalignment='center',
                 transform=axes.transAxes)
    # Save diagonal figures
    suffix = '_control' if results_folder == 'locacue' else ''
    fname = op.join(path_data, 'fig4', analysis + '.png')
    plt.tight_layout()
    plt.savefig(fname, transparent=True)

    # Open TG figure with 1 subplot for cue epoch
    # Create colormap
    color = np.array(hex_to_rgb(fig_param[results_folder + '_' + analysis][2]))/255.
    color = np.concatenate((color, [1]), axis=0)
    cmap = deepcopy(plt.get_cmap('magma_r'))
    cmap.colors = np.c_[np.linspace(1, color[0], 256),
                        np.linspace(1, color[1], 256),
                        np.linspace(1, color[2], 256)]

    fig_gat, axes = plt.subplots()
    fig_gat.set_size_inches(3.65, 2.75)
    imshow = axes.imshow(np.mean((all_scores), axis=0), origin='lower',
                         cmap=cmap,
                         extent=[tmin, tmax, tmin, tmax], vmin=0.5, vmax=0.7)
    cbar = plt.colorbar(imshow, ax=axes, ticks=[0.5, 0.7])
    cbar.ax.set_yticklabels(['0.5', '0.7'])
    cbar.set_label('AUC', rotation=270, fontsize=legend_size)

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

    fname = op.join(path_data, 'fig4', analysis + 'GAT.png')
    plt.tight_layout()
    plt.savefig(fname, transparent=True)

    # Plot mean subjects control
    fig_gat, axes = plt.subplots()
    fig_gat.set_size_inches(3.65, 2.75)
    imshow = axes.imshow(np.mean((all_scores_con), axis=0), origin='lower',
                         cmap='gray_r',
                         extent=[tmin, tmax, tmin, tmax], vmin=0.5, vmax=0.7)
    cbar = plt.colorbar(imshow, ax=axes, ticks=[0.5, 0.7])
    cbar.ax.set_yticklabels(['0.5', '0.7'])
    cbar.set_label('AUC', rotation=270, fontsize=legend_size)

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
    plt.contour(xx, yy, sig_con, colors='Gray', levels=[0],
                linestyles='solid', linewidths=1)
    # Save TG figure
    fname = op.join(path_data, 'fig4', analysis + suffix + 'GAT_con.png')
    plt.tight_layout()
    plt.savefig(fname, transparent=True)
