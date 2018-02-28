"""
Supplementary Figure 4
Plot diagonal and time-freq decoding performance for the rule trained on WM
task and tested on WM task and control task (localizer)
"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from config import path_data, subjects_con
from base import gat_stats
from copy import deepcopy
from webcolors import hex_to_rgb

subjects = [subjects_con]

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
results_folder = 'cross_task_cue_control'
# Define colors
colors = ['#1f77b4', '#d62728', '#ff7f0e']
# Define times for x-axis
epoch_type = 'Cue'
sfreq = 120
tmin = -.2
tmax = 1.5
sample_times = np.linspace(0, (tmax-tmin)*sfreq, (tmax-tmin)*sfreq + 1)
times = sample_times/sfreq + tmin

# Loop across each analysis
for analysis in analyses:
    all_diag = list()
    all_diag_con = list()
    all_scores = list()  # scores trained/tested on WM task
    all_scores_con = list()  # scores trained on WM task and tested on loca
    for subject in subjects:
        fname_scores = '%s_scores_%s.npy' % (
            subject, analysis)
        scores = np.load(op.join(path_data, 'results/', subject,
                                 results_folder, fname_scores))
        fname_scores = '%s_scores_%s_con.npy' % (
            subject, analysis)
        scores_con = np.load(op.join(path_data, 'results/', subject,
                                     results_folder, fname_scores))
        diag = np.diag(scores)
        diag_con = np.diag(scores_con)
        all_scores.append(scores)
        all_scores_con.append(scores_con)
        all_diag.append(diag)
        all_diag_con.append(diag_con)
    all_scores = np.array(all_scores)
    all_scores_con = np.array(all_scores_con)
    all_diag = np.array(all_diag)
    all_diag_con = np.array(all_diag_con)

    chance = .5

    # stats on gat matrix at each time sample (cluster permutation test)
    gat_p_values = gat_stats(np.array(all_scores) - chance)
    sig = np.array(gat_p_values < 0.05)
    gat_p_values = gat_stats(np.array(all_scores_con) - chance)
    sig_con = np.array(gat_p_values < 0.05)

    # Plot diagonal curve
    # Define name, ymax, ylim and colors for each analysis
    fig_param = {'cue_side': ['Cue Side', 0.93, colors[0]],
                 'cue_type': ['Cue Type', 0.93, colors[0]]}
    title = fig_param[analysis][0]
    ymax = fig_param[analysis][1]
    ymin = chance - (ymax/10.)/2.
    color = fig_param[analysis][2]

    # Plot diagonal curve
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
    sem_con = np.std(all_diag_con, axis=0)/np.sqrt(len(subjects))
    wh = np.where(times)
    mean1 = np.array(np.mean(all_diag, axis=0)) + (np.array(sem))
    mean2 = np.array(np.mean(all_diag, axis=0))-(np.array(sem))
    mean1_con = np.array(np.mean(all_diag_con, axis=0)) + (np.array(sem))
    mean2_con = np.array(np.mean(all_diag_con, axis=0))-(np.array(sem))

    plt.fill_between(times[wh], mean1[wh], mean2[wh], color='0.5')
    plt.fill_between(times[wh], mean1[wh], mean2[wh],
                     where=np.diag(sig[wh]),
                     color=color, alpha=0.3)
    plt.fill_between(times[wh], mean1[wh], chance,
                     where=np.diag(sig[wh]), color=color, alpha=0.3)
    plt.fill_between(times[wh], mean1_con[wh], mean2_con[wh], color='0.5')
    plt.fill_between(times[wh], mean1_con[wh], mean2_con[wh],
                     where=np.diag(sig_con[wh]),
                     color='gray', alpha=0.3)
    plt.fill_between(times[wh], mean1_con[wh], chance,
                     where=np.diag(sig[wh]), color='gray', alpha=0.3)

    # Save figure diag
    fname = op.join(path_data, 'fig_supp/fig_supp_4', analysis + '.png')
    plt.tight_layout()
    plt.savefig(fname, transparent=True)

    # Plot time generalization
    color = np.array(hex_to_rgb(fig_param[analysis][2]))/255.
    color = np.concatenate((color, [1]), axis=0)
    cmap = deepcopy(plt.get_cmap('magma_r'))
    cmap.colors = np.c_[np.linspace(1, color[0], 256),
                        np.linspace(1, color[1], 256),
                        np.linspace(1, color[2], 256)]

    fig_mean, axes = plt.subplots()
    fig_mean.set_size_inches(3.65, 2.75)
    fig_mean_con, axes_con = plt.subplots()
    fig_mean_con.set_size_inches(3.65, 2.75)
    imshow = axes.imshow(np.mean((all_scores), axis=0), origin='lower',
                         cmap=cmap,
                         extent=[tmin, tmax, tmin, tmax], vmin=0, vmax=0.7)

    imshow = axes_con.imshow(np.mean((all_scores_con), axis=0), origin='lower',
                             cmap='gray_r',
                             extent=[tmin, tmax, tmin, tmax], vmin=0, vmax=0.7)

    cbar = fig_mean.colorbar(imshow, ax=axes, ticks=[0.5, 0.7])
    cbar.ax.set_yticklabels(['0.5', '0.7'])
    cbar.set_label('AUC', rotation=270, fontsize=legend_size)
    cbar_con = fig_mean_con.colorbar(imshow, ax=axes, ticks=[0.5, 0.7])
    cbar_con.ax.set_yticklabels(['0.5', '0.7'])
    cbar_con.set_label('AUC', rotation=270, fontsize=legend_size)
    axes.fill_between(times, times[0], times[-1],
                      where=(times >= 0) & (times <= 0.1),
                      alpha=0.2,
                      color='gray',
                      interpolate=True)
    axes.fill_betweenx(times, times[0], times[-1],
                       where=(times >= 0) & (times <= 0.1),
                       alpha=0.2,
                       color='gray')
    axes_con.fill_between(times, times[0], times[-1],
                          where=(times >= 0) & (times <= 0.1),
                          alpha=0.2,
                          color='gray',
                          interpolate=True)
    axes_con.fill_betweenx(times, times[0], times[-1],
                           where=(times >= 0) & (times <= 0.1),
                           alpha=0.2,
                           color='gray')
    axes.set_xticks(np.linspace(0, 1.5, 4))
    axes.set_yticks(np.linspace(0, 1.5, 4))
    axes.set_xlabel('Test Times', fontsize=legend_size)
    axes.set_ylabel('Train Times', fontsize=legend_size)
    axes.xaxis.set_ticks_position('bottom')
    axes_con.set_xticks(np.linspace(0, 1.5, 4))
    axes_con.set_yticks(np.linspace(0, 1.5, 4))
    axes_con.set_xlabel('Test Times', fontsize=legend_size)
    axes_con.set_ylabel('Train Times', fontsize=legend_size)
    axes_con.xaxis.set_ticks_position('bottom')

    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    axes.contour(xx, yy, sig, colors='Gray', levels=[0],
                     linestyles='solid', linewidths=1)
    axes_con.contour(xx, yy, sig_con, colors='Gray', levels=[0],
                         linestyles='solid', linewidths=1)

    # Save time generalization figure
    fname = op.join(path_data, 'fig_supp/fig_supp_4', analysis + 'GAT.png')
    fig_mean.tight_layout()
    fig_mean.savefig(fname, transparent=True)
    fname = op.join(path_data, 'fig_supp/fig_supp_4', analysis + 'GAT_con.png')
    fig_mean_con.tight_layout()
    fig_mean_con.savefig(fname, transparent=True)

# Plot time_frequency
# Define results to plot
results_folder = 'cross_task_time_freq_cue_control'
# Define frequencies
freqs = np.arange(2, 60, 2)
# Loop across each analysis
for analysis in analyses:
    chance = 0.5
    all_scores = list()  # trained/tested on WM task
    all_scores_con = list()  # trained on WM task and tested on loca
    for subject in subjects:
        fname_scores = '%s_scores_tf_%s.npy' % (
            subject, analysis)
        scores = np.load(op.join(path_data, 'results/', subject,
                                 results_folder, fname_scores))
        fname_scores = '%s_scores_tf_%s_con.npy' % (
            subject, analysis)
        scores_con = np.load(op.join(path_data, 'results/', subject,
                                     results_folder, fname_scores))
        all_scores.append(scores)
        all_scores_con.append(scores_con)
    all_scores = np.array(all_scores)
    all_scores_con = np.array(all_scores_con)

    # stats on gat matrix at each time sample (cluster permutation test)
    all_gat_p_values = gat_stats(np.array(all_scores) - chance)
    all_sig = np.array(all_gat_p_values < 0.05)
    all_gat_p_values = gat_stats(np.array(all_scores_con) - chance)
    all_sig_con = np.array(all_gat_p_values < 0.05)

    # Define name, ymax, ylim and colors for each analysis
    fig_param = {'cue_side': ['Cue Side (without associated rule)', 0.65, colors[0]],
                 'cue_type': ['Cue Type (without associated rule)', 0.65, colors[0]]}
    title = fig_param[analysis][0]
    ymax = fig_param[analysis][1]
    color = fig_param[analysis][2]
    # Create cmap
    color = np.array(hex_to_rgb(fig_param[analysis][2]))/255.
    color = np.concatenate((color, [1]), axis=0)
    cmap = deepcopy(plt.get_cmap('magma_r'))
    cmap.colors = np.c_[np.linspace(1, color[0], 256),
                        np.linspace(1, color[1], 256),
                        np.linspace(1, color[2], 256)]
    ymin = 0.5

    # Plot time-freq figure
    fig, ax1 = plt.subplots()
    fig_con, ax1_con = plt.subplots()
    fig.set_size_inches(4.5, 2.75)
    fig_con.set_size_inches(4.5, 2.75)
    ax1.set_ylabel('Frequencies', fontsize=legend_size)
    ax1_con.set_ylabel('Frequencies', fontsize=legend_size)
    ax1.get_xaxis().tick_bottom()
    ax1.get_xaxis().tick_bottom()
    ax1.set_xticks(np.linspace(0, 1.5, 4))
    ax1.set_xlabel('Times (s)', fontsize=legend_size)
    ax1_con.get_xaxis().tick_bottom()
    ax1_con.get_xaxis().tick_bottom()
    ax1_con.set_xticks(np.linspace(0, 1.5, 4))
    ax1_con.set_xlabel('Times (s)', fontsize=legend_size)

    axes1 = ax1.imshow(np.mean((all_scores), axis=0), aspect='auto',
                       origin='lower', cmap=cmap,
                       extent=[tmin, tmax, freqs[0], freqs[-1]],
                       vmin=ymin, vmax=ymax)
    axes1_con = ax1_con.imshow(np.mean((all_scores_con), axis=0),
                               aspect='auto', origin='lower', cmap='gray_r',
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

    ax1_con.contour(xx, yy, all_sig_con, colors='Gray', levels=[0],
                    linestyles='solid', linewidths=1)
    ax1_con.fill_between(times, freqs[0], freqs[-1],
                         where=(times >= 0) & (times <= 0.1),
                         alpha=0.2,
                         color='gray',
                         interpolate=True)

    cbar = fig.colorbar(axes1, ticks=[ymin, ymax])
    cbar.set_label('AUC', rotation=270, fontsize=legend_size)
    cbar_con = fig_con.colorbar(axes1_con, ticks=[ymin, ymax])
    cbar_con.set_label('AUC', rotation=270, fontsize=legend_size)

    # Save time-freq figure
    fname = op.join(path_data, 'fig_supp/fig_supp_4', analysis + 'tf.png')
    fig.tight_layout()
    fig.savefig(fname, transparent=True)
    fname = op.join(path_data, 'fig_supp/fig_supp_4', analysis + 'tf_con.png')
    fig_con.tight_layout()
    fig_con.savefig(fname, transparent=True)
