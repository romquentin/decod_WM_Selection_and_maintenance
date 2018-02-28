"""
Figure 2 - Left
Plot diagonal decoding performance computed from
run_decoding_WM.py
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
from scipy.stats import ttest_1samp

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26
plt.rcParams['axes.linewidth'] = 2
title_size = 26
asterisk_size = 38

# Define analyses
analyses = ['left_angle', 'right_angle', 'left_sfreq', 'right_sfreq',
            'target_angle_cue_angle', 'target_sfreq_cue_sfreq',
            'cue_side', 'cue_type',
            'probe_angle', 'probe_sfreq']

# Define results to plot
results_folder = 'decoding_sensors'

# Define colors
colors = ['#1f77b4', '#d62728', '#ff7f0e']

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
    full_diag = list()
    sig = list()
    avg_sigs = list()
    for epoch_type in ['Target', 'Cue', 'Probe']:  # Loop accross each epoch
        all_diag = list()
        all_scores = list()
        for subject in subjects:
            if 'target' in analysis:
                fname_scores_cued = '%s_scores_%s_%s.npy' % (
                    subject, epoch_type, analysis)
                scores_cued = np.load(op.join(path_data, 'results/', subject,
                                      results_folder, fname_scores_cued))
                if 'angle' in analysis:
                    fname_scores_uncued = '%s_scores_%s_distr_angle.npy' % (
                        subject, epoch_type)
                    scores_uncued = np.load(op.join(path_data,
                                                    'results/', subject,
                                                    results_folder,
                                                    fname_scores_uncued))
                elif 'sfreq' in analysis:
                    fname_scores_uncued = '%s_scores_%s_distr_sfreq.npy' % (
                        subject, epoch_type)
                    scores_uncued = np.load(op.join(path_data,
                                                    'results/', subject,
                                                    results_folder,
                                                    fname_scores_uncued))
                scores = scores_cued - scores_uncued
            else:
                fname_scores = '%s_scores_%s_%s.npy' % (
                    subject, epoch_type, analysis)
                scores = np.load(op.join(path_data, 'results/', subject,
                                         results_folder, fname_scores))
            all_scores.append(scores)
            diag = np.diag(scores)
            all_diag.append(diag)
        all_scores = np.array(all_scores)  # all subjects scores for one epoch
        all_diag = np.array(all_diag)  # all subjects scores diag for one epoch
        full_diag.append(all_diag)  # all subjects scores diag for all epochs
        # stats on TG matrix at each time sample (cluster permutation test)
        p_values = gat_stats(np.array(all_scores) - chance)
        sig.append(np.diag(p_values < 0.05))  # permutation test signif < 0.05
        # stats on average decoding
        if epoch_type == 'Target':
            where = (times >= 0.05)
        elif epoch_type == 'Cue':
            where = (times_cue >= 0)
        elif epoch_type == 'Probe':
            where = (times >= 0) & (times < 0.4)
        t, p = ttest_1samp(all_diag[:, where].mean(axis=1), chance)
        avg_sigs.append([t, p])

    # Plot diagonal curve
    # Define name, ymax, ylim and colors for each analysis
    fig_param = {'left_angle': ['Left Angle', 0.14, -0.16/6., colors[2]],
                 'right_angle': ['Right Angle', 0.14, -0.16/10., colors[2]],
                 'left_sfreq': ['Left Spatial Frequency',
                                0.5, -0.16/10., colors[2]],
                 'right_sfreq': ['Right Spatial Frequency',
                                 0.5, -0.16/10., colors[2]],
                 'target_angle_cue_angle': ['Cued - Uncued Angle',
                                            0.12, -0.14/3, colors[1]],
                 'target_sfreq_cue_sfreq': ['Cued - Uncued Spatial Frequency',
                                            0.12, -0.14/3, colors[1]],
                 'cue_side': ['Cue Side (Spatial Rule)', 0.9,
                              chance - (0.9/10.)/2., colors[0]],
                 'cue_type': ['Cue Type (Object Rule)', 0.9,
                              chance - (0.9/10.)/2., colors[0]],
                 'probe_angle': ['Probe Angle', 0.18,
                                 -0.16/5., colors[2]],
                 'probe_sfreq': ['Probe Spatial Frequency', 0.6,
                                 -0.16/8., colors[2]]}
    title = fig_param[analysis][0]
    ymax = fig_param[analysis][1]
    ymin = fig_param[analysis][2]
    color = fig_param[analysis][3]

    # Open figure with 3 subplots (for stim, cue and probe epoch)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(16, 3.6),
                                        gridspec_kw={'width_ratios':
                                                     [132, 177, 48]})
    fig.set_dpi(80)
    # plot diag performance during stimulus epoch on ax1
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines['left'].set_color('k')
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
    ax1.get_yaxis().tick_left()
    ax1.set_ylim(ymin, ymax)
    ax1.set_yticks([chance, ymax])
    ax1.set_ylabel('r ', fontsize=title_size)
    if ('cue_type' in analysis) or ('cue_side' in analysis):
        ax1.set_ylabel('AUC', fontsize=title_size)
    ax1.axhline(y=chance, linewidth=0.5, color='black', ls='dotted')
    ax1.fill_between(times, ymin, ymax,
                     where=(times >= 0) & (times <= 0.1),
                     alpha=0.2,
                     color='gray',
                     interpolate=True)
    sem1 = np.std(full_diag[0], axis=0)/np.sqrt(len(subjects))
    mean1 = np.array(np.mean(full_diag[0], axis=0)) + (np.array(sem1))
    mean2 = np.array(np.mean(full_diag[0], axis=0))-(np.array(sem1))
    ax1.fill_between(times, mean1, mean2, color='0.6')
    ax1.fill_between(times, mean1, mean2, where=sig[0],
                     color=color, alpha=1)
    ax1.fill_between(times, mean1, chance,
                     where=sig[0], color=color, alpha=0.7)
    if (avg_sigs[0][1] <= 0.001) & (avg_sigs[0][0] > 0):
        ax1.text(0.5, 0.85, '***', fontsize=asterisk_size,
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes)
    elif (avg_sigs[0][1] <= 0.01) & (avg_sigs[0][1] > 0.001) & \
         (avg_sigs[0][0] > 0):
        ax1.text(0.5, 0.85, '**', fontsize=asterisk_size,
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes)
    elif (avg_sigs[0][1] <= 0.05) & (avg_sigs[0][1] > 0.01) & \
         (avg_sigs[0][0] > 0):
        ax1.text(0.5, 0.85, '*', fontsize=asterisk_size,
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes)

    # plot diag performance during cue epoch on ax2
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
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
    ax2.tick_params(
        axis='y',
        which='both',
        left='off',
        right='off',
        labelbottom='off')

    ax2.axhline(y=chance, linewidth=0.5, color='black', ls='dotted')
    ax2.fill_between(times, ymin, ymax,
                     where=(times >= 0) & (times <= 0.1),
                     alpha=0.2,
                     color='gray',
                     interpolate=True)
    sem2 = np.std(full_diag[1], axis=0)/np.sqrt(len(subjects))
    wh = np.where(times_cue >= 0)
    mean1 = np.array(np.mean(full_diag[1], axis=0)) + (np.array(sem2))
    mean2 = np.array(np.mean(full_diag[1], axis=0))-(np.array(sem2))
    ax2.fill_between(times_cue[wh], mean1[wh], mean2[wh], color='0.6')
    ax2.fill_between(times_cue[wh], mean1[wh], mean2[wh], where=sig[1][wh],
                     color=color, alpha=1)
    ax2.fill_between(times_cue[wh], mean2[wh], chance,
                     where=sig[1][wh], color=color, alpha=0.7)
    if (avg_sigs[1][1] <= 0.001) & (avg_sigs[1][0] > 0):
        ax2.text(0.5, 0.85, '***', fontsize=asterisk_size,
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes)
    elif (avg_sigs[1][1] <= 0.01) & (avg_sigs[1][1] > 0.001) & \
         (avg_sigs[1][0] > 0):
        ax2.text(0.5, 0.85, '**', fontsize=asterisk_size,
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes)
    elif (avg_sigs[1][1] <= 0.05) & (avg_sigs[1][1] > 0.01) & \
         (avg_sigs[1][0] > 0):
        ax2.text(0.5, 0.85, '*', fontsize=asterisk_size,
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes)

    # plot diag performance during probe epoch on ax3
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    if analysis == 'probe_angle':
        ax3.get_xaxis().tick_bottom()
        ax3.set_xticks(np.linspace(-0, 0.4, 2))
    else:
        ax3.tick_params(
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off')
    ax3.tick_params(
        axis='y',
        which='both',
        left='off',
        right='off',
        labelbottom='off')
    ax3.axhline(y=chance, linewidth=0.5, color='black', ls='dotted')
    ax3.fill_between(times, ymin, ymax,
                     where=(times >= 0) & (times <= 0.4),
                     alpha=0.2,
                     color='gray',
                     interpolate=True)
    sem3 = np.std(full_diag[2], axis=0)/np.sqrt(len(subjects))
    wh = np.where((times >= 0) & (times < 0.4))
    mean1 = np.array(np.mean(full_diag[2], axis=0)) + (np.array(sem3))
    mean2 = np.array(np.mean(full_diag[2], axis=0))-(np.array(sem3))
    ax3.fill_between(times[wh], mean1[wh], mean2[wh], color='0.6')
    ax3.fill_between(times[wh], mean1[wh], mean2[wh], where=sig[2][wh],
                     color=color, alpha=1)
    ax3.fill_between(times[wh], mean2[wh], chance,
                     where=sig[2][wh], color=color, alpha=0.7)
    if (avg_sigs[2][1] <= 0.001) & (avg_sigs[2][0] > 0):
        ax3.text(0.5, 0.85, '***', fontsize=asterisk_size,
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes)
    elif (avg_sigs[2][1] <= 0.01) & (avg_sigs[2][1] > 0.001) & \
         (avg_sigs[2][0] > 0):
        ax3.text(0.5, 0.85, '**', fontsize=asterisk_size,
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes)
    elif (avg_sigs[2][1] <= 0.05) & (avg_sigs[2][1] > 0.01) & \
         (avg_sigs[2][0] > 0):
        ax3.text(0.5, 0.85, '*', fontsize=asterisk_size,
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes)
    # Save figure
    fname = op.join(path_data, 'fig1/fig_diff', analysis + '.png')
    plt.savefig(fname, transparent=True, dpi='figure')
