"""
Supplementary Figure 3
Plot diagonal decoding performance from eye positions computed from
run_decoding_eyelink.py
"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from config import subjects, path_data
from base import decod_stats
from scipy.stats import ttest_1samp

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
            'cue_side', 'cue_type']
# Define results to plot
results_folder = 'decoding_from_eyelink'
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
    for subject in subjects:
        fname_scores = '%s_scores_%s_%s.npy' % (
            subject, epoch_type, analysis)
        scores = np.load(op.join(path_data, 'results/', subject,
                                 results_folder, fname_scores))
        diag = np.diag(scores)
        all_diag.append(diag)
    all_diag = np.array(all_diag)
    if ('cue_type' in analysis) or ('cue_side' in analysis):
        chance = .5
    else:
        chance = 0
    decod_p_values = decod_stats(np.array(all_diag) - chance)
    sig = decod_p_values < 0.05
    # stats on average decoding
    where = (times >= 0)
    _, p = ttest_1samp(all_diag[:, where].mean(axis=1), chance)

    # Plot diagonal curve
    # Define name, ymax, ylim and colors for each analysis
    fig_param = {'target_angle_cue_angle': ['Cued Angle', 0.1, colors[1]],
                 'target_sfreq_cue_sfreq': ['Cued Spatial Frequency', 0.1, colors[1]], # noqa
                 'cue_side': ['Cue Side', 0.9, colors[0]],
                 'cue_type': ['Cue Type', 0.9, colors[0]]}
    title = fig_param[analysis][0]
    ymax = fig_param[analysis][1]
    if ('cue_type' in analysis) or ('cue_side' in analysis):
        ymin = chance - (ymax/10.)/2.
    else:
        ymin = -ymax/2.5
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
    plt.yticks(np.linspace(-0.02, 0.1, 7))
    plt.xlabel('Time (s)', fontsize=legend_size)
    plt.ylabel('Decoding Performance (r)', fontsize=legend_size)
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
    plt.fill_between(times[wh], mean1[wh], mean2[wh], where=sig[wh],
                     color=color, alpha=1)
    plt.fill_between(times[wh], mean1[wh], chance,
                     where=sig[wh], color=color, alpha=0.7)
    if p <= 0.001:
        plt.text(0.5, 0.9, '***', fontsize=32, horizontalalignment='left',
                 verticalalignment='center', transform=axes.transAxes)
    elif (p <= 0.01) & (p > 0.001):
        plt.text(0.5, 0.9, '**', fontsize=32, horizontalalignment='left',
                 verticalalignment='center', transform=axes.transAxes)
    elif (p <= 0.05) & (p > 0.01):
        plt.text(0.5, 0.9, '*', fontsize=32, horizontalalignment='left',
                 verticalalignment='center', transform=axes.transAxes)
    # Save diag figure
    fname = op.join(path_data, 'fig_supp/fig_supp_3', analysis + '.png')
    plt.tight_layout()
    plt.savefig(fname, transparent=True)
