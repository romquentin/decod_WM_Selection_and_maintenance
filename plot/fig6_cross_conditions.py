"""
Figure 6
Plot barplot of mean decoding performance across memory and visual
conditions computed from run_decoding_WM_across_epochs_and_conditions.py
"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from config import subjects, path_data

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Define results to plot
results_folder = 'sensors_accross_epochs_and_conditions'
# Define times
sfreq = 120
tmin = -.2
tmax = .9
tmax_cue = 1.5
sample_times = np.linspace(0, (tmax-tmin)*sfreq, (tmax-tmin)*sfreq + 1)
sample_times_cue = np.linspace(0, (tmax_cue-tmin)*sfreq,
                               (tmax_cue-tmin)*sfreq + 1)
times = sample_times/sfreq + tmin
times_cue = sample_times_cue/sfreq + tmin

# decoding performance during perception
all_perc = list()
# decoding performance trained during memory and tested during perception
all_perc_cross_mem = list()
# decoding performance during memory
all_mem = list()
# decoding performance trained during perception and tested during memory
all_mem_cross_perc = list()
for subject in subjects:
    mean_perc = list()
    mean_perc_cross_mem = list()
    mean_mem = list()
    mean_mem_cross_perc = list()
    # Loop across each analysis
    for analysis in ['left_sfreq', 'right_sfreq', 'left_angle', 'right_angle']:
        border = [0, 133, 0, 133]  # stimulus epoch time (train and test)
        fname = '%s_scores_%s.npy' % (subject, analysis)
        scores = np.load(op.join(path_data, 'results/', subject,
                                 results_folder, fname))
        where = times > 0
        mean_score = np.diag(scores)[where].mean(0)
        mean_perc.append(mean_score)
    mean_perc = np.array(mean_perc)  # Corresponds to mean decod during vision
    # Loop across each analysis
    for analysis in ['left_sfreq_cross_target_sfreq_cue_left_sfreq',
                     'right_sfreq_cross_target_sfreq_cue_right_sfreq',
                     'left_angle_cross_target_angle_cue_left_angle',
                     'right_angle_cross_target_angle_cue_right_angle']:
        border = [133, 314, 0, 133]  # cue epoch train time and stim test time
        fname = '%s_scores_%s.npy' % (subject, analysis)
        scores = np.load(op.join(path_data, 'results/', subject,
                                 results_folder, fname))
        mean_score = np.diag(scores[border[0]:border[1],
                                    border[2]:border[3]]).mean(0)
        mean_perc_cross_mem.append(mean_score)
    mean_perc_cross_mem = np.array(mean_perc_cross_mem)
    for analysis in ['target_angle_cue_left_angle',
                     'target_angle_cue_right_angle',
                     'target_sfreq_cue_left_sfreq',
                     'target_sfreq_cue_right_sfreq']:
        border = [133, 314, 133, 314]  # cue epoch time (train and test)
        fname = '%s_scores_%s.npy' % (subject, analysis)
        scores = np.load(op.join(path_data, 'results/', subject,
                                 results_folder, fname))
        where = times_cue > 0
        mean_score = np.diag(scores)[where].mean(0)
        mean_mem.append(mean_score)
    mean_mem = np.array(mean_mem)
    for analysis in ['target_sfreq_cue_left_sfreq_cross_left_sfreq',
                     'target_sfreq_cue_right_sfreq_cross_right_sfreq',
                     'target_angle_cue_left_angle_cross_left_angle',
                     'target_angle_cue_right_angle_cross_right_angle']:
        border = [0, 133, 133, 314]  # stim ep train time and cue ep test time
        fname = '%s_scores_%s.npy' % (subject, analysis)
        scores = np.load(op.join(path_data, 'results/', subject,
                                 results_folder, fname))
        mean_score = np.diag(scores[border[0]:border[1],
                                    border[2]:border[3]]).mean(0)
        mean_mem_cross_perc.append(mean_score)
    mean_mem_cross_perc = np.array(mean_mem_cross_perc)

    all_perc.append(mean_perc.mean())
    all_perc_cross_mem.append(mean_perc_cross_mem.mean())
    all_mem.append(mean_mem.mean())
    all_mem_cross_perc.append(mean_mem_cross_perc.mean())
all_perc = np.array(all_perc)
all_perc_cross_mem = np.array(all_perc_cross_mem)
all_mem = np.array(all_mem)
all_mem_cross_perc = np.array(all_mem_cross_perc)

# Open barplot figure
fig, ax = plt.subplots()
fig.set_size_inches(5, 4)
ind = [0, 2, 5, 7]
value = [all_perc.mean(0), all_perc_cross_mem.mean(0),
         all_mem.mean(0), all_mem_cross_perc.mean(0)]
std = [all_perc.std(0), all_perc_cross_mem.std(0),
       all_mem.std(0), all_mem_cross_perc.std(0)]
al, ca, cs, cl = plt.bar(ind, value, yerr=std, align='center',
                         color=['#ff7f0e', '#d62728', '#d62728', '#ff7f0e'],
                         error_kw=dict(ecolor='gray', lw=2, capsize=5,
                                       capthick=2))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_yaxis().tick_left()
ax.tick_params(
    axis='x',
    which='both',
    top='off',
    labelbottom='off')
ax.axhline(y=0, linewidth=1, color='black')
ax.set_yticks([0, 0.1])
ax.set_xticks(ind)
ax.set_xticklabels('off')
# Save figure
plt.tight_layout()
fname = op.join(path_data, 'fig5', 'cross_plot_bar.png')
plt.savefig(fname, transparent=True)
