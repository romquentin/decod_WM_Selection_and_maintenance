"""
Figure 5
Plot time generalization across memory and visual conditions computed from
run_decoding_WM_across_epochs_and_conditions.py
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
from webcolors import hex_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define colors
colors = ['#1f77b4', '#d62728', '#ff7f0e']

title_size = 12
legend_size = 16
ticks_size = 12
asterisk_size = 24
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.labelsize'] = ticks_size
plt.rcParams['ytick.labelsize'] = ticks_size
# Define pair of cross analyses
analyses = {'Tar_csf_cr_sf': ['target_sfreq_cue_left_sfreq_cross_left_sfreq',
                              'target_sfreq_cue_right_sfreq_cross_right_sfreq',
                              0.1, colors[1]],
            'Sf_cr_tar_csf': ['left_sfreq_cross_target_sfreq_cue_left_sfreq',
                              'right_sfreq_cross_target_sfreq_cue_right_sfreq',
                              0.1, colors[2]],
            'Tar_can_cr_an': ['target_angle_cue_left_angle_cross_left_angle',
                              'target_angle_cue_right_angle_cross_right_angle',
                              0.1, colors[1]],
            'An_cr_tar_can': ['left_angle_cross_target_angle_cue_left_angle',
                              'right_angle_cross_target_angle_cue_right_angle',
                              0.1, colors[2]]
            }
# Define results to plot
results_folder = 'sensors_accross_epochs_and_conditions_Kfold7'
# Define times
sfreq = 120
tmin = -.2
tmin_cue = 0.
tmax = .9
tmax_cue = 1.5
sample_times = np.linspace(0, (tmax-tmin)*sfreq, (tmax-tmin)*sfreq + 1)
sample_times_cue = np.linspace(0, (tmax_cue-tmin_cue)*sfreq,
                               (tmax_cue-tmin_cue)*sfreq + 1)
times = sample_times/sfreq + tmin
times_cue = sample_times_cue/sfreq + tmin_cue
chance = 0
# Loop across each pair of analyses
for analysis, sub_analysis in analyses.iteritems():
    all_scores = list()
    for subject in subjects:
        fname0 = '%s_scores_%s.npy' % (subject, sub_analysis[0])
        scores0 = np.load(op.join(path_data, 'results/', subject,
                                  results_folder, fname0))
        fname1 = '%s_scores_%s.npy' % (subject, sub_analysis[1])
        scores1 = np.load(op.join(path_data, 'results/', subject,
                                  results_folder, fname1))
        scores = (scores0 + scores1)/2.  # Mean cue left and cue right
        all_scores.append(scores)
    all_scores = np.array(all_scores)

    ymax = sub_analysis[2]
    color = sub_analysis[3]
    color = np.array(hex_to_rgb(color))/255.
    color = np.concatenate((color, [1]), axis=0)
    cmap = deepcopy(plt.get_cmap('magma_r'))
    cmap.colors = np.c_[np.linspace(1, color[0], 256),
                        np.linspace(1, color[1], 256),
                        np.linspace(1, color[2], 256)]

    if 'Tar' in analysis[:4]:
        borders = [[133, 314, 133, 314], [133, 314, 0, 133],
                   [0, 133, 133, 314]]
    else:
        borders = [[0, 133, 0, 133], [133, 314, 0, 133],
                   [0, 133, 133, 314]]
    # Separate epoch times
    for num, border in enumerate(borders):
        all_scores_sub = all_scores[:, border[0]:border[1],
                                    border[2]:border[3]]

        gat_p_values = gat_stats(np.array(all_scores_sub))
        sig = np.array(gat_p_values < 0.05)

        if all_scores_sub.shape[1] == 133:
            y_times = times
            y_size = 3
        else:
            y_times = times_cue
            y_size = 4.09
        if all_scores_sub.shape[2] == 133:
            x_times = times
            x_size = 3
        else:
            x_times = times_cue
            x_size = 4.09
        # Plot mean subjects
        fig_mean, axes = plt.subplots()
        fig_mean.set_size_inches(x_size, y_size)
        imshow = axes.imshow(np.mean((all_scores_sub), axis=0), origin='lower',
                             cmap=cmap,
                             extent=[x_times[0], x_times[-1], y_times[0], y_times[-1]], vmin=0, vmax=ymax)
        axes_divider = make_axes_locatable(axes)
        cax = axes_divider.append_axes("top", size="7%", pad="2%")
        cbar = plt.colorbar(imshow, cax=cax, ticks=[0, ymax], orientation="horizontal")
        cax.xaxis.set_ticks_position('top')
        cbar.ax.set_xticklabels(['', ymax])
        if 'Tar' in analysis[:4]:
            if num == 0:
                axes.set_xticks(np.linspace(0, 1.4, 8))
                axes.tick_params(
                    axis='y',
                    which='both',
                    labelleft='off')
                axes.fill_between(times_cue, times_cue[0], times_cue[-1],
                                  where=(times_cue >= 0) & (times_cue <= 0.1),
                                  alpha=0.2,
                                  color='gray')
                axes.fill_betweenx(times_cue, times_cue[0], times_cue[-1],
                                   where=(times_cue >= 0) & (times_cue <= 0.1),
                                   alpha=0.2,
                                   color='gray')
            elif num == 1:
                axes.set_xticks(np.linspace(-.2, 0.8, 6))
                axes.set_yticks(np.linspace(0, 1.4, 8))
                axes.fill_between(times_cue, times_cue[0], times_cue[-1],
                                  where=(times_cue >= 0) & (times_cue <= 0.1),
                                  alpha=0.2,
                                  color='gray')
                axes.fill_betweenx(times, times[0], times[-1],
                                   where=(times >= 0) & (times <= 0.1),
                                   alpha=0.2,
                                   color='gray')
            elif num == 2:
                axes.set_xticks(np.linspace(0, 1.4, 8))
                axes.tick_params(
                    axis='y',
                    which='both',
                    labelleft='off')
                axes.fill_between(times, times[0], times[-1],
                                  where=(times >= 0) & (times <= 0.1),
                                  alpha=0.2,
                                  color='gray')
                axes.fill_betweenx(times_cue, times_cue[0], times_cue[-1],
                                   where=(times_cue >= 0) & (times_cue <= 0.1),
                                   alpha=0.2,
                                   color='gray')
        else:
            if num == 0:
                axes.set_xticks(np.linspace(-.2, 0.8, 6))
                axes.set_yticks(np.linspace(-.2, 0.8, 6))
                axes.fill_between(times, times[0], times[-1],
                                  where=(times >= 0) & (times <= 0.1),
                                  alpha=0.2,
                                  color='gray')
                axes.fill_betweenx(times, times[0], times[-1],
                                   where=(times >= 0) & (times <= 0.1),
                                   alpha=0.2,
                                   color='gray')
            elif num == 1:
                axes.set_xticks(np.linspace(-.2, 0.8, 6))
                axes.set_yticks(np.linspace(0, 1.4, 8))
                axes.fill_between(times_cue, times_cue[0], times_cue[-1],
                                  where=(times_cue >= 0) & (times_cue <= 0.1),
                                  alpha=0.2,
                                  color='gray')
                axes.fill_betweenx(times, times[0], times[-1],
                                   where=(times >= 0) & (times <= 0.1),
                                   alpha=0.2,
                                   color='gray')
            elif num == 2:
                axes.set_xticks(np.linspace(0, 1.4, 8))
                axes.tick_params(
                    axis='y',
                    which='both',
                    labelleft='off')
                axes.fill_between(times, times[0], times[-1],
                                  where=(times >= 0) & (times <= 0.1),
                                  alpha=0.2,
                                  color='gray')
                axes.fill_betweenx(times_cue, times_cue[0], times_cue[-1],
                                   where=(times_cue >= 0) & (times_cue <= 0.1),
                                   alpha=0.2,
                                   color='gray')

        xx, yy = np.meshgrid(x_times, y_times,
                             copy=False, indexing='xy')
        axes.contour(xx, yy, sig, colors='Gray', levels=[0],
                    linestyles='solid')
        # Save cross analyses figure
        plt.tight_layout()
        fname = op.join(path_data, 'fig_supp/fig_supp_5/', analysis
                        + str(num) + '.png')
        plt.savefig(fname, transparent=True)

# # Define pair of analyses
analyses = {'target_sfreq': ['target_sfreq_cue_left_sfreq',
                             'target_sfreq_cue_right_sfreq',
                             0.1, colors[1]],
            'stim_sfreq': ['left_sfreq',
                           'right_sfreq',
                           0.4, colors[2]],
            'target_angle': ['target_angle_cue_left_angle',
                             'target_angle_cue_right_angle',
                             0.1, colors[1]],
            'stim_angle': ['left_angle',
                           'right_angle',
                           0.1, colors[2]]
            }
# Define results to plot
results_folder = 'sensors_accross_epochs_and_conditions'
# Loop across each pair of analyses
for analysis, sub_analysis in analyses.iteritems():
    all_scores = list()
    for subject in subjects:
        fname0 = '%s_scores_%s.npy' % (subject, sub_analysis[0])
        scores0 = np.load(op.join(path_data, 'results/', subject,
                                  results_folder, fname0))
        fname1 = '%s_scores_%s.npy' % (subject, sub_analysis[1])
        scores1 = np.load(op.join(path_data, 'results/', subject,
                                  results_folder, fname1))
        scores = (scores0 + scores1)/2.  # Mean cue left and cue right
        all_scores.append(scores)
    all_scores = np.array(all_scores)

    ymax = sub_analysis[2]
    color = sub_analysis[3]
    color = np.array(hex_to_rgb(color))/255.
    color = np.concatenate((color, [1]), axis=0)
    cmap = deepcopy(plt.get_cmap('magma_r'))
    cmap.colors = np.c_[np.linspace(1, color[0], 256),
                        np.linspace(1, color[1], 256),
                        np.linspace(1, color[2], 256)]
    if 'tar' in analysis[:4]:
        borders = [133, 314, 133, 314]
        x_times = y_times = times_cue
        x_size = y_size = 4.09
    else:
        borders = [0, 133, 0, 133]
        x_times = y_times = times
        x_size = y_size = 3
    all_scores_sub = all_scores[:, borders[0]:borders[1], borders[2]:borders[3]]
    gat_p_values = gat_stats(np.array(all_scores_sub))
    sig = np.array(gat_p_values < 0.05)
    # Plot mean subjects
    fig_mean, axes = plt.subplots()
    fig_mean.set_size_inches(x_size, y_size)
    imshow = axes.imshow(np.mean((all_scores_sub), axis=0), origin='lower',
                         cmap=cmap,
                         extent=[x_times[0], x_times[-1], y_times[0], y_times[-1]], vmin=0, vmax=ymax)
    axes_divider = make_axes_locatable(axes)
    cax = axes_divider.append_axes("top", size="7%", pad="2%")
    cbar = plt.colorbar(imshow, cax=cax, ticks=[ymax], orientation="horizontal")
    cax.xaxis.set_ticks_position('top')
    cbar.ax.set_yticklabels([ymax])

    if 'tar' in analysis[:4]:
        axes.set_xticks(np.linspace(0, 1.4, 8))
        axes.tick_params(
            axis='y',
            which='both',
            labelleft='off')
        axes.fill_between(times_cue, times_cue[0], times_cue[-1],
                          where=(times_cue >= 0) & (times_cue <= 0.1),
                          alpha=0.2,
                          color='gray',
                          interpolate=True)
        axes.fill_betweenx(times_cue, times_cue[0], times_cue[-1],
                           where=(times_cue >= 0) & (times_cue <= 0.1),
                           alpha=0.2,
                           color='gray')
    else:
        axes.set_xticks(np.linspace(-.2, 0.8, 6))
        axes.set_yticks(np.linspace(-.2, 0.8, 6))
        axes.fill_between(times, times[0], times[-1],
                          where=(times >= 0) & (times <= 0.1),
                          alpha=0.2,
                          color='gray',
                          interpolate=True)
        axes.fill_betweenx(times, times[0], times[-1],
                           where=(times >= 0) & (times <= 0.1),
                           alpha=0.2,
                           color='gray')
    xx, yy = np.meshgrid(x_times, y_times,
                         copy=False, indexing='xy')
    axes.contour(xx, yy, sig, colors='Gray', levels=[0],
                linestyles='solid')
    # Save figure for non-crossed analysis
    plt.tight_layout()
    fname = op.join(path_data, 'fig_supp/fig_supp_5/', analysis
                    + '.png')
    plt.savefig(fname, transparent=True)
