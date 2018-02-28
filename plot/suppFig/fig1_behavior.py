"""
Supplementary Figure 1
Plot behavioral performance
"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import numpy as np
from h5io import read_hdf5
from base import complete_behavior
from config import subjects, path_data
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

all_correct = list()
cueangle_correct = list()
cuesfreq_correct = list()
cueleft_correct = list()
cueright_correct = list()

for subject in subjects:
    # Read behav file (hdf5)
    print '**********' + subject + '************'
    fname = op.join(path_data, subject, 'behavior_target.hdf5')
    events = read_hdf5(fname)
    events = complete_behavior(events)
    # Select behav perf on all trials
    isfixed = np.where(events['is_eye_fixed'] == 1)
    iscorrect = np.array(events['is_correct'])
    iscorrect_fixed = iscorrect[isfixed]
    if len(iscorrect_fixed) != 800:
        warnings.warn("Total isfixed trial is not 800")
        print 'total is:' + str(len(iscorrect_fixed))
    perc = sum(iscorrect_fixed)/len(iscorrect_fixed)
    all_correct.append(perc)

    # behav perf only cue angle
    cue_angle = np.where(events['cue_type'] == 'angle')
    if len(cue_angle[0]) != 400:
        warnings.warn("Total trial with cue angle is not 400")
        print 'total is:' + str(len(cue_angle[0]))
    iscorrect_cue_angle = iscorrect[cue_angle]
    perc = sum(iscorrect_cue_angle)/len(iscorrect_cue_angle)
    cueangle_correct.append(perc)

    # behav perf only cue sf
    cue_sfreq = np.where(events['cue_type'] == 'sfreq')
    if len(cue_sfreq[0]) != 400:
        warnings.warn("Total trial with cue sfreq is not 400")
        print 'total is: ' + str(len(cue_sfreq[0]))
    iscorrect_cue_sfreq = iscorrect[cue_sfreq]
    perc = sum(iscorrect_cue_sfreq)/len(iscorrect_cue_sfreq)
    cuesfreq_correct.append(perc)

    # behav perf only cue left
    cue_left = np.where(events['cue_side'] == 'left')
    if len(cue_left[0]) != 400:
        warnings.warn("Total trial with cue left is not 400")
        print 'total is:' + str(len(cue_left[0]))
    iscorrect_cue_left = iscorrect[cue_left]
    perc = sum(iscorrect_cue_left)/len(iscorrect_cue_left)
    cueleft_correct.append(perc)

    # behav perf only cue left
    cue_right = np.where(events['cue_side'] == 'right')
    if len(cue_right[0]) != 400:
        warnings.warn("Total trial with cue right is not 400")
        print 'total is:' + str(len(cue_right[0]))
    iscorrect_cue_right = iscorrect[cue_right]
    perc = sum(iscorrect_cue_right)/len(iscorrect_cue_right)
    cueright_correct.append(perc)

all_correct = np.array(all_correct)
cueangle_correct = np.array(cueangle_correct)
cuesfreq_correct = np.array(cuesfreq_correct)
cueleft_correct = np.array(cueleft_correct)
cueright_correct = np.array(cueright_correct)

# Plot figure


def to_percent(y, position):
    s = str(100 * y)
    return s[:-2] + '%'


fig, ax = plt.subplots()
ind = [0, 2, 3, 5, 6]
value = [all_correct.mean(0), cueangle_correct.mean(0),
         cuesfreq_correct.mean(0), cueleft_correct.mean(0),
         cueright_correct.mean(0)]
std = [all_correct.std(0), cueangle_correct.std(0),
       cuesfreq_correct.std(0), cueleft_correct.std(0),
       cueright_correct.std(0)]
al, ca, cs, cl, cr = plt.bar(ind, value, yerr=std, align='center',
                             color=['gray', 'silver', 'silver', 'silver',
                                    'silver'],
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

ax.set_xticks(ind)
ax.set_xticklabels('off')
ax.set_ylim(0.5)
ax.set_yticks(np.linspace(0.5, 1, 6))
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
# Save figure
fname = op.join(path_data, 'fig_supp/fig_supp_1', 'behav.png')
plt.savefig(fname, transparent=True)
