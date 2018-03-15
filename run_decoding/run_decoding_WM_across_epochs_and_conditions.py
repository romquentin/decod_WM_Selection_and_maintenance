"""Run decoding analyses in sensors space accross memory content and
visual perception for the working memory task and save decoding performance"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import numpy as np
import mne
from h5io import read_hdf5
from mne.decoding import GeneralizingEstimator, LinearModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from jr.gat import (AngularRegression, scorer_spearman,
                    scorer_angle)
from base import (complete_behavior, get_events_interactions)
from config import path_data
import sys
subject = sys.argv[1]  # read a swarm file for parralel computing on biowulf

output_folder = '/sensors_accross_epochs_and_conditions/'
# Create result folder
results_folder = op.join(path_data + 'results/' + subject + output_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# read behavior
fname = op.join(path_data, subject, 'behavior_Target.hdf5')
events = read_hdf5(fname)
events = complete_behavior(events)
events = get_events_interactions(events)
# read stimulus epochs
fname = op.join(path_data, subject, 'epochs_Target.fif')
epochs_target = mne.read_epochs(fname)
epochs_target.pick_types(meg=True, ref_meg=False)
epochs_target.crop(-0.2, 0.9)
# read cue epochs
fname = op.join(path_data, subject, 'epochs_Cue.fif')
epochs_cue = mne.read_epochs(fname)
epochs_cue.pick_types(meg=True, ref_meg=False)
epochs_cue.crop(0, 1.5)
# read probe epochs
fname = op.join(path_data, subject, 'epochs_Probe.fif')
epochs_probe = mne.read_epochs(fname)
epochs_probe.pick_types(meg=True, ref_meg=False)
epochs_probe.crop(0, 0.9)
# Concatenate the data of the three epochs
X0 = epochs_target._data
X1 = epochs_cue._data
X2 = epochs_probe._data
X = np.concatenate((X0, X1, X2), axis=2)

# Define pair of analyses (train on the 2nd and test on the 1st )
paired_analyses = [['target_sfreq_cue_left_sfreq', 'left_sfreq'],
                   ['target_sfreq_cue_right_sfreq', 'right_sfreq'],
                   ['left_sfreq', 'target_sfreq_cue_left_sfreq'],
                   ['right_sfreq', 'target_sfreq_cue_right_sfreq'],
                   ['target_angle_cue_left_angle', 'left_angle'],
                   ['target_angle_cue_right_angle', 'right_angle'],
                   ['left_angle', 'target_angle_cue_left_angle'],
                   ['right_angle', 'target_angle_cue_right_angle']]
# Loop across each pair of analyses
for paired_analysis in paired_analyses:
    y_test = np.array(events[paired_analysis[0]])
    y_train = np.array(events[paired_analysis[1]])
    # Define estimators depending on the analysis
    if 'angle' in paired_analysis[0][:14]:
        clf = make_pipeline(StandardScaler(),
                            LinearModel(AngularRegression(Ridge(),
                                                          independent=False)))
        scorer = scorer_angle
        kwargs = dict()
        gat = GeneralizingEstimator(clf, scoring=make_scorer(scorer),
                                    n_jobs=24, **kwargs)
        y_test = np.array(y_test, dtype=float)
        y_train = np.array(y_train, dtype=float)
    elif 'sfreq' in paired_analysis[0][:14]:
        clf = make_pipeline(StandardScaler(), LinearModel(Ridge()))
        scorer = scorer_spearman
        kwargs = dict()
        gat = GeneralizingEstimator(clf, scoring=make_scorer(scorer),
                                    n_jobs=24, **kwargs)
        y_test = np.array(y_test, dtype=float)
        y_train = np.array(y_train, dtype=float)
    # only consider trials with correct fixation
    sel = np.where(events['is_eye_fixed'] == 1)[0]
    y_train = y_train[sel]
    y_test = y_test[sel]
    X = np.concatenate((X0, X1, X2), axis=2)
    X = X[sel]
    # only consider non NaN values
    # Run decoding accross condition
    cv = StratifiedKFold(7)
    scores = list()
    scs = list()
    if np.isnan(y_train).any():
        sel = np.where(~np.isnan(y_train))[0]
        for train, test in cv.split(X[sel], y_train[sel]):
            gat.fit(X[sel][train], y_train[sel][train])
            score = gat.score(X[sel][test], y_test[sel][test])
            sc = gat.score(X[sel][test], y_train[sel][test])  # test on same
            scores.append(score)
            scs.append(sc)
        scores = np.mean(scores, axis=0)
        scs = np.mean(scs, axis=0)
    else:
        for train, test in cv.split(X, y_train):
            y_te = y_test[test]
            X_te = X[test]
            y_te = y_te[np.where(~np.isnan(y_te))[0]]
            X_te = X_te[np.where(~np.isnan(y_te))[0]]
            y_tr = y_train[train]
            X_tr = X[train]
            y_tr = y_tr[np.where(~np.isnan(y_tr))[0]]
            X_tr = X_tr[np.where(~np.isnan(y_tr))[0]]
            y_tr_te = y_train[test]
            X_tr_te = X[test]
            y_tr_te = y_tr_te[np.where(~np.isnan(y_tr_te))[0]]
            X_tr_te = X_tr_te[np.where(~np.isnan(y_tr_te))[0]]
            gat.fit(X_tr, y_tr)
            score = gat.score(X_te, y_te)
            sc = gat.score(X_tr_te, y_tr_te)   # test on same
            scores.append(score)
            scs.append(sc)
        scores = np.mean(scores, axis=0)
        scs = np.mean(scs, axis=0)

    # save cross-validated scores
    fname = results_folder +\
        '%s_scores_%s_cross_%s.npy' % (subject,
                                       paired_analysis[0],
                                       paired_analysis[1])
    np.save(fname, np.array(scores))  # save accross condition scores
    fname = results_folder +\
        '%s_scores_%s.npy' % (subject, paired_analysis[1])
    np.save(fname, np.array(scs))  # save scores test/train on same condition
