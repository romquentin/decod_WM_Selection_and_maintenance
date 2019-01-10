"""Run decoding analyses in time-frequency source space domain for the
working memory task and save decoding performance and pattern"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import numpy as np
import mne
from h5io import read_hdf5
import pandas as pd
from mne.decoding import SlidingEstimator, get_coef, LinearModel
from mne.forward import read_forward_solution
from mne.channels import read_dig_montage
from mne.minimum_norm import (make_inverse_operator)
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from jr.gat import (AngularRegression, scorer_spearman, scorer_auc,
                    scorer_angle)
from base import (complete_behavior, get_events_interactions, read_hpi_mri)
from config import path_data
from sklearn import preprocessing
from sklearn.metrics import make_scorer
import sys
subject = sys.argv[1]  # read a swarm file for parralel computing on biowulf
target_baseline = False  # If True baseline applied from target
freqs = np.array([10])
output_folder = '/review/time_frequency_in_source_%s/' % freqs[0]
# Define analyses
analyses_target = ['left_sfreq', 'right_sfreq', 'left_angle', 'right_angle']
analyses_cue = ['cue_side', 'cue_type', 'target_angle_cue_angle',
                'target_sfreq_cue_sfreq']
analyses = dict(Target=analyses_target,
                Cue=analyses_cue)


def load(subject, event_type):
    # Behavior
    fname = op.join(path_data, subject, 'behavior_%s.hdf5' % event_type)
    events = read_hdf5(fname)
    # add explicit conditions
    events = complete_behavior(events)

    # MEG
    if target_baseline:
        fname = op.join(path_data, subject, 'epochs_tf_%s.fif' % event_type) # noqa
    else:
        fname = op.join(path_data, subject, 'epochs_tf_%s_bsl.fif' % event_type) # noqa
    epochs = mne.read_epochs(fname)
    # epochs.decimate(10)
    return epochs, events


# define frequency range
# freqs = np.array([3, 6, 10, 16])
# freqs = np.arange(2, 60, 2)
n_cycles = freqs / 2.

# Define freesurfer, results and subject folder
freesurf_subject = subject
results_folder = op.join(path_data + 'results/' + subject + output_folder)
subject_path = op.join(path_data, subject)
subjects_dir = op.join(path_data, 'subjects')
freesurf_subject_path = op.join(subjects_dir, freesurf_subject)
# Define folder containing HPI position in MRI coordinates for registration
neuronav_path = op.join(subject_path, 'neuronav')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Compute noise covariance on Target epoch
if target_baseline:
    epochs, events = load(subject, 'Target')
    noise_cov = mne.compute_covariance(epochs, tmax=0)

for epoch_type, epoch_analyses in analyses.iteritems():
    epochs, events = load(subject, epoch_type)
    events = get_events_interactions(events)
    # read hpi position in device space
    hpi = list()
    idx = 0
    for this_hpi in epochs.info['hpi_results'][idx]['dig_points']:
        if this_hpi['kind'] == 1 or this_hpi['kind'] == 2:
            hpi.append(this_hpi['r'])
    hpi = np.array(hpi)
    # read hpi_mri.txt (hpi position in MRI coord from brainsight)
    hpi_fname = op.join(neuronav_path, 'hpi_mri_surf.txt')
    landmark = read_hpi_mri(hpi_fname)
    point_names = ['NEC', 'LEC', 'REC']  # Nasion, Left and Right electrodes
    elp = np.array([landmark[key] for key in point_names])
    # Set montage
    dig_montage = read_dig_montage(hsp=None, hpi=hpi, elp=elp,
                                   point_names=point_names, unit='mm',
                                   transform=False,
                                   dev_head_t=True)
    epochs.set_montage(dig_montage)
    # # Visually check the montage
    # plot_trans(epochs.info, trans=None, subject=freesurf_subject, dig=True,
    #            meg_sensors=True, subjects_dir=subjects_dir, brain=True)

    # Create or Read forward model
    fwd_fname = op.join(subject_path, '%s-fwd.fif' % subject)
    if not op.isfile(fwd_fname):
        bem_dir = op.join(freesurf_subject_path, 'bem')
        bem_sol_fname = op.join(bem_dir, freesurf_subject + '-5120-bem-sol.fif')
        src_fname = op.join(bem_dir, freesurf_subject + '-oct-6-src.fif')
        fwd = mne.make_forward_solution(
            info=epochs.info, trans=None, src=src_fname,
            bem=bem_sol_fname, meg=True, eeg=False, mindist=5.0)
        # Convert to surface orientation for better visualization
        fwd = mne.convert_forward_solution(fwd, surf_ori=True)
        # save
        mne.write_forward_solution(fwd_fname, fwd, overwrite=True)

    fwd = read_forward_solution(fwd_fname)

    # Setup inverse model
    epochs.pick_types(meg=True, ref_meg=False)
    # inv = make_inverse_operator(epochs.info, fwd, noise_cov,
    #                             loose=0.2, depth=0.8)
    method = 'beamformer'  # use of beamformer method
    # reconstruct source signal at the single trial
    data_cov = mne.compute_covariance(epochs, tmin=0.04)
    if not target_baseline:
        noise_cov = mne.compute_covariance(epochs, tmax=0)

    filters = make_lcmv(epochs.info, fwd, noise_cov=noise_cov,
                        data_cov=data_cov, reg=0.05,
                        pick_ori='max-power')
    stcs = apply_lcmv_epochs(epochs, filters, max_ori_out='signed')

    n_times = len(epochs.times)
    n_vertices = len(stcs[0].data)
    n_epochs = len(epochs.events)
    X_data = np.zeros([n_epochs, n_vertices, n_times])
    for jj, stc in enumerate(stcs):
            X_data[jj] = stc.data
    X = mne.time_frequency.tfr_array_morlet(X_data,
                                            sfreq=epochs.info['sfreq'],
                                            freqs=freqs,
                                            output='power',
                                            n_cycles=n_cycles,
                                            n_jobs=24)
    n_epochs, n_channels, n_freqs, n_times = X.shape
    X = X.reshape(n_epochs, n_channels, -1)  # collapse freqs and time
    # Run decoding for each analysis
    for analysis in epoch_analyses:
        # define to-be-predicted values
        y = np.array(events[analysis])
        if 'angle' in analysis[:14]:
            clf = make_pipeline(StandardScaler(),
                                LinearModel(AngularRegression(Ridge(),
                                                              independent=False)))
            scorer = scorer_angle
            kwargs = dict()
            y = np.array(y, dtype=float)
        elif 'sfreq' in analysis[:14]:
            clf = make_pipeline(StandardScaler(), LinearModel(Ridge()))
            scorer = scorer_spearman
            kwargs = dict()
            y = np.array(y, dtype=float)
        elif ('cue_side' in analysis or 'cue_type' in analysis):
            clf = make_pipeline(StandardScaler(),
                                LinearModel(LogisticRegression()))
            scorer = scorer_auc
            kwargs = dict()
            y[np.where(pd.isnull(y))] = 'NaN'
            le = preprocessing.LabelEncoder()
            le.fit(y)
            y = le.transform(y)
        # only consider non NaN values
        if ('cue_side' in analysis or 'cue_type' in analysis):
            sel = np.where(y != 0)[0]
        else:
            # When decoding memory at probe time, use only trial with
            # different probe compare to target
            if (epoch_type == 'Probe') & ('target' in analysis):
                sel = np.where((events['Change'] == 1) & (~np.isnan(y)))[0]
            else:
                sel = np.where(~np.isnan(y))[0]
        td = SlidingEstimator(clf, scoring=make_scorer(scorer),
                              n_jobs=24, **kwargs)
        # run decoding
        cv = StratifiedKFold(8)
        scores = list()
        patterns = list()
        filters = list()
        for train, test in cv.split(X[sel], y[sel]):
            td.fit(X[sel][train], y[sel][train])
            score = td.score(X[sel][test], y[sel][test])
            scores.append(score)
            patterns.append(get_coef(td, 'patterns_', inverse_transform=True))
            filters.append(get_coef(td, 'filters_', inverse_transform=True))
        scores = np.mean(scores, axis=0)
        patterns = np.mean(patterns, axis=0)
        filters = np.mean(filters, axis=0)
        if 'angle' in analysis:
            patterns = np.mean(np.abs(patterns), axis=1)
            filters = np.mean(np.abs(filters), axis=1)
        scores = np.reshape(scores, (n_freqs, n_times))
        patterns = np.reshape(patterns, (n_channels, n_freqs, n_times))
        filters = np.reshape(filters, (n_channels, n_freqs, n_times))
        # save cross-validated scores
        fname = results_folder +\
            '%s_tf_scores_%s_%s.npy' % (subject, epoch_type, analysis)
        np.save(fname, np.array(scores))
        fname = results_folder +\
            '%s_tf_patterns_%s_%s.npy' % (subject, epoch_type, analysis)
        np.save(fname, np.array(patterns))
        fname = results_folder +\
            '%s_tf_filters_%s_%s.npy' % (subject, epoch_type, analysis)
        np.save(fname, np.array(filters))
