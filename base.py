""" Generic functions """

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import os
import os.path as op
from nose.tools import assert_true
from numpy.testing.utils import assert_array_equal, assert_array_almost_equal
from pandas import DataFrame as df


def get_events_from_mat(fname_behavior):
    """Read CSV file and output pandas.DataFrame object"""
    # Read events from CSV file
    events = np.genfromtxt(fname_behavior, dtype=float, skip_header=1,
                           delimiter=',', names=True)
    labels = (
        'NbTrial', 'FixNbTrial', 'isFixed', 'GaborLeft', 'GaborRight',
        'sfLeft', 'orientLeft', 'phaseLeft', 'sfRight', 'orientLeft',
        'phaseRight', 'TrialTime', 'runningTime', 'fixcrossTime',
        'gaborTime', 'postgaborTime', 'triggGabor'
    )
    event_data = list()
    for event in events:
        event_dict = {key: value for (key, value) in zip(event, labels)}
        event_data.append(event_dict)
    events_behavior = df(events)
    # define block
    events_behavior['block'] = np.floor(events_behavior['FixNbTrial'] / 50.01)
    events_behavior['triggTarget'] = events_behavior['triggGabor']
    return events_behavior


def check_triggers(events_meg, events_behavior):
    """Test whether the trigger found in the MEG match those of Matlab"""
    from nose.tools import assert_true
    # check that number of trigger is a multiple of 3
    assert_true(len(events_meg) % 3 == 0)

    # check that all triggers follow the pattern target, cue, probe
    events_target = range(1, 126)
    events_cue = range(126, 130)
    events_probe = range(130, 140)
    for ii in range(len(events_meg) / 3):
        assert_true(events_meg[ii * 3 + 0, 2] in events_target)
        assert_true(events_meg[ii * 3 + 1, 2] in events_cue)
        assert_true(events_meg[ii * 3 + 2, 2] in events_probe)

    # now we can select only the triggers corresponding to the target
    events_meg = events_meg[::3, :]

    # check that it corresponds to the mat file
    n_trial = len(events_meg)
    np.testing.assert_array_equal(events_meg[:, 2],
                                  events_behavior.triggGabor[:n_trial])


def read_raw_triggers(fname_raw):
    import mne
    from mne.io import Raw
    # read data
    raw = Raw(fname_raw, preload=True)
    # find trigger channel
    channel_trigger = np.where(np.array(raw.ch_names) == 'USPT001')[0][0]
    # find where the trigger == 255
    trigger_baseline = np.where(raw._data[channel_trigger, :] == 255)[0]
    # replace 255 values with 0
    raw._data[channel_trigger, trigger_baseline] = 0.
    # find correct triggers
    events_meg = mne.find_events(raw)
    return raw, events_meg


def fix_triggers(events_meg, events_behavior, event_type):

    """ Use this function when the triggers are not identical between the meg
    and the behavioral file output by Matlab.
    """
    from nose.tools import assert_true
    from Levenshtein import editops
    # copy because can change data in place
    events_meg = np.copy(events_meg)
    events_behavior = events_behavior.copy()
    # initialize new field in behavioral file
    n_trials = len(events_behavior)
    events_type_labels = ['triggTarget', 'triggCue', 'triggProbe']
    for label in events_type_labels:
        events_behavior[label + '_missing'] = np.zeros(n_trials, bool)
    # Add a column to notice if the trial is complete or no
    events_behavior['missing'] = np.zeros(n_trials, bool)
    # concatenate all behavioral events into one long series of triggers
    events_behavior_triggers = np.reshape(np.vstack((
        events_behavior.triggTarget,
        events_behavior.triggCue,
        events_behavior.triggProbe)).T, [-1])

    # Identify missed, exchanged or additional trigger values in behavioral
    # file as compared to the meg trigger

    def int_to_unicode(array):
        return ''.join([str(chr(int(ii))) for ii in array])

    changes = editops(int_to_unicode(events_behavior_triggers),
                      int_to_unicode(events_meg[:, 2]))

    # for each modification
    mod = 0
    print changes
    for modification, from_trigger, _ in changes:
        if modification == 'delete':
            mod = 1
            this_trial = np.floor(from_trigger / 3.)
            this_event_type = int(from_trigger % 3.)
            # set False value to trigg[Type]_missing
            this_key = events_type_labels[this_event_type] + '_missing'
            events_behavior.set_value(this_trial, this_key, True)
            # report True if at least one trigger is missing in the trial
            events_behavior.set_value(this_trial, 'missing', True)
        else:
            # TODO: implement other types of deletion, replacement etc error
            NotImplementedError()
            # TODO: remove or add elements in events_meg
    events_behavior['trial'] = range(len(events_behavior))

    # delete all trials with at least one trigger absent
    sel = np.where(events_behavior['missing'] == False)[0]  # noqa
    # print sel
    events_behavior = events_behavior.iloc[sel]
    events_behavior.reset_index()

    # ---- remove incomplete trials in events_meg
    # concatenate again all behavioral events into one long series of triggers
    # to compare it with meg_events
    if mod == 1:
        events_behavior_triggers = np.reshape(np.vstack((
            events_behavior.triggTarget,
            events_behavior.triggCue,
            events_behavior.triggProbe)).T, [-1])
        # check a second time changes after deletion of uncomplete behavioral
        # trials
        changes = editops(int_to_unicode(events_meg[:, 2]),
                          int_to_unicode(events_behavior_triggers))

        print changes
        delete = np.zeros(len(changes))
        nbdelete = 0
        for modification, from_trigger, _ in changes:
            if modification == 'delete':
                delete[nbdelete] = from_trigger
                nbdelete += 1
        events_meg = np.delete(events_meg, delete, 0)
    # Check if each trial in events_meg has the good succession of trigger
    for ntri in range(0, (len(events_meg)), 3):
        assert_true(events_meg[ntri, 3] == 1)
        assert_true(events_meg[ntri+1, 3] == 2)
        assert_true(events_meg[ntri+2, 3] == 3)

    # Returns specific types of events (Target, Cue or Probe)
    start = np.where([event_type == ii for ii in events_type_labels])[0][0]
    events_meg = events_meg[start::3, :]

    # check that same number of trials in MEG and behavior
    assert_true(len(events_meg) == len(events_behavior))

    events_behavior['meg_event_tsample'] = events_meg[:, 0]
    events_behavior['meg_file'] = events_meg[:, 1]
    events_behavior['meg_event_value'] = events_meg[:, 2]
    events_behavior = events_behavior.reset_index()
    return events_behavior


def complete_behavior(events):

    only_fix_trial = 1
    only_fixandcorrect = 0

    """ This is to add some information on each trials to make them more
    explicit.
    """
    events.reset_index(inplace=True)  # to be removed
    # XXX FIXME : systematic naming ----------------
    # initialize new variables
    for new_key in ['cue_side', 'cue_type', 'target_angleidx',
                    'target_sfreq', 'distr_angleidx',
                    'distr_sfreq', 'probe_sfreq',
                    'probe_angleidx']:
        events[new_key] = [[]] * len(events)

    replace = dict(
        left_angleidx='orientLeft',
        right_angleidx='orientRight',
        left_phaseidx='phaseLeft',
        right_phaseidx='phaseRight',
        probe_phaseidx='phaseResp',
        left_sfreq='sfLeft',
        right_sfreq='sfRight',
        is_correct='isCorrect',
        is_eye_fixed='isFixed',
        reaction_time='reactionTime')

    for new, old in replace.iteritems():
        events[new] = events[old]
        events.drop(old, axis=1)

    # ---------------------------------------------

    for trial in range(len(events)):
        event = events.iloc[trial]

        # FIXME : missing Probe information -----------
        # Get probe attribute: the probe can be congruent to the target
        # depending on the cue. This is what the key 'Change' indicates.
        cue_index = event['Cue']
        angle = dict(left=event['left_angleidx'],
                     right=event['right_angleidx'])
        sfreq = dict(left=event['left_sfreq'],
                     right=event['right_sfreq'])
        phase = dict(left=event['left_phaseidx'],
                     right=event['right_phaseidx'])

        tmp = dict()
        tmp['cue_side'] = 'left' if cue_index in [1, 2] else 'right'
        tmp['cue_type'] = 'angle' if cue_index in [2, 4] else 'sfreq'

        if tmp['cue_type'] == 'angle':
            tmp['probe_sfreq'] = event['randomSF']
            if event['Change']:  # probe incongruent to target
                tmp['probe_angleidx'] = event['randomOrient']
            else:
                tmp['probe_angleidx'] = angle[tmp['cue_side']]

        elif tmp['cue_type'] == 'sfreq':
            tmp['probe_angleidx'] = event['randomOrient']
            if event['Change']:  # probe incongruent to target
                tmp['probe_sfreq'] = event['randomSF']
            else:
                tmp['probe_sfreq'] = sfreq[tmp['cue_side']]

        else:
            RuntimeError()
        tmp['probe_phaseidx'] = event['probe_phaseidx']

        # Add missing values in temporary dictionary
        tmp['target_angleidx'] = angle[tmp['cue_side']]
        tmp['target_sfreq'] = sfreq[tmp['cue_side']]
        tmp['target_phaseidx'] = phase[tmp['cue_side']]

        noncue_side = 'left' if tmp['cue_side'] == 'right' else 'right'
        tmp['distr_angleidx'] = angle[noncue_side]
        tmp['distr_sfreq'] = sfreq[noncue_side]
        tmp['distr_phaseidx'] = phase[noncue_side]

        # Save target, distr and probe values in events
        for stim in ['target', 'distr', 'probe']:
            for value in ['angleidx', 'sfreq', 'phaseidx']:
                key = stim + '_' + value
                events.set_value(trial, key, tmp[key])
        for rule in ['side', 'type']:
            key = 'cue_' + rule
            events.set_value(trial, key, tmp[key])

    # for naming consistency

    def index_to_rad(key):
        idx = np.array(np.copy(events[key].as_matrix()), float)
        angles = np.deg2rad(idx * 36. * 2.)
        return angles

    # Orientation indices to radians
    for stim in ['left', 'right', 'target', 'distr', 'probe']:
        # XXX angles, unlike phases are coded as 1, 2, ... 5
        idx = np.array(np.copy(events[stim + '_angleidx'].as_matrix()), float)
        events[stim + '_angle'] = np.deg2rad(idx * 36. * 2.)
        assert_array_almost_equal(np.unique(events[stim + '_angle']),
                                  np.linspace(0, 2 * np.pi, 6)[1:])

    # Phase "indices" to radians
    for stim in ['left', 'right', 'target', 'distr', 'probe']:
        # XXX phases, unlike angles, are coded as 0., 0.2, ... 0.8
        idx = np.array(np.copy(events[stim + '_phaseidx'].as_matrix()), float)
        events[stim + '_phase'] = np.deg2rad((idx + .2) * 5. * 36. * 2)
        assert_array_almost_equal(np.unique(events[stim + '_phase']),
                                  np.linspace(0, 2 * np.pi, 6)[1:])

    # Cleanup non necessary keys
    keep = ['index', 'block',
            'reaction_time', 'is_correct', 'is_eye_fixed',
            'meg_event_tsample', 'meg_event_value', 'meg_file', 'Change']
    keep += ['cue_side', 'cue_type']
    for stim in ['left', 'right', 'target', 'distr', 'probe']:
        for value in ['angle', 'angleidx', 'sfreq', 'phase', 'phaseidx']:
            keep += [stim + '_' + value]

    events = events[keep]

    # Analysing only fix trials or only fix and correct trials
    if only_fix_trial == 1:
        sel = np.where(events['is_eye_fixed'] == 0)[0]
        for column in keep[8:]:
            events[column][sel] = np.nan
    elif only_fixandcorrect == 1:
        sel = np.where((events['is_eye_fixed'] == 0) |
                       (events['is_correct'] == 0))[0]
        for column in keep[8:]:
            events[column][sel] = np.nan

    return events


def get_events_interactions(events):
    """Add subconditions for easy analyses"""
    # divide total trials number by 2 (800/2.=400 trials)
    subconditions = dict(side=['left', 'right'], type=['angle', 'sfreq'])
    for stim in ['target', 'distr']:
        for value in ['angle', 'sfreq']:
            for cue, subs in subconditions.iteritems():
                for this_sub in subs:
                    # copy information:
                    # e.g. target_angle_cue_angle = target_angle
                    ori_key = '%s_%s' % (stim, value)
                    new_key = '%s_%s_cue_%s' % (stim, value, this_sub)
                    events[new_key] = events[ori_key]
                    # NaN when cue subcondition isn't met
                    sel = np.where(events['cue_%s' % cue] != this_sub)[0]
                    events[new_key][sel] = np.nan
    # divide total trials number by 4 (800/4.=200 trials)
    subconditions = dict(type=['angle', 'sfreq'])
    for stim in ['target', 'distr']:
        for value in ['left', 'right']:
            for cue, subs in subconditions.iteritems():
                for this_sub in subs:
                    # copy information:
                    # e.g. target_angle_cue_left_angle = target_angle_cue_left
                    ori_key = '%s_%s_cue_%s' % (stim, this_sub, value)
                    new_key = '%s_%s_cue_%s_%s' % (stim, this_sub, value, this_sub)
                    events[new_key] = events[ori_key]
                    # NaN when cue subcondition isn't met
                    sel = np.where(events['cue_%s' % cue] != this_sub)[0]
                    events[new_key][sel] = np.nan
    assert_true('target_angle_cue_angle' in events.keys())
    assert_array_equal(
        np.where(np.isnan(events['target_angle_cue_angle']))[0],
        np.where(events['cue_type'] != 'angle')[0])
    assert_true('target_angle_cue_left_angle' in events.keys())
    assert_array_equal(
        np.where(np.isnan(events['target_angle_cue_left_angle']))[0],
        np.where(((events['cue_type'] != 'angle')) | (events['cue_side'] != 'left'))[0])
    return events


# Define here analyses of interest
analyses_target = ['right_angle', 'left_angle', 'right_sfreq', 'left_sfreq']
analyses_cue = ['target_angle', 'target_sfreq', 'distr_angle', 'distr_sfreq',
                'cue_side', 'cue_type']
analyses_probe = ['probe_angle', 'probe_sfreq']
interaction_analyses = ['target_angle_cue_angle', 'target_angle_cue_sfreq',
                        'target_sfreq_cue_angle', 'target_sfreq_cue_sfreq']
analyses = dict(Target=analyses_target,
                Cue=analyses_cue,
                Probe=analyses_cue + analyses_probe)


def read_headshape(fname):
    f = open(fname)
    text = [line.strip('\n') for line in f.readlines()]
    f.close()
    idx = 0
    hsp = np.empty([0, 3])
    while idx < len(text):
        line = text[idx]
        if line[:6] == 'Sample':
            _, _, _, x, y, z = line.split('\t')[:6]
            hsp = np.vstack((hsp, [float(x), float(y), float(z)]))
        idx += 1
    return hsp


def read_hpi_mri(fname):
    landmark = dict()
    f = open(fname)
    text = [l.strip('\n') for l in f.readlines()]
    f.close()
    idx = 0
    while idx < len(text):
        line = text[idx]
        if line[:4] in ('NEC\t', 'LEC\t', 'REC\t'):
            code, _, _, x, y, z = line.split('\t')[:6]
            landmark[code] = [float(x), float(y), float(z)]
        if line[:5] in 'le\tSe':
            _, _, x, y, z = line.split('\t')[:5]
            landmark['lpa'] = [float(x), float(y), float(z)]
        elif line[:5] in 're\tSe':
            _, _, x, y, z = line.split('\t')[:5]
            landmark['rpa'] = [float(x), float(y), float(z)]
        elif line[:5] in 'rn\tSe':
            _, _, x, y, z = line.split('\t')[:5]
            landmark['nasion'] = [float(x), float(y), float(z)]
        idx += 1
    return landmark


def check_freesurfer(subjects_dir, subject):  # from jr.meg
    # Check freesurfer finished without any errors
    fname = op.join(subjects_dir, subject, 'scripts', 'recon-all.log')
    if op.isfile(fname):
        with open(fname, 'rb') as fh:
            fh.seek(-1024, 2)
            last = fh.readlines()[-1].decode()
        print last
        print('{}: ok'.format(subject))
        return True
    else:
        print('{}: missing'.format(subject))
        return False


def check_libraries():  # from jr.meg
    """Raise explicit error if mne and freesurfer or mne c are not installed"""
    from mne.utils import has_mne_c, has_freesurfer
    import subprocess
    if not (has_freesurfer() and has_mne_c() and
            op.isfile(subprocess.check_output(['which', 'freesurfer'])[:-1])):
        # export FREESURFER_HOME=/usr/local/freesurfer
        # source $FREESURFER_HOME/SetUpFreeSurfer.sh
        # export MNE_ROOT=/home/jrking/MNE-2.7.4-3452-Linux-x86_64
        # source $MNE_ROOT/bin/mne_setup_sh
        # export LD_LIBRARY_PATH=/home/jrking/anaconda/lib/
        raise('Check your freesurfer and mne c paths')


def create_bem_surf(subject, subjects_dir=None, overwrite=False):  # from jr.meg # noqa
    # from mne.bem import make_watershed_bem
    # from mne.commands.mne_make_scalp_surfaces import _run as make_scalp_surface
    from mne.utils import get_config
    if subjects_dir is None:
        subjects_dir = get_config('SUBJECTS_DIR')

    # Set file name ----------------------------------------------------------
    bem_dir = op.join(subjects_dir, subject, 'bem')
    src_fname = op.join(bem_dir, subject + '-oct-6-src.fif')
    bem_fname = op.join(bem_dir, subject + '-5120-bem.fif')
    bem_sol_fname = op.join(bem_dir, subject + '-5120-bem-sol.fif')

    # Skip make_watershed_bem and make_scalp_surface because it is
    # already done from bash shell (freesurfer command)
    miss_surface_copy = False
    for surface in ['inner_skull', 'outer_skull', 'outer_skin']:
        fname = op.join(bem_dir, '%s.surf' % surface)
        if not op.isfile(fname):
            miss_surface_copy = True
    if overwrite or miss_surface_copy:
        for surface in ['inner_skull', 'outer_skull', 'outer_skin']:
            from shutil import copyfile
            from_file = op.join(bem_dir,
                                'watershed/%s_%s_surface' % (subject, surface))
            to_file = op.join(bem_dir, '%s.surf' % surface)
            if op.exists(to_file):
                os.remove(to_file)
            copyfile(from_file, to_file)

    # 3. Setup source space
    if overwrite or not op.isfile(src_fname):
        from mne import setup_source_space
        check_libraries()
        files = ['lh.white', 'rh.white', 'lh.sphere', 'rh.sphere']
        for fname in files:
            if not op.exists(op.join(subjects_dir, subject, 'surf', fname)):
                raise RuntimeError('missing: %s' % fname)

        setup_source_space(subject=subject, subjects_dir=subjects_dir,
                           fname=src_fname,
                           spacing='oct6', surface='white', overwrite=True,
                           add_dist=True, n_jobs=-1, verbose=None)

    # 4. Prepare BEM model
    if overwrite or not op.exists(bem_sol_fname):
        from mne.bem import (make_bem_model, write_bem_surfaces,
                             make_bem_solution, write_bem_solution)
        check_libraries()
        # run with a single layer model (enough for MEG data)
        surfs = make_bem_model(subject, conductivity=[0.3],
                               subjects_dir=subjects_dir)
        # surfs = make_bem_model(subject=subject, subjects_dir=subjects_dir)
        write_bem_surfaces(fname=bem_fname, surfs=surfs)
        bem = make_bem_solution(surfs)
        write_bem_solution(fname=bem_sol_fname, bem=bem)


def decod_stats(X):
    from mne.stats import permutation_cluster_1samp_test
    """Statistical test applied across subjects"""
    # check input
    X = np.array(X)

    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = permutation_cluster_1samp_test(
        X, out_type='mask', n_permutations=2**12, n_jobs=6,
        verbose=False)

    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster] = pval

    return np.squeeze(p_values_)


def gat_stats(X):
    from mne.stats import spatio_temporal_cluster_1samp_test
    """Statistical test applied across subjects"""
    # check input
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X

    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask',
        n_permutations=2**12, n_jobs=-1, verbose=False)

    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval

    return np.squeeze(p_values_).T


def class_angle(y):
    a = np.where((y > 0) & (y < 1.3))
    y[a] = 1
    a = np.where((y > 2.2) & (y < 2.7))
    y[a] = 2
    a = np.where((y > 3.5) & (y < 3.9))
    y[a] = 3
    a = np.where((y > 4.9) & (y < 5.1))
    y[a] = 4
    a = np.where((y > 6.1) & (y < 6.4))
    y[a] = 5
    return y
