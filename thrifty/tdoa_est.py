#!/usr/bin/env python

"""
Estimate TDOA values of mobile unit transmissions from SOA values.

A TDOA estimator that uses beacon transmissions to synchronise SOAs values from
different receivers and to estimate TDOA values of mobile unit detections
relative to beacon detections.
"""

from __future__ import division
from __future__ import print_function

from bisect import bisect_left, bisect_right
import collections
import itertools

import numpy as np

from thrifty import matchmaker
from thrifty import toads_data
from thrifty.settings import parse_kvconfig

SPEED_OF_LIGHT = 2.997e8
MAX_TDOA = 30e3 / SPEED_OF_LIGHT


TdoaInfo = collections.namedtuple('TdoaInfo', [
    'rx0', 'rx1', 'tdoa', 'snr', 'model_quality', 'det0_idx', 'det1_idx'])

TdoaGroup = collections.namedtuple('TdoaGroup', [
    'group_id', 'timestamp', 'tx', 'tdoas'])

TDOA_DTYPE = {'names': ('rx0', 'rx1', 'tdoa', 'snr', 'model_quality',
                        'det0_idx', 'det1_idx'),
              'formats': ('i4', 'i4', 'f8', 'f8', 'f8', 'i4', 'i4')}

MATRIX_DTYPE = {'names': ('group_id', 'timestamp', 'tx') + TDOA_DTYPE['names'],
                'formats': ('i4', 'f8', 'i4') + TDOA_DTYPE['formats']}


def make_detection_extractor(detections, matches):
    rxpair_detections = collections.defaultdict(list)
    for group in matches:
        for det0_id, det1_id in itertools.combinations(group, 2):
            det0 = detections[det0_id]
            det1 = detections[det1_id]
            if det0.rxid > det1.rxid:
                det0, det1 = det1, det0
            rxpair_detections[(det0.rxid, det1.rxid)].append((det0, det1))

    timestamps = {}
    for pair, detections in rxpair_detections.iteritems():
        detections.sort(cmp=lambda x, y: x[0].timestamp < y[0].timestamp)
        timestamps[pair] = [d[0].timestamp for d in detections]

    def extract(rxid0, rxid1, timestamp_start, timestamp_stop):
        assert rxid0 < rxid1
        pair = (rxid0, rxid1)
        left = bisect_left(timestamps[pair], timestamp_start)
        right = bisect_right(timestamps[pair], timestamp_stop)
        detection_pairs = rxpair_detections[pair][left:right]
        return detection_pairs

    return extract


def estimate_model_quality(model, detection_pairs):
    # TODO: estimate model quality from SNR and/or residuals and/or model
    # covariance matrix and/or model residuals.
    # Alternative names to replace "quality": confidence / beacon SNR
    sqrt_snr0 = np.array([d[0].corr_info.energy / d[0].corr_info.noise
                          for d in detection_pairs])
    sqrt_snr1 = np.array([d[1].corr_info.energy / d[1].corr_info.noise
                          for d in detection_pairs])
    snr = (np.mean(sqrt_snr0**2) + np.mean(sqrt_snr1**2)) / 2

    return snr


def build_model_poly(detection_pairs, beacon_sdoa, nominal_sample_rate, deg=2):
    if len(detection_pairs) < deg + 1:
        # not enough beacon transmissions
        return None

    soa0 = np.array([d[0].soa for d in detection_pairs])
    soa1 = np.array([d[1].soa for d in detection_pairs])
    soa1at0 = soa1 + np.array(beacon_sdoa)
    coef = np.polyfit(soa1at0, soa0, deg)
    fit = np.poly1d(coef)
    # residuals = soa0 - fit(soa1at0)
    # print(np.mean(residuals))

    def evaluate(det0, det1):
        return (det0.soa - fit(det1.soa)) / nominal_sample_rate

    return evaluate


def find_nearest_value(list_, value):
    idx = bisect_left(list_, value)
    if idx > 0 and (idx == len(list_) or
                    abs(value - list_[idx-1]) < abs(value - list_[idx])):
        return idx - 1
    else:
        return idx


def test_find_nearest_value():
    list_ = [5, 10, 15]
    values = [4, 5, 6, 9, 10, 11, 14, 16]
    expected_output = [0, 0, 0, 1, 1, 1, 2, 2]
    nearest = [find_nearest_value(list_, v) for v in values]
    np.testing.assert_equal(nearest, expected_output)


def build_model_nearest(detection_pairs, beacon_sdoa, nominal_sample_rate):
    if len(detection_pairs) < 1:
        # not enough beacon transmissions
        return None

    pairs = sorted(detection_pairs,
                   cmp=lambda x, y: x[0].timestamp < y[0].timestamp)
    timestamps = [p[0] for p in pairs]

    def evaluate(det0, det1):
        idx = find_nearest_value(timestamps, det0.timestamp)
        dsoa0 = det0.soa - pairs[idx][0].soa
        dsoa1 = det1.soa - pairs[idx][1].soa
        return (dsoa0 - dsoa1 + beacon_sdoa[idx]) / nominal_sample_rate

    return evaluate


# default model
build_model = build_model_poly

# TODO: More models:
#  _nearest
#  _linear
#  _quad (_poly)
#  _weighted_quad (weighted on SNR and time from mobile unit TX)


def _dist(vector1, vector2):
    diff = np.array(vector1) - np.array(vector2)
    return np.sqrt(np.sum(diff**2))


def estimate_tdoas(detections, matches, window_size, beacon_pos, rx_pos,
                   sample_rate, model_builder=build_model, model_params=None):
    if model_params is None:
        model_params = {}

    beacon_matches = [m for m in matches
                      if detections[m[0]].txid in beacon_pos]
    mobile_matches = [(i, m) for i, m in enumerate(matches)
                      if detections[m[0]].txid not in beacon_pos]

    def _beacon_tdoa(rxid0, rxid1, beaconid):
        return (_dist(rx_pos[rxid0], beacon_pos[beaconid]) -
                _dist(rx_pos[rxid1], beacon_pos[beaconid])) / SPEED_OF_LIGHT

    tdoa_groups = []
    failures = []
    extractor = make_detection_extractor(detections, beacon_matches)
    for group_idx, group in mobile_matches:
        tdoas = []
        group_timestamp = detections[group[0]].timestamp
        for det0_id, det1_id in itertools.combinations(group, 2):
            if detections[det0_id].rxid > detections[det1_id].rxid:
                det0_id, det1_id = det1_id, det0_id

            det0, det1 = detections[det0_id], detections[det1_id]

            window_start, window_stop = (det0.timestamp - window_size,
                                         det0.timestamp + window_size)
            beacon_pairs = extractor(det0.rxid, det1.rxid,
                                     window_start, window_stop)
            beacon_tdoa = [_beacon_tdoa(det0.rxid, det1.rxid, b[0].txid)
                           for b in beacon_pairs]
            beacon_sdoa = np.array(beacon_tdoa) * sample_rate

            model = model_builder(beacon_pairs,
                                  beacon_sdoa,
                                  sample_rate,
                                  **model_params)
            if model is None:
                failures.append((det0_id, det1_id))
                continue
            model_quality = estimate_model_quality(model, beacon_pairs)
            tdoa = model(det0, det1)

            # Ignore outliers
            if abs(tdoa) >= MAX_TDOA:
                failures.append((det0_id, det1_id))
                continue

            snr0 = (det0.corr_info.energy / det0.corr_info.noise)**2
            snr1 = (det1.corr_info.energy / det1.corr_info.noise)**2
            snr = (snr0 + snr1) / 2

            tdoas.append(TdoaInfo(rx0=det0.rxid,
                                  rx1=det1.rxid,
                                  tdoa=tdoa,
                                  snr=snr,
                                  model_quality=model_quality,
                                  det0_idx=det0_id,
                                  det1_idx=det1_id))

        if len(tdoas) > 0:
            tdoas_array = np.array(tdoas, dtype=TDOA_DTYPE)
            tdoa_groups.append(TdoaGroup(group_id=group_idx,
                                         timestamp=group_timestamp,
                                         tx=det0.txid,
                                         tdoas=tdoas_array))

    return tdoa_groups, failures


def save_tdoa_groups(output, tdoa_groups):
    for group in tdoa_groups:
        for tdoa in group.tdoas:
            tdoa = tdoa.copy()
            tdoa['tdoa'] *= 1e9
            print(group.group_id, "%.06f" % group.timestamp, group.tx,
                  *tdoa, file=output)


def load_tdoa_matrix(fname):
    data = np.loadtxt(fname, dtype=MATRIX_DTYPE)
    data['tdoa'] /= 1e9
    return data


def groups_to_matrix(groups):
    data = []
    for group in groups:
        group_info = (group.group_id, group.timestamp, group.tx)
        for tdoa in group.tdoas:
            row = group_info + tdoa.tolist()
            data.append(row)
    return np.array(data, dtype=MATRIX_DTYPE)


def load_tdoa_groups(fname):
    matrix = load_tdoa_matrix(fname)
    tdoa_groups = collections.OrderedDict()
    for row in matrix:
        group_id = row['group_id']
        fields = list(TDOA_DTYPE['names'])
        if group_id not in tdoa_groups:
            in_group = matrix['group_id'] == group_id
            tdoa_groups[group_id] = TdoaGroup(group_id=group_id,
                                              timestamp=row['timestamp'],
                                              tx=row['tx'],
                                              tdoas=matrix[fields][in_group])
    return tdoa_groups.values()


def load_pos_config(file_):
    strings = parse_kvconfig(file_)
    txfreqs = {int(id_): np.array([float(x) for x in pos_str.split()])
               for id_, pos_str in strings.iteritems()}
    return txfreqs


def _main():
    import argparse

    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)

    parser.add_argument('toads', nargs='?',
                        type=argparse.FileType('r'), default='data.toads',
                        help="toads data (\"-\" streams from stdin)")
    parser.add_argument('matches', nargs='?',
                        type=argparse.FileType('r'), default='data.match',
                        help="match data (\"-\" streams from stdin)")
    parser.add_argument('-o', '--output', dest='output',
                        type=argparse.FileType('w'), default='data.tdoa',
                        help="output file (\'-\' for stdout)")
    parser.add_argument('-r', '--rx-coordinates', dest='rx_pos',
                        type=argparse.FileType('r'), default='pos-rx.cfg',
                        help="path to config file that contains the "
                             "coordinates of the receivers")
    parser.add_argument('-b', '--beacon-coordinates', dest='beacon_pos',
                        type=argparse.FileType('r'), default='pos-beacon.cfg',
                        help="path to config file that contains the "
                             "coordinates of the beacon transmitters")
    parser.add_argument('-w', '--window-size', dest='window_size',
                        type=float, default=8,
                        help="maximum difference in timestamp between a beacon"
                             " transmission and a mobile unit transmission for"
                             " the beacon transmission to be eligible to be"
                             " used for estimating the TDOA of the mobile unit"
                             " transmission")
    parser.add_argument('-s', '--sample-rate', dest='sample_rate',
                        type=float, default=2.4e6,
                        help="nominal sample rate of receivers")
    args = parser.parse_args()

    toads = toads_data.load_toads(args.toads)
    matches = matchmaker.load_matches(args.matches)
    rx_pos = load_pos_config(args.rx_pos)
    beacon_pos = load_pos_config(args.beacon_pos)
    tdoa_groups, failures = estimate_tdoas(toads, matches, args.window_size,
                                           beacon_pos, rx_pos,
                                           args.sample_rate)

    print("Number of TDOA estimations:", len(tdoa_groups))
    print("Number of TDOA estimation failures:", len(failures))
    save_tdoa_groups(args.output, tdoa_groups)


if __name__ == '__main__':
    _main()
