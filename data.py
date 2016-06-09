#!/usr/bin/env python

'''
Common data structures and utility functions for loading / storing data.
'''

from collections import namedtuple
import numpy as np

# s/Result/Info

# TODO: class description (detection within a block), field descriptions
CarrierSyncResult = namedtuple('CarrierSyncResult', [
    'bin',
    'offset',
    'energy',
    'noise'])

# TODO: s/toa/? peak?
ToaDetectionResult = namedtuple('ToaDetectionResult', [
    'sample',
    'offset',
    'energy',
    'noise'])


class DetectionResult:
    # (timestamp, block, toa_detection, carrier_sync)
    def __init__(self, timestamp, block, carrier, toa, rxid=None, txid=None):
        '''
        Args:
            timestamp: approximate time at which block was captured
            block: block index
            carrier: instance of CarrierSyncResult
            toa: instance of ToaDetectionResult
        '''
        self.timestamp = timestamp
        self.block = block
        self.carrier = carrier
        self.toa = toa
        self.rxid = rxid
        self.txid = txid


    def serialize(self):
        toa, c = self.toa, self.carrier
        s = '{t:.6f} {b} {ts} {to} {te} {tn} {cb} {co} {ce} {cn}'.format(
                t=self.timestamp, b=self.block,
                ts=toa.sample, to=toa.offset, te=toa.energy, tn=toa.noise,
                cb=c.bin, co=c.offset, ce=c.energy, cn=c.noise)

        # Prepend TX and RX IDs
        if self.txid != None:
            s = str(self.txid) + ' ' + s
        if self.rxid != None:
            s = str(self.rxid) + ' ' + s

        return s


    @classmethod
    def deserialize(cls, s, with_rxid=False, with_txid=False):
        fields = s.split()

        # Strip RX ID and TX ID
        rxid = int(fields.pop(0)) if with_rxid else None
        txid = int(fields.pop(0)) if with_txid else None

        t, b, ts, to, te, tn, cb, co, ce, cn = map(float, fields)

        timestamp, block = t, int(b)
        toa = ToaDetectionResult(sample=int(ts), offset=to, energy=te, noise=tn)
        carrier = CarrierSyncResult(bin=int(cb), offset=co, energy=ce, noise=cn)

        return DetectionResult(timestamp, block, carrier, toa, rxid, txid)


def _load_toads(stream, with_rxid=True, with_txid=True):
    '''Load list of TOA detection serializations from stream'''
    toads = []

    for l in stream:
        if len(l) == 0 or l[0] == '#':
            continue
        detection = DetectionResult.deserialize(
                l, with_rxid=with_rxid, with_txid=with_txid)
        toads.append(detection)

    return toads


def load_toad(stream):
    return _load_toads(stream, False, False)


def load_toads(stream):
    return _load_toads(stream, True, True)


def toads_array(toads, with_ids=True):
    '''Create nparray from ToaDetectionResult serializations'''
    data = [(i, t.rxid if with_ids else -1, t.txid if with_ids else -1,
        t.timestamp, t.block,
        t.toa.sample, t.toa.offset,
        t.toa.energy, t.toa.noise,
        t.carrier.bin, t.carrier.offset,
        t.carrier.energy, t.carrier.noise)
        for i, t in enumerate(toads)]
    return np.array(data,
        dtype=[('idx', i4), ('rxid', 'i4'), ('txid', 'i4'),
            ('timestamp', 'f8'), ('block', 'i4'),
            ('sample', 'i4'), ('offset', 'f8'),
            ('energy', 'f8'), ('noise', 'f8'),
            ('carrier_bin', 'i4'), ('carrier_offset', 'f8'),
            ('carrier_energy', 'f8'), ('carrier_noise', 'f8')])

