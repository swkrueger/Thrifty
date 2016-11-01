"""Data structures and utility functions for working with .toad(s) files."""

from collections import namedtuple

import numpy as np


CarrierSyncInfo = namedtuple('CarrierSyncInfo', [
    'bin',
    'offset',
    'energy',
    'noise'])


CorrDetectionInfo = namedtuple('CorrDetectionInfo', [
    'sample',
    'offset',
    'energy',
    'noise'])


class DetectionResult(object):
    """Collection of information stored in .toad(s) files."""
    def __init__(self, timestamp, block, soa, carrier_info, corr_info,
                 rxid=None, txid=None):
        """
        Parameters
        ----------
        timestamp: float
            Approximate time at which block has been captured.
        block: int
            Block index.
        soa: int
            Sample-of-arrival relative to the receiver's start time.
        carrier_info: CarrierSyncResult
        corr_info: CorrDetectionResult
        """
        # pylint: disable=too-many-arguments
        self.timestamp = timestamp
        self.block = block
        self.soa = soa
        self.carrier_info = carrier_info
        self.corr_info = corr_info
        self.rxid = rxid
        self.txid = txid

    def serialize(self):
        """Serialize detection as a string written to a .toad(s) file."""
        # pylint: disable=invalid-name
        corr, carr = self.corr_info, self.carrier_info
        s = ('{t:.6f} {b} {s:.8f} {ps} {po} {pe} {pn} '
             '{cb} {co} {ce} {cn}'.format(
             t=self.timestamp, b=self.block, s=self.soa,
             ps=corr.sample, po=corr.offset, pe=corr.energy, pn=corr.noise,
             cb=carr.bin, co=carr.offset, ce=carr.energy, cn=carr.noise))
        # Prepend TX and RX IDs
        if self.txid is not None:
            s = str(self.txid) + ' ' + s
        if self.rxid is not None:
            s = str(self.rxid) + ' ' + s
        return s

    @classmethod
    def deserialize(cls, string, with_rxid=False, with_txid=False):
        """Deserialize detection from string read from .toad(s) file."""
        # pylint: disable=invalid-name,too-many-locals
        fields = string.split()
 
        if len(fields) < 11 + with_rxid + with_txid:
            return None

        # Strip RX ID and TX ID
        rxid = int(fields.pop(0)) if with_rxid else None
        txid = int(fields.pop(0)) if with_txid else None

        t, b, s, ps, po, pe, pn, cb, co, ce, cn = map(float, fields)

        timestamp, block, soa = t, int(b), float(s)
        corr_info = CorrDetectionInfo(sample=int(ps), offset=po,
                                      energy=pe, noise=pn)
        carrier_info = CarrierSyncInfo(bin=int(cb), offset=co,
                                       energy=ce, noise=cn)

        return DetectionResult(timestamp=timestamp,
                               block=block,
                               soa=soa,
                               carrier_info=carrier_info,
                               corr_info=corr_info,
                               rxid=rxid,
                               txid=txid)


def _load_toads(stream, with_rxid=True, with_txid=True):
    """Load list of TOA detection serializations from a file."""
    toads = []
    if isinstance(stream, basestring):
        stream = open(stream, 'r')
    for i, line in enumerate(stream):
        if len(line) == 0 or line[0] == '#':
            continue

        detection = DetectionResult.deserialize(line,
                                                with_rxid=with_rxid,
                                                with_txid=with_txid)
        if detection is None:
            print("WARNING: skipped line #{}: "
                  "line's formatting is invalid".format(i+1))
            continue
        toads.append(detection)
    return toads


def load_toad(stream):
    """Load a single receiver's detection data from a .toad file."""
    return _load_toads(stream, True, False)


def load_toads(stream):
    """Load multiple receivers' detection data from a .toads file."""
    return _load_toads(stream, True, True)


def toads_array(detections, with_ids=True):
    """Create a structured nparray from an array of `DetectionResult`s."""
    data = [
        (
            i, line.rxid if with_ids else -1, line.txid if with_ids else -1,
            line.timestamp, line.block, line.soa,
            line.corr_info.sample, line.corr_info.offset,
            line.corr_info.energy, line.corr_info.noise,
            line.carrier_info.bin, line.carrier_info.offset,
            line.carrier_info.energy, line.carrier_info.noise
        )
        for i, line in enumerate(detections)
    ]
    return np.array(data, dtype=[
        ('idx', 'i4'), ('rxid', 'i4'), ('txid', 'i4'),
        ('timestamp', 'f8'), ('block', 'i4'), ('soa', 'f8'),
        ('sample', 'i4'), ('offset', 'f8'),
        ('energy', 'f8'), ('noise', 'f8'),
        ('carrier_bin', 'i4'), ('carrier_offset', 'f8'),
        ('carrier_energy', 'f8'), ('carrier_noise', 'f8')
    ])
