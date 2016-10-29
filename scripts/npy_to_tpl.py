#!/usr/bin/env python
# TODO: don't use native endianness, but standardize on either little
#       or big endian
# TODO: use struct to pack data explicitly
#       (e.g. http://stackoverflow.com/a/31307217)

from __future__ import print_function

import os
import sys
import numpy as np

source = 'template.npy' if len(sys.argv) < 2 else sys.argv[1]
dest = (os.path.splitext(source)[0] + '.tpl'
        if len(sys.argv) < 3 else sys.argv[2])
print("{} -> {}".format(source, dest))

dest_file = open(dest, 'wb')

template = np.load(source)
np.int16(len(template)).tofile(dest_file)
template.astype(np.float32).tofile(dest_file)
# print(template)
# print(len(template))
