"""Read/write numeric NPY arrays without requiring NumPy or allowing pickle."""

import array
import ast
import math
import struct
import sys


def read_npy(stream):
    if stream.read(6) != b'\x93NUMPY':
        raise ValueError('Not a NumPy NPY file.')
    version = stream.read(2)
    if version == b'\x01\x00':
        length = struct.unpack('<H', stream.read(2))[0]
    elif version in (b'\x02\x00', b'\x03\x00'):
        length = struct.unpack('<I', stream.read(4))[0]
    else:
        raise ValueError(f'Unsupported NPY version {version!r}.')
    if length > 1024 * 1024:
        raise ValueError('NPY header is unexpectedly large.')
    header = ast.literal_eval(stream.read(length).decode('utf-8' if version[0] == 3 else 'latin1'))
    dtype, shape = header['descr'], header['shape']
    if header['fortran_order']:
        raise ValueError('Only C-order arrays are supported.')
    if not isinstance(shape, tuple) or any(type(d) is not int or d < 0 for d in shape):
        raise ValueError('Invalid NPY shape.')
    codes = {'i4': 'i', 'i8': 'q', 'f4': 'f', 'f8': 'd'}
    if not isinstance(dtype, str) or dtype[0] not in '<>=|' or dtype[1:] not in codes:
        raise ValueError(f'Unsupported NPY dtype {dtype!r}; expected a numeric integer or float array.')
    result = array.array(codes[dtype[1:]])
    size = math.prod(shape) * result.itemsize
    data = stream.read(size + 1)
    if len(data) != size:
        raise ValueError('NPY payload size does not match its shape.')
    result.frombytes(data)
    endian = '<' if sys.byteorder == 'little' else '>'
    if dtype[0] in '<>' and dtype[0] != endian:
        result.byteswap()
    return result, shape


def write_vector(filename, values):
    """Write a one-dimensional little-endian int32 NPY file."""
    data = array.array('i', values)
    if sys.byteorder != 'little':
        data.byteswap()
    header = repr({'descr': '<i4', 'fortran_order': False, 'shape': (len(data),)})
    padding = (64 - ((10 + len(header) + 1) % 64)) % 64
    encoded = (header + ' ' * padding + '\n').encode('latin1')
    with open(filename, 'wb') as stream:
        stream.write(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(encoded)) + encoded)
        data.tofile(stream)
