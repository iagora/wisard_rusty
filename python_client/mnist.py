import gzip

import numpy as np


def load_mnist(folder, prefix):
    data_buffer = gzip.open(folder + prefix + '-images-idx3-ubyte.gz')
    data_buffer = data_buffer.read()
    int_type = np.dtype('int32').newbyteorder('>')
    metadata_bytes = 4 * int_type.itemsize

    data = np.frombuffer(data_buffer, dtype='ubyte')
    magic_bytes, n_images, width, height = np.frombuffer(
        data[:metadata_bytes].tobytes(), int_type)
    data = data[metadata_bytes:].astype(dtype='ubyte').reshape(
        [n_images, width * height])

    label_buffer = gzip.open(folder + prefix + '-labels-idx1-ubyte.gz')
    label_buffer = label_buffer.read()
    labels = np.frombuffer(label_buffer, dtype='ubyte')[2 * int_type.itemsize:]

    return data, labels
