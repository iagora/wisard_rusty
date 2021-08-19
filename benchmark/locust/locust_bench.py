import gzip
import random

import numpy as np

from locust import HttpUser, between, task

FOLDER = "data/mnist/"


class WisardUser(HttpUser):
    wait_time = between(0.1, 0.2)

    def load_mnist(self, folder, prefix):
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
        labels = np.frombuffer(label_buffer,
                               dtype='ubyte')[2 * int_type.itemsize:]

        return list(data), list(labels)

    def on_start(self):
        self.data_train, self.labels_train = self.load_mnist(FOLDER, "train")
        self.data_test, self.labels_test = self.load_mnist(FOLDER, "t10k")
        rnd_index = random.randint(0, len(self.labels_train))
        self.client.post("http://localhost:8080/train",
                         params={'label': str(self.labels_train[rnd_index])},
                         data=bytes(self.data_train[rnd_index]))

    @task(6)
    def train(self):
        rnd_index = random.randint(0, len(self.labels_train))
        self.client.post("http://localhost:8080/train",
                         params={'label': str(self.labels_train[rnd_index])},
                         data=bytes(self.data_train[rnd_index]))

    @task
    def classify(self):
        rnd_index = random.randint(0, len(self.labels_test))
        self.client.post("http://localhost:8080/classify",
                         data=bytes(self.data_test[rnd_index]))
