import asyncio
import gzip
import json
import time
from io import BytesIO

import httpx
import numpy as np


def load_mnist(prefix, folder):
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


async def async_bytes(payload):
    yield payload


async def train_image(client, query, payload):
    await client.post("http://localhost:8080/train",
                      params=query,
                      data=async_bytes(payload))


async def classify_image(client, payload):
    resp = await client.post("http://localhost:8080/classify",
                             data=async_bytes(payload))
    r = resp.json()
    return r['label']


async def main():

    print("Pythonesque WiSARD - MNIST 🐍🐍🐍")
    async with httpx.AsyncClient() as client:
        # Get info
        r = await client.get("http://localhost:8080/info")
        wis_info = r.json()

        print("Number of hashtables: {}".format(wis_info['hashtables']))
        print("Address size: {}".format(wis_info['addresses']))
        print("Bleaching: {}".format(wis_info['bleach']))

    print("\n-----------------\nTraining\n-----------------")
    start_time = time.time()
    training_images, training_labels = load_mnist("train", "data/mnist/")
    print("Training data has {} images".format(len(training_labels)))
    print("Parsing the training dataset took: {:.0f} milliseconds".format(
        1000 * (time.time() - start_time)))

    start_time = time.time()
    async with httpx.AsyncClient() as client:
        tasks = []
        for (img, label) in zip(training_images, training_labels):
            payload = bytes(img)
            query = {'label': str(label)}
            tasks.append(
                asyncio.ensure_future(train_image(client, query, payload)))
        await asyncio.gather(*tasks)
    print("Training took: {:.0f} milliseconds".format(
        1000 * (time.time() - start_time)))

    print("-----------------\nTesting\n-----------------")
    start_time = time.time()
    test_images, test_labels = load_mnist("t10k", "data/mnist/")
    print("Testing data has {} images".format(len(test_labels)))
    print("Parsing the test dataset took: {:.0f} milliseconds".format(
        1000 * (time.time() - start_time)))

    hit = 0
    count = 0
    start_time = time.time()
    async with httpx.AsyncClient() as client:
        tasks = []
        for img in test_images:
            payload = bytes(img)
            tasks.append(asyncio.ensure_future(classify_image(client,
                                                              payload)))
        responses = await asyncio.gather(*tasks)
        for (response, label) in zip(responses, test_labels):
            if response == str(label):
                hit = hit + 1
            count = count + 1

    print("Testing took: {:.0f} milliseconds".format(
        1000 * (time.time() - start_time)))

    print("Accuracy: {:.4f}".format(hit / count))


asyncio.run(main())
