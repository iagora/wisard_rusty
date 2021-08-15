import asyncio
import time

import httpx

from async_requests import classify_request, train_request
from mnist import load_mnist


async def main():

    print("Pythonesque WiSARD Client - MNIST 🐍🐍🐍")
    async with httpx.AsyncClient() as client:
        # Get info
        r = await client.get("http://localhost:8080/info")
        wis_info = r.json()

        print("Number of hashtables: {}".format(wis_info['hashtables']))
        print("Address size: {}".format(wis_info['addresses']))
        print("Bleaching: {}".format(wis_info['bleach']))

    print("\n-----------------\nTraining\n-----------------")
    start_time = time.time()
    training_images, training_labels = load_mnist("data/mnist/", "train")
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
                asyncio.ensure_future(train_request(client, query, payload)))
        await asyncio.gather(*tasks)
    print("Training took: {:.0f} milliseconds".format(
        1000 * (time.time() - start_time)))

    print("-----------------\nTesting\n-----------------")
    start_time = time.time()
    test_images, test_labels = load_mnist("data/mnist/", "t10k")
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
            tasks.append(
                asyncio.ensure_future(classify_request(client, payload)))
        responses = await asyncio.gather(*tasks)
        for (response, label) in zip(responses, test_labels):
            if response == str(label):
                hit = hit + 1
            count = count + 1

    print("Testing took: {:.0f} milliseconds".format(
        1000 * (time.time() - start_time)))

    print("Accuracy: {:.4f}".format(hit / count))


asyncio.run(main())
