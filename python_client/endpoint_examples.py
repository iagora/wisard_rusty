import json

import requests

# set hyperparameters using a json
# { "hashtables":35,"addresses":21,
#   "bleach":0, "target_size":[28,28],
#   "mapping":[72,465,223,690,718,280,395,400,...]}
#
# Obs.: Target size and mapping are optional, and
# default to (28, 28) and random mapping respectively
with open('weights_u8_info.json', 'r') as json_file:
    payload = json.load(json_file)
r = requests.post("http://localhost:8080/new", json=payload)
print("Initilized the hyperparameters")

# Train an image with label
with open('zero.png', 'rb') as img:
    payload = img.read()
    query = {'label': '0'}
r = requests.post("http://localhost:8080/train", params=query, data=payload)
print("Trained image of a handwriten zero with label '0'")

# Classify image
with open('zero.png', 'rb') as img:
    payload = img.read()
r = requests.post("http://localhost:8080/classify", data=payload)
response = json.loads(r.content)
print("Classified image of a handwritten zero as a {}".format(
    response['label']))

# Get info
r = requests.get("http://localhost:8080/info")
response = json.loads(r.content)
print("Get info out of wisard")
print(response)  # if saved to a file, it's
# the same format of weights_u8_info.json
# which means you can recreate the exact mapping used

# Save model for future use
r = requests.get("http://localhost:8080/model", stream=True)
response = r.raw
with open('weights.bin.gz', 'wb') as f:
    f.write(response.read())
print("Saved the trained model to weights.bin.gz")

# Erase rams from wisard, all hyperparameters are back to default
r = requests.delete("http://localhost:8080/model")
print("Erased all the rams from wisard")

# Load model
with open('weights.bin.gz', 'rb') as f:
    payload = f.read()
r = requests.post("http://localhost:8080/model",
                  data=payload,
                  headers={'content-encoding': 'gzip'})
print("Load the model we just saved")

# Classify again
with open('zero.png', 'rb') as img:
    payload = img.read()
r = requests.post("http://localhost:8080/classify", data=payload)
response = json.loads(r.content)
print("Classified image of a handwritten zero as a {}".format(
    response['label']))

# Get info
r = requests.get("http://localhost:8080/info")
response = json.loads(r.content)
print("Get info out of wisard")
print(response)  # if saved to a file, it's
# the same format of weights_u8_info.json
# which means you can recreate the exact mapping used
