async def async_bytes(payload):
    yield payload


async def train_request(client, query, payload):
    await client.post("http://localhost:8080/train",
                      params=query,
                      data=async_bytes(payload))


async def classify_request(client, payload):
    resp = await client.post("http://localhost:8080/classify",
                             data=async_bytes(payload))
    r = resp.json()
    return r['label']
