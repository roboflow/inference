import asyncio
import base64
import os

import aiohttp
import requests
from PIL import Image

PORT = 9001
API_KEY = os.environ["API_KEY"]
IMAGE_PATH = "image.jpg"


def encode_bas64(image_path):
    with open(image_path, "rb") as image:
        x = image.read()
        image_string = base64.b64encode(x)

    return image_string.decode("ascii")


async def do_cog_request(session):
    api_key = API_KEY
    prompt = (
        "The player on the left's name is Moky."
        " What round of the tournament is he in? Answer in one word."
    )

    print(f"Starting")
    infer_payload = {
        "image": {
            "type": "base64",
            "value": encode_bas64(IMAGE_PATH),
        },
        "api_key": api_key,
        "prompt": prompt,
    }
    async with session.post(
        f"http://localhost:{PORT}/llm/cogvlm",
        json=infer_payload,
    ) as response:
        if response.status != 200:
            print(response.status)
            print(await response.json())
            raise RuntimeError
        resp = await response.json()
        res = resp["response"]
        print(resp)
    infer_payload = {
        "image": {
            "type": "base64",
            "value": encode_bas64(IMAGE_PATH),
        },
        "api_key": api_key,
        "prompt": "What is the name of the player on the left?",
        "history": [(prompt, res)],
    }
    async with session.post(
        f"http://localhost:{PORT}/llm/cogvlm",
        json=infer_payload,
    ) as response:
        if response.status != 200:
            print(response.status)
            print(await response.json())
            raise RuntimeError
        resp = await response.json()
        res = resp["response"]
        print(resp)


async def main():
    import time

    start = time.perf_counter()
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=100)
    async with aiohttp.ClientSession(read_timeout=0, connector=connector) as session:
        await do_cog_request(session)
    total = time.perf_counter() - start
    print(f"Total time: {total:.2f} seconds")


if __name__ == "__main__":
    print(
        "Running this example make sure you use `inference<=0.37.1`. We have announced "
        "End Of Life for Cog VLM in `inference` - this code would only work if you run the "
        "`inference` server in a version <= 0.37.1. Beware that the deprecation was related to "
        "security issue: https://nvd.nist.gov/vuln/detail/CVE-2024-11393 and we advise not to use "
        "the legacy `inference` builds in production."
    )
    Image.open(
        requests.get(
            "https://source.roboflow.com/ACrZ7Hz8DRUB1NBMMtDoQK84Hf22/0qUjAGRJQWWhT5j9hUOG/original.jpg",
            stream=True,
        ).raw
    ).convert("RGB").save(IMAGE_PATH)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
