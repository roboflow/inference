import base64
import requests
from multiprocessing import Pool
import asyncio
import aiohttp


def encode_bas64(image_path, vips=False):
    with open(image_path, "rb") as image:
        x = image.read()
        if vips:
            return x
        image_string = base64.b64encode(x)

    return image_string.decode("ascii")



def request1(i):
    print(i)
    r = requests.post(
        "http://localhost:8000/infer",
        json={"image": encode_bas64("testim.jpg"), "model": "model"},
    )
    print(r.text)
    print(f"FINISHED {i}")
    if r.status_code == 200:
        return 1
    return 0


def request2(i):
    r = requests.post(
        "http://localhost:8000/infer",
        json={"image": encode_bas64("testim.jpg"), "model": "model2"},
    )
    print(r.text)
    print(f"FINISHED {i}")
    if r.status_code == 200:
        return 1
    return 0

def request3(i):
    print(i)
    print(requests.get("http://localhost:8000/").text)
    print(f"FINISHED {i}")

async def do_request(session, i):
    print(f"Starting {i}")
    async with session.post(
        'http://localhost:8000/infer',
        json={"image": encode_bas64("youtube-19-small.jpg"), "model": "model2"},
    ) as response:
        resp = await response.json()
        print(f"Finished {i}")
        print(resp)
        return resp
    
async def do_old_request(session, i):

    project_id = ""
    model_version = ""
    image_url = ""
    confidence = 0.75
    api_key = "Nw3QZal3hhwHP5npbWmw"

    print(f"Starting {i}")
    infer_payload = {
        "image": [{
            "type": "base64",
            "value": encode_bas64("youtube-19-small.jpg"),
        }]*64,
        "confidence": 0.4,
        "iou_threshold": 0.5,
        "api_key": api_key,
        "model_id": "melee/5"
    }
    async with session.post(
        f'http://localhost:9001/infer/object_detection',
        json=infer_payload,
    ) as response:
        resp = await response.json()
        print(f"Finished {i}")
        print(resp)
        return resp


async def do_request_test(session, i):
    print(f"Starting {i}")
    async with session.get(
        'http://localhost:8000/'
    ) as response:
        resp = await response.json()
        print(f"Finished {i}")
        print(resp)
        return resp

def test_image_decompress_speed():
    import pyvips
    from PIL import Image
    import time
    import io
    im = encode_bas64("youtube-19.jpg", vips=True)
    start = time.time()
    im = pyvips.Image.new_from_buffer(im, "")
    im = Image.fromarray(im.numpy())
    im.draft("RGB", (640, 640))
    im.load()
    im = im.resize((640, 640))
    print(f"VIPS Took {time.time() - start} seconds")

    im = encode_bas64("youtube-19.jpg")
    start = time.time()
    decoded_string = io.BytesIO(base64.b64decode(im))
    im = Image.open(decoded_string)
    im.load()
    im = im.resize((640, 640))
    print(f"Pillow Took {time.time() - start} seconds")

    im = encode_bas64("youtube-19.jpg")
    start = time.time()
    decoded_string = io.BytesIO(base64.b64decode(im))
    im = Image.open(decoded_string)
    im.draft("RGB", im.size)
    im.load()
    print(f"Image size after draft {im.size}")
    im = im.resize((640, 640))
    print(f"Pillow (Draft) Took {time.time() - start} seconds")

    start = time.time()
    im = Image.open("youtube-19.jpg")
    print(im.draft("RGB", im.size))
    print(f"Last size {im.size}")
    im.load()
    print(f"Last size {im.size}")
    im = im.resize((640, 640))
    print(f"Pillow (Draft) (FILE) Took {time.time() - start} seconds")
    print(im)
    
    from PIL import features
    print(features.check_feature('libjpeg_turbo'))

async def main():
    tasks = []
    import time
    start = time.time()
    request_batch_size = 64
    num_requests = 1000 // request_batch_size
    connector = aiohttp.TCPConnector(limit=10000, limit_per_host=100000)
    async with aiohttp.ClientSession(read_timeout=0, connector=connector) as session:
        for i in range(num_requests):
            tasks.append(do_old_request(session, i))
        await asyncio.gather(*tasks)
    total = time.time() - start
    print(f"Total time: {total:.2f} seconds")
    print(f"{num_requests * request_batch_size / total} fps")

if __name__ == "__main__":
    from PIL import Image
    im = Image.open("youtube-19.jpg")
    im = im.resize((640, 640))
    im.save("youtube-19-small.jpg")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    # test_image_decompress_speed()