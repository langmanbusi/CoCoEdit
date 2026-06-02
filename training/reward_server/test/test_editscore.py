import requests
from PIL import Image
import io
import pickle
import glob
import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
from time import time

BATCH_SIZE = 1

paths = [os.path.join("/home/notebook/data/group/wyh/reward-server/image", "a photo of a brown giraffe and a white stop sign.png")]
# paths = [
#     "/home/notebook/data/group/wyh/Step1X-Edit/GEdit-Bench/gedit-bench/fullset/subject-replace/en/0051b688bcfc65a4fc1063488eb9da0c.png", 
#     "/home/notebook/data/group/wyh/Step1X-Edit/GEdit-Bench/gedit-bench/fullset/color_alter/en/00644e09e285f614bbfae5883328b4df.png",
# ]

def f(_):
    t1 = time()
    for i in tqdm.tqdm(range(0, len(paths), BATCH_SIZE)):
        batch_paths = paths[i : i + BATCH_SIZE]

        jpeg_data = []
        for path in batch_paths:
            image = Image.open(path)

            # Compress the images using JPEG
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            jpeg_data.append(buffer.getvalue())

        data = {"images": jpeg_data}
        data = {"ref_images": jpeg_data}
        data["prompts"] = ["change the giraffe into black"]
        data_bytes = pickle.dumps(data)

        # Send the JPEG data in an HTTP POST request to the server
        url = "http://10.77.224.48:18086"
        response = requests.post(url, data=data_bytes)

        # Print the response from the server
        print("response", response)
        print("content", response.content)
        response_data = pickle.loads(response.content)

        print("response_data", response_data)
        print("time", time() - t1)

# with ThreadPoolExecutor(max_workers=8) as executor:
#     for _ in executor.map(f, range(8)):
#         pass
f(1)