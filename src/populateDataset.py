import requests
import pandas as pd
import os
import time

# testUrl = f'https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={raEx}&dec={dcEx}&scale=0.1&width=200&height=200'
df0 = (pd.read_fwf("PCC_cat.txt", header=None),)
ra = df0[2]
dec = df0[3]


def populateDataset(
    outDir="SDSS200",
    height=200,
    width=200,
    scale=0.1,
):
    for i in range(len(ra)):
        urlVar = (
            f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/"
            f"getjpeg?TaskName=Skyserver.Explore.Image&ra={str(ra[i]).strip()}"
            f"&dec={str(dec[i]).strip()}&scale={scale}&width={width}&height={height}"
        )

        # tell the loop to pause for a bit - every 100 images (to avoid timeout)
        if i % 100 == 0:
            time.sleep(5)

        img_data = requests.get(urlVar).content
        out_path = os.path.join(outDir, f"sdss_ra={str(ra[i])}_dec={str(dec[i])}.png")
        with open(out_path, "wb") as handler:
            handler.write(img_data)
            
if __name__ == "__main__":
    populateDataset()