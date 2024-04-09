
import requests
import pandas as pd
import sys
import os
#from skimage import io, transform
import matplotlib.pyplot as plt
import time

# testUrl = f'https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={raEx}&dec={dcEx}&scale=0.1&width=200&height=200' 


df0 = pd.read_fwf('PCC_cat.txt', header=None)
ra = df0[2]
dec = df0[3]
outDir = 'SDSS400'
height = 400
width = 400

for i in range(len(ra)):
# for i in range(800):
    urlVar = f'https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/'    f'getjpeg?TaskName=Skyserver.Explore.Image&ra={str(ra[i]).strip()}'    f'&dec={str(dec[i]).strip()}&scale=0.1&width={width}&height={height}'
    
    # tell the loop to pause for a bit - every 100 images
    if i%100==0:
        time.sleep(5)
        
    img_data = requests.get(urlVar).content
    with open(f'{outDir}/sdss_ra={ra[i]}_dec={dec[i]}.png', 'wb') as handler:
        handler.write(img_data)


#images = [f'https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/'          f'getjpeg?TaskName=Skyserver.Explore.Image&ra={str(ra[i]).strip()}'          f'&dec={str(dec[i]).strip()}&scale=0.1&width=200&height=200' for i in range(len(ra))]

# os.path.join('SDSS', f'sdss_ra={str(df0[2][1])}_dec={str(df0[3][1])}.jpeg')
