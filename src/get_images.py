import numpy as np
import pandas as pd
import SciServer.CasJobs as CasJobs # query with CasJobs, the primary database for the SDSS
import SciServer.SkyServer as SkyServer # show individual objects through SkyServer
import SciServer.SciDrive
import warnings
import os
import PIL

# warnings.filterwarnings('ignore')

def get_images(df_in, dirName): 
        
    # df_in = df_in.drop_duplicates(subset = 'objID')
    df_in['labels'] = pd.to_numeric(df_in['labels'], downcast='integer')
    
    img_width, img_height = 200, 200
    SkyServer_DataRelease = 'DR16'
    # dirName = 'PCC-and-SpecSearch'
    
    outDir = os.path.join('..', 'Images', dirName)
    
    fileList = list()

    ## create directory if doesn't currently exist 
    if not os.path.exists(outDir):
       os.makedirs(outDir)

    # if len(glob.glob(os.path.join(outDir, '*.png'))) == df_in.shape[0]:
    #     print('Skipping Populate')
    # else:
        # for id, r, d in zip(searchDf['objID'], df_in['ra'], df_in['dec']):
    
    for r, d, l in zip(df_in['ra'], df_in['dec'], df_in['labels']):
        img_array = SkyServer.getJpegImgCutout(ra = r, 
                                               dec = d, 
                                               width = img_width, 
                                               height = img_height, 
                                               scale = 0.1, 
                                               dataRelease = SkyServer_DataRelease)
        
        # print(f'{id}-label={labeler(z)}')
        # outPicTemplate = f'{id}-label={labeler(z)}.png'
        
        outPicTemplate = f'sdss_ra={r}_dec={d}-label={l}.png'
        
        img0 = PIL.Image.fromarray(img_array, 'RGB')
        img0.save(f'{outDir}/{outPicTemplate}')
        fileList.append(f'{outPicTemplate}')

    # print(f'Finished populate with {len(fileList)} images')