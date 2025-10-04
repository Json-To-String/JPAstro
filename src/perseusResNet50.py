#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import sys
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import PIL
import cv2
import seaborn as sns

# from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
# from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
# from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.applications.vgg16 import preprocess_input

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
# from sklearn.metrics import confusion_matrix

from tensorflow.keras import activations

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, Flatten, AveragePooling2D, Dense, Activation, Add
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


import SciServer.CasJobs as CasJobs # query with CasJobs, the primary database for the SDSS
import SciServer.SkyServer as SkyServer # show individual objects through SkyServer
import SciServer.SciDrive
import warnings
# warnings.filterwarnings('ignore')


# In[2]:


# # Helper functions for visualization:
def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    From scikit-learn: plots a confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    #fixes "squishing of plot"
    plt.ylim([1.5, -.5])

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

def plot_model_history(history, n_epochs):
    '''Plot the training and validation history for a TensorFlow network'''

    # Extract loss and accuracy
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,5))
    ax[0].plot(np.arange(n_epochs), loss, label='Training Loss')
    ax[0].plot(np.arange(n_epochs), val_loss, label='Validation Loss')
    ax[0].set_title('Loss Curves')
    ax[0].legend()
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')

    ax[1].plot(np.arange(n_epochs), acc, label='Training Accuracy')
    ax[1].plot(np.arange(n_epochs), val_acc, label='Validation Accuracy')
    ax[1].set_title('Accuracy Curves')
    ax[1].legend()
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    
    
# # Check balance of labels/data in dataframe

def checkBalance(df):
    all_labels = df['labels']
    all_labels = all_labels.tolist()
    balance = df['labels'].value_counts()
    print(balance)
    for i in range(len(balance)):
        print(f'{balance[i]*200/df.size:.2f} %')


# In[3]:


df0 = pd.read_fwf('../PCC_cat.txt', header=None)
# df0[21] # 21 is the label entry index


# # Here we have 7 unique labels:
labels = np.unique(df0[21])

# access ra and dec from their columns in the dataframe
ra = df0[2]
dec = df0[3]

# want only bright objects above r_mag < 19.4  (the magnitude decreases as brightness increases)
bright = np.where(df0[4] <= 19.4)
brightDF = df0.iloc[bright].copy()

labels = np.unique(brightDF[21])

# access ra and dec from their columns in the dataframe
ra = brightDF[2]
dec = brightDF[3]

filenames = []
for r, d in zip(ra, dec):
    fn = f'sdss_ra={r}_dec={d}.png'
    filenames.append(fn)

brightDF_reduced = pd.DataFrame({'files' : filenames,
                                 'labels': brightDF[21]})


# checkBalance(brightDF_reduced)
df1 = brightDF_reduced
unique_labels = np.unique(df1['labels'])

clusterBG_LTG = df1.loc[(df1['labels']==unique_labels[0])]
BG_ETG = df1.loc[(df1['labels']==unique_labels[1])]
clusterBG_edgeDisk = df1.loc[(df1['labels']==unique_labels[2])]
likely_dE_ETGcluster = df1.loc[(df1['labels']==unique_labels[3])]
likely_merging = df1.loc[(df1['labels']==unique_labels[4])]
poss_dE_ETGcluster = df1.loc[(df1['labels']==unique_labels[5])]
weak_bg = df1.loc[(df1['labels']==unique_labels[6])]

downSampleDf0 = pd.concat([clusterBG_LTG, # 384
                         BG_ETG.sample(frac = 400/3008),
                         clusterBG_edgeDisk.sample(frac = 400/1049),
                         likely_dE_ETGcluster, # 398
                         likely_merging, # 23
                         poss_dE_ETGcluster, # 98
                         weak_bg # 477
                         ])
# checkBalance(downSampleDf0)

def replace(df, ind):
    label = list(df['labels'])[0]
    newDf = df.replace(label, str(ind))
    return(newDf)


# combined 3 and 5
second = pd.concat([
                    replace(likely_dE_ETGcluster, 1), # old 3
                    replace(poss_dE_ETGcluster, 1) # old 5
                    ])
# combine 0,1,2,6
first = pd.concat([replace(clusterBG_LTG, 0), # old 0
                   replace(BG_ETG, 0), # old 1
                   replace(clusterBG_edgeDisk, 0), # old 2
                   replace(weak_bg, 0) # old 6
                    ])

lenSecond = len(second.index)
lenFirst = len(first.index)

# df with PCC objects reduced to 0s or 1s
downSampleDf1 = pd.concat([first.sample(frac = lenSecond/lenFirst), second])

# 0 is background
# 1 is dE/ETGcluster
downSampleDf1


# In[4]:


pcc_crossmatchQuery = '''CREATE TABLE #upload ( up_id int, up_files varchar(32), up_name varchar(32), up_labels varchar(32), up_reds varchar(32), up_ra float, up_dec float ) 
INSERT INTO #upload values ( 1, 'sdss_ra=49.5885_dec=41.6896.png', 'PCC-2087', '0', '2.39', 49.5885, 41.6896),( 2, 'sdss_ra=49.4776_dec=41.384.png', 'PCC-1378', '0', '6.325', 49.4776, 41.384),( 3, 'sdss_ra=50.004_dec=41.341.png', 'PCC-5423', '0', '5.62', 50.004, 41.341),( 4, 'sdss_ra=49.4142_dec=41.3882.png', 'PCC-0989', '0', '1.6325', 49.4142, 41.3882),( 5, 'sdss_ra=50.0018_dec=41.6806.png', 'PCC-5408', '0', '1.8725', 50.0018, 41.6806),( 6, 'sdss_ra=49.6743_dec=41.3287.png', 'PCC-2818', '0', '0.04', 49.6743, 41.3287),( 7, 'sdss_ra=49.4266_dec=41.3159.png', 'PCC-1064', '0', '1.8225', 49.4266, 41.3159),( 8, 'sdss_ra=49.3303_dec=41.5381.png', 'PCC-0494', '0', '2.5825', 49.3303, 41.5381),( 9, 'sdss_ra=49.8286_dec=41.2883.png', 'PCC-4039', '0', '0.1675', 49.8286, 41.2883),( 10, 'sdss_ra=49.3925_dec=41.5647.png', 'PCC-0870', '0', '1.265', 49.3925, 41.5647),( 11, 'sdss_ra=49.4784_dec=41.2846.png', 'PCC-1388', '0', '3.62', 49.4784, 41.2846),( 12, 'sdss_ra=49.5392_dec=41.4577.png', 'PCC-1782', '0', '2.515', 49.5392, 41.4577),( 13, 'sdss_ra=49.7004_dec=41.5288.png', 'PCC-3025', '0', '1.695', 49.7004, 41.5288),( 14, 'sdss_ra=49.7521_dec=41.666.png', 'PCC-3450', '0', '1.74', 49.7521, 41.666),( 15, 'sdss_ra=49.7463_dec=41.6377.png', 'PCC-3408', '0', '5.7625', 49.7463, 41.6377),( 16, 'sdss_ra=49.7314_dec=41.5563.png', 'PCC-3286', '0', '17.815', 49.7314, 41.5563),( 17, 'sdss_ra=49.5063_dec=41.5123.png', 'PCC-1565', '0', '6.8275', 49.5063, 41.5123),( 18, 'sdss_ra=49.5319_dec=41.5225.png', 'PCC-1742', '0', '2.4475', 49.5319, 41.5225),( 19, 'sdss_ra=49.5132_dec=41.3783.png', 'PCC-1617', '0', '2.0275', 49.5132, 41.3783),( 20, 'sdss_ra=49.4477_dec=41.575.png', 'PCC-1191', '0', '1.6075', 49.4477, 41.575),( 21, 'sdss_ra=49.2565_dec=41.4396.png', 'PCC-0057', '0', '3.51', 49.2565, 41.4396),( 22, 'sdss_ra=49.3508_dec=41.3929.png', 'PCC-0626', '0', '19.445', 49.3508, 41.3929),( 23, 'sdss_ra=49.5144_dec=41.6052.png', 'PCC-1623', '0', '17.38', 49.5144, 41.6052),( 24, 'sdss_ra=49.3878_dec=41.5814.png', 'PCC-0826', '0', '0.705', 49.3878, 41.5814),( 25, 'sdss_ra=49.7118_dec=41.7236.png', 'PCC-3109', '0', '1.48', 49.7118, 41.7236),( 26, 'sdss_ra=49.7624_dec=41.308.png', 'PCC-3531', '0', '4.0975', 49.7624, 41.308),( 27, 'sdss_ra=49.8839_dec=41.7175.png', 'PCC-4515', '0', '6.82', 49.8839, 41.7175),( 28, 'sdss_ra=49.7006_dec=41.4398.png', 'PCC-3028', '0', '9.215', 49.7006, 41.4398),( 29, 'sdss_ra=49.6028_dec=41.5367.png', 'PCC-2195', '0', '8.365', 49.6028, 41.5367),( 30, 'sdss_ra=49.2466_dec=41.4451.png', 'PCC-0026', '0', '0.445', 49.2466, 41.4451),( 31, 'sdss_ra=49.6212_dec=41.7373.png', 'PCC-2365', '0', '2.2225', 49.6212, 41.7373),( 32, 'sdss_ra=49.407_dec=41.3729.png', 'PCC-0954', '0', '2.32', 49.407, 41.3729),( 33, 'sdss_ra=49.3329_dec=41.4999.png', 'PCC-0514', '0', '0.92', 49.3329, 41.4999),( 34, 'sdss_ra=49.2744_dec=41.5706.png', 'PCC-0160', '0', '1.22', 49.2744, 41.5706),( 35, 'sdss_ra=49.4898_dec=41.5315.png', 'PCC-1467', '0', '13.855', 49.4898, 41.5315),( 36, 'sdss_ra=49.4525_dec=41.5851.png', 'PCC-1222', '0', '1.9275', 49.4525, 41.5851),( 37, 'sdss_ra=49.3368_dec=41.4033.png', 'PCC-0538', '0', '13.4775', 49.3368, 41.4033),( 38, 'sdss_ra=49.5331_dec=41.4152.png', 'PCC-1749', '0', '2.0575', 49.5331, 41.4152),( 39, 'sdss_ra=49.5756_dec=41.4865.png', 'PCC-1992', '0', '2.2225', 49.5756, 41.4865),( 40, 'sdss_ra=50.0051_dec=41.6557.png', 'PCC-5428', '0', '1.8225', 50.0051, 41.6557),( 41, 'sdss_ra=49.7564_dec=41.2183.png', 'PCC-3477', '0', '6.9975', 49.7564, 41.2183),( 42, 'sdss_ra=49.6283_dec=41.2297.png', 'PCC-2441', '0', '13.07', 49.6283, 41.2297),( 43, 'sdss_ra=49.3541_dec=41.4703.png', 'PCC-0641', '0', '10.3425', 49.3541, 41.4703),( 44, 'sdss_ra=49.3081_dec=41.4699.png', 'PCC-0365', '0', '6.7025', 49.3081, 41.4699),( 45, 'sdss_ra=49.3881_dec=41.3949.png', 'PCC-0828', '0', '13.7625', 49.3881, 41.3949),( 46, 'sdss_ra=49.6684_dec=41.258.png', 'PCC-2760', '0', '4.1625', 49.6684, 41.258),( 47, 'sdss_ra=49.5907_dec=41.6753.png', 'PCC-2113', '0', '8.5275', 49.5907, 41.6753),( 48, 'sdss_ra=49.5877_dec=41.3917.png', 'PCC-2079', '0', '3.29', 49.5877, 41.3917),( 49, 'sdss_ra=49.4508_dec=41.5244.png', 'PCC-1208', '0', '0.745', 49.4508, 41.5244),( 50, 'sdss_ra=49.6184_dec=41.4546.png', 'PCC-2341', '0', '1.7125', 49.6184, 41.4546),( 51, 'sdss_ra=49.3497_dec=41.2962.png', 'PCC-0619', '0', '1.265', 49.3497, 41.2962),( 52, 'sdss_ra=49.9814_dec=41.4202.png', 'PCC-5240', '0', '1.585', 49.9814, 41.4202),( 53, 'sdss_ra=49.5539_dec=41.502.png', 'PCC-1867', '0', '10.2025', 49.5539, 41.502),( 54, 'sdss_ra=49.5094_dec=41.5135.png', 'PCC-1592', '0', '11.1625', 49.5094, 41.5135),( 55, 'sdss_ra=49.6309_dec=41.2656.png', 'PCC-2462', '0', '7.75', 49.6309, 41.2656),( 56, 'sdss_ra=49.4709_dec=41.3465.png', 'PCC-1336', '0', '0.0125', 49.4709, 41.3465),( 57, 'sdss_ra=49.2475_dec=41.3259.png', 'PCC-0027', '0', '1.7425', 49.2475, 41.3259),( 58, 'sdss_ra=49.623_dec=41.7693.png', 'PCC-2383', '0', '2.5125', 49.623, 41.7693),( 59, 'sdss_ra=49.9366_dec=41.7691.png', 'PCC-4910', '0', '2.935', 49.9366, 41.7691),( 60, 'sdss_ra=49.5669_dec=41.3688.png', 'PCC-1940', '0', '1.5825', 49.5669, 41.3688),( 61, 'sdss_ra=49.4854_dec=41.4154.png', 'PCC-1433', '0', '3.59', 49.4854, 41.4154),( 62, 'sdss_ra=49.4978_dec=41.6188.png', 'PCC-1515', '0', '3.41', 49.4978, 41.6188),( 63, 'sdss_ra=49.8832_dec=41.3043.png', 'PCC-4508', '0', '9.5525', 49.8832, 41.3043),( 64, 'sdss_ra=49.9842_dec=41.2138.png', 'PCC-5265', '0', '0.8025', 49.9842, 41.2138),( 65, 'sdss_ra=49.4511_dec=41.3201.png', 'PCC-1212', '0', '0.665', 49.4511, 41.3201),( 66, 'sdss_ra=49.7175_dec=41.6788.png', 'PCC-3159', '0', '12.7875', 49.7175, 41.6788),( 67, 'sdss_ra=49.7208_dec=41.5487.png', 'PCC-3190', '0', '42.3225', 49.7208, 41.5487),( 68, 'sdss_ra=49.9435_dec=41.6033.png', 'PCC-4967', '0', '18.6625', 49.9435, 41.6033),( 69, 'sdss_ra=49.8546_dec=41.2312.png', 'PCC-4260', '0', '4.5675', 49.8546, 41.2312),( 70, 'sdss_ra=49.5411_dec=41.5389.png', 'PCC-1795', '0', '6.025', 49.5411, 41.5389),( 71, 'sdss_ra=49.5078_dec=41.4149.png', 'PCC-1578', '0', '0.3725', 49.5078, 41.4149),( 72, 'sdss_ra=49.4815_dec=41.434.png', 'PCC-1406', '0', '0.8525', 49.4815, 41.434),( 73, 'sdss_ra=49.5511_dec=41.4042.png', 'PCC-1849', '0', '0.89', 49.5511, 41.4042),( 74, 'sdss_ra=49.9437_dec=41.3786.png', 'PCC-4969', '0', '7.01', 49.9437, 41.3786),( 75, 'sdss_ra=49.5475_dec=41.5225.png', 'PCC-1827', '0', '10.325', 49.5475, 41.5225),( 76, 'sdss_ra=49.8615_dec=41.4943.png', 'PCC-4319', '0', '23.2275', 49.8615, 41.4943),( 77, 'sdss_ra=49.6876_dec=41.226.png', 'PCC-2928', '0', '4.625', 49.6876, 41.226),( 78, 'sdss_ra=49.2669_dec=41.5264.png', 'PCC-0110', '0', '2.055', 49.2669, 41.5264),( 79, 'sdss_ra=49.3766_dec=41.5364.png', 'PCC-0774', '0', '0.0975', 49.3766, 41.5364),( 80, 'sdss_ra=49.5082_dec=41.4069.png', 'PCC-1581', '0', '2.4', 49.5082, 41.4069),( 81, 'sdss_ra=49.9564_dec=41.3138.png', 'PCC-5087', '0', '31.96', 49.9564, 41.3138),( 82, 'sdss_ra=49.7705_dec=41.3615.png', 'PCC-3605', '0', '7.835', 49.7705, 41.3615),( 83, 'sdss_ra=49.3431_dec=41.5987.png', 'PCC-0571', '0', '1.5575', 49.3431, 41.5987),( 84, 'sdss_ra=49.5268_dec=41.3295.png', 'PCC-1707', '0', '6.0975', 49.5268, 41.3295),( 85, 'sdss_ra=49.6744_dec=41.3395.png', 'PCC-2819', '0', '1.1975', 49.6744, 41.3395),( 86, 'sdss_ra=49.9834_dec=41.4553.png', 'PCC-5257', '0', '5.9725', 49.9834, 41.4553),( 87, 'sdss_ra=49.5811_dec=41.3339.png', 'PCC-2031', '0', '0.6825', 49.5811, 41.3339),( 88, 'sdss_ra=49.8625_dec=41.6234.png', 'PCC-4328', '0', '4.6825', 49.8625, 41.6234),( 89, 'sdss_ra=49.7875_dec=41.5292.png', 'PCC-3700', '0', '11.1775', 49.7875, 41.5292),( 90, 'sdss_ra=49.9985_dec=41.3856.png', 'PCC-5387', '0', '3.3975', 49.9985, 41.3856),( 91, 'sdss_ra=49.6469_dec=41.4505.png', 'PCC-2605', '0', '3.2575', 49.6469, 41.4505),( 92, 'sdss_ra=49.242_dec=41.4454.png', 'PCC-0011', '0', '0.24', 49.242, 41.4454),( 93, 'sdss_ra=49.4487_dec=41.3384.png', 'PCC-1198', '0', '1.0375', 49.4487, 41.3384),( 94, 'sdss_ra=49.9154_dec=41.2302.png', 'PCC-4755', '0', '3.9775', 49.9154, 41.2302),( 95, 'sdss_ra=49.5705_dec=41.4591.png', 'PCC-1967', '0', '4.16', 49.5705, 41.4591),( 96, 'sdss_ra=49.3855_dec=41.3037.png', 'PCC-0817', '0', '10.1125', 49.3855, 41.3037),( 97, 'sdss_ra=49.3893_dec=41.5721.png', 'PCC-0837', '0', '0.4225', 49.3893, 41.5721),( 98, 'sdss_ra=49.6164_dec=41.4551.png', 'PCC-2329', '0', '1.3825', 49.6164, 41.4551),( 99, 'sdss_ra=49.36_dec=41.3234.png', 'PCC-0675', '0', '0.555', 49.36, 41.3234),( 100, 'sdss_ra=49.985_dec=41.3319.png', 'PCC-5271', '0', '4.315', 49.985, 41.3319),( 101, 'sdss_ra=49.6116_dec=41.197.png', 'PCC-2283', '0', '1.5825', 49.6116, 41.197),( 102, 'sdss_ra=49.3594_dec=41.3355.png', 'PCC-0670', '0', '0.485', 49.3594, 41.3355),( 103, 'sdss_ra=49.5006_dec=41.3793.png', 'PCC-1530', '0', '0.575', 49.5006, 41.3793),( 104, 'sdss_ra=49.6822_dec=41.2678.png', 'PCC-2878', '0', '2.2375', 49.6822, 41.2678),( 105, 'sdss_ra=50.0028_dec=41.3384.png', 'PCC-5417', '0', '3.2525', 50.0028, 41.3384),( 106, 'sdss_ra=49.3464_dec=41.4658.png', 'PCC-0592', '0', '7.6125', 49.3464, 41.4658),( 107, 'sdss_ra=49.5247_dec=41.4305.png', 'PCC-1693', '0', '0.4275', 49.5247, 41.4305),( 108, 'sdss_ra=49.7345_dec=41.6346.png', 'PCC-3313', '0', '4.3675', 49.7345, 41.6346),( 109, 'sdss_ra=49.5236_dec=41.5182.png', 'PCC-1689', '0', '1.99', 49.5236, 41.5182),( 110, 'sdss_ra=49.4158_dec=41.3219.png', 'PCC-0997', '0', '3.605', 49.4158, 41.3219),( 111, 'sdss_ra=49.2411_dec=41.4991.png', 'PCC-0008', '0', '4.1925', 49.2411, 41.4991),( 112, 'sdss_ra=49.2758_dec=41.5415.png', 'PCC-0168', '0', '5.9025', 49.2758, 41.5415),( 113, 'sdss_ra=49.5591_dec=41.5027.png', 'PCC-1896', '0', '12.3575', 49.5591, 41.5027),( 114, 'sdss_ra=49.6769_dec=41.3202.png', 'PCC-2835', '0', '5.01', 49.6769, 41.3202),( 115, 'sdss_ra=49.8577_dec=41.3264.png', 'PCC-4286', '0', '3.9675', 49.8577, 41.3264),( 116, 'sdss_ra=49.2516_dec=41.3224.png', 'PCC-0040', '1', '4.3075', 49.2516, 41.3224),( 117, 'sdss_ra=49.2835_dec=41.3141.png', 'PCC-0219', '1', '3.31', 49.2835, 41.3141),( 118, 'sdss_ra=49.2874_dec=41.4252.png', 'PCC-0246', '1', '3.59', 49.2874, 41.4252),( 119, 'sdss_ra=49.3008_dec=41.3811.png', 'PCC-0314', '1', '11.765', 49.3008, 41.3811),( 120, 'sdss_ra=49.3048_dec=41.3333.png', 'PCC-0349', '1', '12.4125', 49.3048, 41.3333),( 121, 'sdss_ra=49.3063_dec=41.2975.png', 'PCC-0358', '1', '3.5275', 49.3063, 41.2975),( 122, 'sdss_ra=49.3067_dec=41.4355.png', 'PCC-0360', '1', '0.7175', 49.3067, 41.4355),( 123, 'sdss_ra=49.3237_dec=41.4215.png', 'PCC-0462', '1', '7.3625', 49.3237, 41.4215),( 124, 'sdss_ra=49.3256_dec=41.4929.png', 'PCC-0469', '1', '1.3375', 49.3256, 41.4929),( 125, 'sdss_ra=49.3476_dec=41.47.png', 'PCC-0602', '1', '13.6375', 49.3476, 41.47),( 126, 'sdss_ra=49.3663_dec=41.3535.png', 'PCC-0708', '1', '9.7675', 49.3663, 41.3535),( 127, 'sdss_ra=49.387_dec=41.297.png', 'PCC-0823', '1', '1.22', 49.387, 41.297),( 128, 'sdss_ra=49.4082_dec=41.3954.png', 'PCC-0958', '1', '10.75', 49.4082, 41.3954),( 129, 'sdss_ra=49.4134_dec=41.5176.png', 'PCC-0985', '1', '1.3875', 49.4134, 41.5176),( 130, 'sdss_ra=49.4149_dec=41.5612.png', 'PCC-0993', '1', '1.1775', 49.4149, 41.5612),( 131, 'sdss_ra=49.4273_dec=41.3231.png', 'PCC-1067', '1', '12.985', 49.4273, 41.3231),( 132, 'sdss_ra=49.4284_dec=41.5999.png', 'PCC-1078', '1', '8.705', 49.4284, 41.5999),( 133, 'sdss_ra=49.432_dec=41.3634.png', 'PCC-1101', '1', '8.7475', 49.432, 41.3634),( 134, 'sdss_ra=49.4519_dec=41.4084.png', 'PCC-1220', '1', '2.6575', 49.4519, 41.4084),( 135, 'sdss_ra=49.4615_dec=41.522.png', 'PCC-1282', '1', '0.8025', 49.4615, 41.522),( 136, 'sdss_ra=49.5028_dec=41.2991.png', 'PCC-1543', '1', '4.92', 49.5028, 41.2991),( 137, 'sdss_ra=49.5078_dec=41.5578.png', 'PCC-1577', '1', '1.0875', 49.5078, 41.5578),( 138, 'sdss_ra=49.5126_dec=41.5128.png', 'PCC-1613', '1', '8.615', 49.5126, 41.5128),( 139, 'sdss_ra=49.5154_dec=41.3231.png', 'PCC-1628', '1', '9.28', 49.5154, 41.3231),( 140, 'sdss_ra=49.5184_dec=41.3698.png', 'PCC-1655', '1', '5.9375', 49.5184, 41.3698),( 141, 'sdss_ra=49.5231_dec=41.4355.png', 'PCC-1684', '1', '3.385', 49.5231, 41.4355),( 142, 'sdss_ra=49.5244_dec=41.5814.png', 'PCC-1691', '1', '6.1025', 49.5244, 41.5814),( 143, 'sdss_ra=49.5308_dec=41.5539.png', 'PCC-1736', '1', '3.3825', 49.5308, 41.5539),( 144, 'sdss_ra=49.5486_dec=41.6169.png', 'PCC-1833', '1', '7.3675', 49.5486, 41.6169),( 145, 'sdss_ra=49.5502_dec=41.4233.png', 'PCC-1842', '1', '15.6625', 49.5502, 41.4233),( 146, 'sdss_ra=49.557_dec=41.4616.png', 'PCC-1884', '1', '2.215', 49.557, 41.4616),( 147, 'sdss_ra=49.5667_dec=41.2776.png', 'PCC-1937', '1', '9.0875', 49.5667, 41.2776),( 148, 'sdss_ra=49.568_dec=41.5799.png', 'PCC-1947', '1', '1.21', 49.568, 41.5799),( 149, 'sdss_ra=49.5741_dec=41.5031.png', 'PCC-1984', '1', '13.1275', 49.5741, 41.5031),( 150, 'sdss_ra=49.5791_dec=41.6937.png', 'PCC-2021', '1', '2.435', 49.5791, 41.6937),( 151, 'sdss_ra=49.582_dec=41.2943.png', 'PCC-2039', '1', '0.8525', 49.582, 41.2943),( 152, 'sdss_ra=49.5958_dec=41.2689.png', 'PCC-2146', '1', '6.9075', 49.5958, 41.2689),( 153, 'sdss_ra=49.5992_dec=41.6442.png', 'PCC-2172', '1', '29.7875', 49.5992, 41.6442),( 154, 'sdss_ra=49.6048_dec=41.5212.png', 'PCC-2214', '1', '1.2525', 49.6048, 41.5212),( 155, 'sdss_ra=49.6078_dec=41.5452.png', 'PCC-2241', '1', '5.425', 49.6078, 41.5452),( 156, 'sdss_ra=49.6152_dec=41.3213.png', 'PCC-2325', '1', '5.2975', 49.6152, 41.3213),( 157, 'sdss_ra=49.6169_dec=41.7091.png', 'PCC-2332', '1', '20.26', 49.6169, 41.7091),( 158, 'sdss_ra=49.6181_dec=41.286.png', 'PCC-2339', '1', '2.9925', 49.6181, 41.286),( 159, 'sdss_ra=49.6241_dec=41.3027.png', 'PCC-2393', '1', '8.96', 49.6241, 41.3027),( 160, 'sdss_ra=49.6382_dec=41.2977.png', 'PCC-2531', '1', '0.6525', 49.6382, 41.2977),( 161, 'sdss_ra=49.6396_dec=41.7522.png', 'PCC-2546', '1', '9.555', 49.6396, 41.7522),( 162, 'sdss_ra=49.6431_dec=41.4925.png', 'PCC-2571', '1', '3.9975', 49.6431, 41.4925),( 163, 'sdss_ra=49.6673_dec=41.4173.png', 'PCC-2754', '1', '2.19', 49.6673, 41.4173),( 164, 'sdss_ra=49.6693_dec=41.2867.png', 'PCC-2772', '1', '2.3975', 49.6693, 41.2867),( 165, 'sdss_ra=49.6694_dec=41.6242.png', 'PCC-2774', '1', '23.785', 49.6694, 41.6242),( 166, 'sdss_ra=49.6869_dec=41.6338.png', 'PCC-2918', '1', '2.355', 49.6869, 41.6338),( 167, 'sdss_ra=49.6925_dec=41.4049.png', 'PCC-2959', '1', '1.8125', 49.6925, 41.4049),( 168, 'sdss_ra=49.6929_dec=41.6028.png', 'PCC-2961', '1', '4.5675', 49.6929, 41.6028),( 169, 'sdss_ra=49.6991_dec=41.748.png', 'PCC-3010', '1', '6.1725', 49.6991, 41.748),( 170, 'sdss_ra=49.7077_dec=41.5067.png', 'PCC-3076', '1', '12.3', 49.7077, 41.5067),( 171, 'sdss_ra=49.712_dec=41.6398.png', 'PCC-3116', '1', '9.2625', 49.712, 41.6398),( 172, 'sdss_ra=49.7177_dec=41.2676.png', 'PCC-3164', '1', '8.89', 49.7177, 41.2676),( 173, 'sdss_ra=49.7227_dec=41.3095.png', 'PCC-3205', '1', '16.1025', 49.7227, 41.3095),( 174, 'sdss_ra=49.727_dec=41.2225.png', 'PCC-3244', '1', '5.9825', 49.727, 41.2225),( 175, 'sdss_ra=49.7333_dec=41.5787.png', 'PCC-3303', '1', '4.17', 49.7333, 41.5787),( 176, 'sdss_ra=49.7334_dec=41.6862.png', 'PCC-3305', '1', '6.01', 49.7334, 41.6862),( 177, 'sdss_ra=49.735_dec=41.2599.png', 'PCC-3316', '1', '2.2425', 49.735, 41.2599),( 178, 'sdss_ra=49.7397_dec=41.359.png', 'PCC-3361', '1', '6.2725', 49.7397, 41.359),( 179, 'sdss_ra=49.7405_dec=41.4161.png', 'PCC-3366', '1', '33.7925', 49.7405, 41.4161),( 180, 'sdss_ra=49.7415_dec=41.6351.png', 'PCC-3379', '1', '8.6625', 49.7415, 41.6351),( 181, 'sdss_ra=49.7454_dec=41.3209.png', 'PCC-3401', '1', '8.215', 49.7454, 41.3209),( 182, 'sdss_ra=49.7484_dec=41.4446.png', 'PCC-3418', '1', '3.915', 49.7484, 41.4446),( 183, 'sdss_ra=49.7516_dec=41.484.png', 'PCC-3444', '1', '10.1325', 49.7516, 41.484),( 184, 'sdss_ra=49.7624_dec=41.5043.png', 'PCC-3529', '1', '7.785', 49.7624, 41.5043),( 185, 'sdss_ra=49.7817_dec=41.2523.png', 'PCC-3661', '1', '5.35', 49.7817, 41.2523),( 186, 'sdss_ra=49.7852_dec=41.5811.png', 'PCC-3687', '1', '11.5075', 49.7852, 41.5811),( 187, 'sdss_ra=49.7895_dec=41.7047.png', 'PCC-3723', '1', '1.0425', 49.7895, 41.7047),( 188, 'sdss_ra=49.7926_dec=41.6781.png', 'PCC-3742', '1', '8.12', 49.7926, 41.6781),( 189, 'sdss_ra=49.7935_dec=41.4936.png', 'PCC-3749', '1', '15.975', 49.7935, 41.4936),( 190, 'sdss_ra=49.7995_dec=41.7588.png', 'PCC-3799', '1', '1.625', 49.7995, 41.7588),( 191, 'sdss_ra=49.8124_dec=41.6778.png', 'PCC-3919', '1', '1.6575', 49.8124, 41.6778),( 192, 'sdss_ra=49.8126_dec=41.2313.png', 'PCC-3921', '1', '7.695', 49.8126, 41.2313),( 193, 'sdss_ra=49.8178_dec=41.7537.png', 'PCC-3963', '1', '4.4325', 49.8178, 41.7537),( 194, 'sdss_ra=49.8495_dec=41.6924.png', 'PCC-4208', '1', '10.34', 49.8495, 41.6924),( 195, 'sdss_ra=49.8501_dec=41.5332.png', 'PCC-4216', '1', '31.8775', 49.8501, 41.5332),( 196, 'sdss_ra=49.8585_dec=41.2621.png', 'PCC-4297', '1', '2.2225', 49.8585, 41.2621),( 197, 'sdss_ra=49.8627_dec=41.6082.png', 'PCC-4330', '1', '7.4875', 49.8627, 41.6082),( 198, 'sdss_ra=49.8715_dec=41.7361.png', 'PCC-4402', '1', '2.66', 49.8715, 41.7361),( 199, 'sdss_ra=49.8823_dec=41.5225.png', 'PCC-4499', '1', '20.3575', 49.8823, 41.5225),( 200, 'sdss_ra=49.8902_dec=41.5536.png', 'PCC-4551', '1', '9.5975', 49.8902, 41.5536),( 201, 'sdss_ra=49.8969_dec=41.6179.png', 'PCC-4608', '1', '6.21', 49.8969, 41.6179),( 202, 'sdss_ra=49.9053_dec=41.486.png', 'PCC-4666', '1', '11.84', 49.9053, 41.486),( 203, 'sdss_ra=49.9145_dec=41.2779.png', 'PCC-4743', '1', '15.625', 49.9145, 41.2779),( 204, 'sdss_ra=49.9151_dec=41.5179.png', 'PCC-4750', '1', '17.335', 49.9151, 41.5179),( 205, 'sdss_ra=49.9175_dec=41.3292.png', 'PCC-4771', '1', '4.87', 49.9175, 41.3292),( 206, 'sdss_ra=49.9235_dec=41.4881.png', 'PCC-4811', '1', '18.5525', 49.9235, 41.4881),( 207, 'sdss_ra=49.9248_dec=41.4995.png', 'PCC-4816', '1', '30.525', 49.9248, 41.4995),( 208, 'sdss_ra=49.9317_dec=41.7131.png', 'PCC-4867', '1', '9.17', 49.9317, 41.7131),( 209, 'sdss_ra=49.9326_dec=41.457.png', 'PCC-4876', '1', '15.7675', 49.9326, 41.457),( 210, 'sdss_ra=49.936_dec=41.4465.png', 'PCC-4900', '1', '17.76', 49.936, 41.4465),( 211, 'sdss_ra=49.9449_dec=41.5293.png', 'PCC-4979', '1', '14.675', 49.9449, 41.5293),( 212, 'sdss_ra=49.9453_dec=41.3786.png', 'PCC-4981', '1', '9.1025', 49.9453, 41.3786),( 213, 'sdss_ra=49.9523_dec=41.5579.png', 'PCC-5047', '1', '28.535', 49.9523, 41.5579),( 214, 'sdss_ra=49.9538_dec=41.5834.png', 'PCC-5063', '1', '16.9225', 49.9538, 41.5834),( 215, 'sdss_ra=49.9574_dec=41.3299.png', 'PCC-5095', '1', '17.1275', 49.9574, 41.3299),( 216, 'sdss_ra=49.9575_dec=41.6693.png', 'PCC-5096', '1', '1.0675', 49.9575, 41.6693),( 217, 'sdss_ra=49.9635_dec=41.5354.png', 'PCC-5136', '1', '25.4425', 49.9635, 41.5354),( 218, 'sdss_ra=49.9636_dec=41.3097.png', 'PCC-5137', '1', '7.905', 49.9636, 41.3097),( 219, 'sdss_ra=49.9656_dec=41.3923.png', 'PCC-5147', '1', '2.3325', 49.9656, 41.3923),( 220, 'sdss_ra=49.967_dec=41.2598.png', 'PCC-5156', '1', '2.78', 49.967, 41.2598),( 221, 'sdss_ra=49.9672_dec=41.648.png', 'PCC-5157', '1', '2.3475', 49.9672, 41.648),( 222, 'sdss_ra=49.9686_dec=41.5498.png', 'PCC-5163', '1', '9.2', 49.9686, 41.5498),( 223, 'sdss_ra=49.9705_dec=41.6088.png', 'PCC-5169', '1', '6.5925', 49.9705, 41.6088),( 224, 'sdss_ra=49.9756_dec=41.3089.png', 'PCC-5196', '1', '11.9175', 49.9756, 41.3089),( 225, 'sdss_ra=49.9932_dec=41.5479.png', 'PCC-5339', '1', '28.4075', 49.9932, 41.5479),( 226, 'sdss_ra=49.9947_dec=41.75.png', 'PCC-5358', '1', '49.0425', 49.9947, 41.75),( 227, 'sdss_ra=49.9967_dec=41.3092.png', 'PCC-5374', '1', '17.355', 49.9967, 41.3092),( 228, 'sdss_ra=49.8825_dec=41.7447.png', 'PCC-4502', '1', '4.63', 49.8825, 41.7447),( 229, 'sdss_ra=49.9387_dec=41.2544.png', 'PCC-4932', '1', '5.585', 49.9387, 41.2544)
create table #x (up_id int,objID bigint)
INSERT INTO #x 
SELECT up_id, dbo.fGetNearestObjIdEq(up_ra,up_dec,0.03) as objId 
     FROM #upload WHERE dbo.fGetNearestObjIdEq(up_ra,up_dec,0.03) IS NOT NULL 
SELECT u.up_files as [files],u.up_name as [name],u.up_labels as [labels],u.up_reds as [reds], 
p.objID, 
dbo.fPhotoTypeN(p.type) as type,
p.ra, p.dec,
p.modelMag_r as R_mag, 
p.modelMag_u - p.modelMag_g as u_g, 
p.modelMag_g - p.modelMag_z as g_z, 
p.modelMag_g - p.modelMag_r as g_r, 
p.modelMag_g - p.modelMag_i as g_i, 
p.modelMag_r - p.modelMag_i as r_i, 
p.modelMag_r - p.modelMag_z as r_z, 
p.petroRad_r,
p.flags, dbo.fPhotoFlagsN(p.flags) as flag_text
FROM #upload u
JOIN #x x ON x.up_id = u.up_id
JOIN PhotoObj p ON p.objID = x.objID 
ORDER BY x.up_id
'''
pcc_crossDf = CasJobs.executeQuery(pcc_crossmatchQuery, "dr16")
# pcc_crossDf

## No color cuts
radialSearchNoColor = f'SELECT TOP 300 p.objID, p.ra, p.dec, \
 p.modelMag_r as R_mag, \
 p.modelMag_r - p.extinction_r as r0, \
 p.modelMag_g - p.extinction_g - p.modelMag_z + p.extinction_z as g_z0, \
 p.modelMag_u - p.modelMag_g as u_g, \
 p.modelMag_g - p.modelMag_z as g_z, \
 p.modelMag_g - p.modelMag_r as g_r, \
 p.modelMag_g - p.modelMag_i as g_i, \
 p.modelMag_r - p.modelMag_i as r_i, \
 p.modelMag_r - p.modelMag_z as r_z, \
 p.petroRad_r, p.flags, dbo.fPhotoFlagsN(p.flags) as flag_text, \
 s.specObjID, s.z, s.zErr, s.zWarning, s.class, s.subClass, \
 N.distance \
FROM \
    photoObj as p \
JOIN SpecObjAll s ON p.objID = s.bestObjID \
JOIN dbo.fGetNearbyObjEq(49.9467, 41.5131, 45) as N ON N.objID = p.objID \
WHERE \
    p.type = 3 \
ORDER BY distance'


searchDf = CasJobs.executeQuery(radialSearchNoColor, "dr16")


# In[5]:


searchDf = searchDf.drop_duplicates(subset = 'objID')
pcc_crossDf = pcc_crossDf.drop_duplicates(subset = 'objID')
## Members 0.01 < spec-z < 0.033, else are NonMembers 
searchDf_specMembers = searchDf.loc[(searchDf['z'] > 0.01) & (searchDf['z'] < 0.033)]
searchDf_specMembers['labels'] = np.zeros(int(searchDf_specMembers.shape[0]))
searchDf_specNonMembers = searchDf.loc[(searchDf['z'] < 0.01) | (searchDf['z'] > 0.033)]
searchDf_specNonMembers['labels'] = np.ones(int(searchDf_specNonMembers.shape[0]))

pcc_nonMembers = pcc_crossDf.loc[(pcc_crossDf['labels'] == '0')]
pcc_Members = pcc_crossDf.loc[(pcc_crossDf['labels'] == '1')]


print(searchDf_specNonMembers.shape[0])
print(searchDf_specMembers.shape[0])
print(pcc_nonMembers.shape[0])
print(pcc_Members.shape[0])
print(f'Total: {searchDf_specNonMembers.shape[0] + searchDf_specMembers.shape[0] + pcc_nonMembers.shape[0] + pcc_Members.shape[0]} ')
# pcc_Members


# In[6]:


# print(trainObjs.isnull().any())
# plt.figure()
# plt.hist(trainObjs['labels'])
# plt.show()


# In[7]:


# pcc_crossDf[['objID', 'ra', 'dec']]
trainObjs = pd.concat([pcc_crossDf[['objID', 'ra', 'dec', 'labels']],
                      searchDf_specNonMembers[['objID', 'ra', 'dec', 'labels']],
                      searchDf_specMembers[['objID', 'ra', 'dec', 'labels']]],
                      keys = ('PCC', 'SpecBg', 'SpecMems')) # want to preserve where these come from

# trainObjs = trainObjs.drop_duplicates(subset = 'objID')
trainObjs['labels'] = pd.to_numeric(trainObjs['labels'], downcast='integer')

img_width, img_height = 200, 200
SkyServer_DataRelease = 'DR16'

dirName = 'PCC-and-SpecSearch'
outDir = os.path.join('..', 'Images', dirName)

fileList = list()
if not os.path.exists(outDir):
   os.makedirs(outDir)
    
if len(glob.glob(os.path.join(outDir, '*.png'))) == trainObjs.shape[0]:
    print('Skipping Populate')
else:
    # for id, r, d in zip(searchDf['objID'], trainObjs['ra'], trainObjs['dec']):
    for id, r, d, l in zip(trainObjs['objID'], trainObjs['ra'], trainObjs['dec'], trainObjs['labels']):
        img_array = SkyServer.getJpegImgCutout(ra=r, dec=d, width=img_width, height=img_height, scale=0.1, 
                                     dataRelease=SkyServer_DataRelease)
        # print(f'{id}-label={labeler(z)}')
        # outPicTemplate = f'{id}-label={labeler(z)}.png'
        outPicTemplate = f'{id}-label={l}.png'
        
        img0 = PIL.Image.fromarray(img_array, 'RGB')
        img0.save(f'{outDir}/{outPicTemplate}')
        fileList.append(f'{outPicTemplate}')

print(f'Finished populate with {len(fileList)} images')


# In[8]:


trainObjs.to_csv('trainObjs')
# trainObjs = trainObjs.drop_duplicates()
trainObjs['files'] = fileList
trainObjs.to_csv('trainObjs')
trainObjs


# In[9]:


outDir


# In[10]:


# remove red contaminants
# downFiles = downSampleDf1['files']
downFiles = trainObjs['files']
# trainObjs['files'] = downFiles
# downFiles = trainObjs['files']
redPercent = [None]*len(downFiles)

counter = 0
# workDir = '../Images/SDSS-png/'
# workDir = os.path.join('..', 'Images', 'SDSS-png')
workDir = outDir

# lower boundary RED color range values; Hue (0 - 10)
lower1 = np.array([0, 80, 20])
upper1 = np.array([10, 255, 255])
 
# upper boundary RED color range values; Hue (160 - 180)
lower2 = np.array([160, 100, 20])
upper2 = np.array([179, 255, 255])

for i, x in enumerate(downFiles):
    # testImgPath = x
    testImgPath = os.path.join(workDir, x)
    image = cv2.imread(testImgPath)
    result = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # check image for pixels on the lower and upper end of hsv (red is weird for hsv)
    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)
    full_mask = lower_mask + upper_mask;

    result = cv2.bitwise_and(result, result, mask=full_mask)
    dim = np.shape(full_mask)[0]
    counts = np.count_nonzero(full_mask)
    percent = 100*counts/dim**2
    redPercent[i] = percent
    subtitle_string = f'{percent}% of the image is red'
    filename = testImgPath.split('\\')[-1]

trainObjs['reds'] = redPercent # add new column of the red percentage of an image
# downSampleDf1
redList = (trainObjs['reds'] >= 50) # percentage threshold of how much red is in the image
# downSampleDf1.shape
df_filtered = trainObjs[trainObjs['reds'] <= 50]

## TODO -- Why are these different per run?
print(f'{trainObjs.shape[0]} images before filtering')
print(f'{df_filtered.shape[0]} images after filtering')

## Uncomment to see red images 
# redInds = np.where(redList)[0] # the indices of the hot pixel images to be removed
# print(len(redInds))
# for n in redInds:
#     red = downSampleDf1['files'].to_list()[n]
#     imStr = '../Images/SDSS-png/' + red
#     im = cv2.imread(imStr)[:,:,::-1] # [:,:,::-1] switches rgb to bgr and vice versa
#     plt.figure
#     plt.imshow(im)
#     plt.show()



# In[11]:


# brightDF


# In[12]:


# want to clean /rotations-png/test and /rotations-png/train/ every run 
imgDirectory = '../Images/rotations-png'
testPath = os.path.join(imgDirectory, 'test', '*')
testImgs = glob.glob(testPath)
trainPath = os.path.join(imgDirectory, 'train', '*')
trainImgs = glob.glob(trainPath)

# testImgs = glob.glob(testDir)
for x in testImgs:
    os.remove(x)
# can't do it all in one loop since in wrong dir
for y in trainImgs:
    os.remove(y)


# In[13]:


# # Generate Rotation data

def applyRotations(originalDf, outDir, greyFlag):

    # files and labels as numpy arrays
    files = originalDf['files'].to_numpy()
    label = originalDf['labels'].to_numpy()
    
    rotDir = '../Images/rotations-png'
    # originalDir = '../Images/SDSS-png/'
    originalDir = '../Images/PCC-and-SpecSearch/'
    # originalDir = workDir

    rotFilenames = list()
    rotLabels = list()

    #angle = [90, 180, 270, 360]
    angle = [30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360]

    # Use PIL to rotate image on angles in list
    for ang in angle:
        for f, l in zip(files, label):
            imgString = originalDir + f
            im = PIL.Image.open(imgString)
            
            if greyFlag == True:
                im = im.convert('L')
            out = im.rotate(ang)
           
            # generated filename
            # outString = f'{rotDir}/{outDir}/{f[:-5]}_rot{ang}_label={l}.png'
            outString = f'{rotDir}/{outDir}/{f[:-5]}_rot{ang}.png'
            
            # filename relative to working directory
            # dfString = f'{outDir}/{f[:-5]}_rot{ang}_label={l}.png'
            dfString = f'{outDir}/{f[:-5]}_rot{ang}.png'

            out.save(outString)
            rotFilenames.append(dfString)
            rotLabels.append(l)

            rotationDf = pd.DataFrame({'files': rotFilenames,
                                    'labels': rotLabels})

    return(rotationDf)

def applyRotations2(originalDf, outDir):
    
    # files and labels as numpy arrays
    files = originalDf['files'].to_list()
    label = originalDf['labels'].to_list()
    
    rotDir = '../Images/rotations-png'
    # originalDir = '../Images/SDSS-png/'
    originalDir = '../Images/PCC-and-SpecSearch/'
    # originalDir = workDir

    rotFilenames = list()
    rotLabels = list()

    #angle = [90, 180, 270, 360]
    angle = [30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360]

    # Use PIL to rotate image on angles in list
    for ang in angle:
        for f, l in zip(files, label):
            print(f, l)


# In[14]:


# df_filtered
df_filtered['labels'] = pd.to_numeric(df_filtered['labels'], downcast='integer')
df_filtered['labels'] = df_filtered['labels'].astype(str)


# In[15]:


# # Train/Test Split 
X = df_filtered['files']
y = df_filtered['labels']
# X = trainObjs['files']
# y = trainObjs['labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

trainDf = pd.DataFrame({'files' : X_train,
                        'labels': y_train})
testDf = pd.DataFrame({'files' : X_test,
                        'labels': y_test})

# ss = trainDf['files'].to_list()[0]
# ss[-12:]
# os.path.basename(ss)
greyFlag = False
trainDf_rot = applyRotations(trainDf, 'train', greyFlag)
testDf_rot = applyRotations(testDf, 'test', greyFlag)
# trainDf_rot = applyRotations2(trainDf, 'train')
# testDf_rot = applyRotations2(testDf, 'test')
# trainDf.to_csv('trainDf')


# In[32]:


# checkBalance(trainDf)
# checkBalance(trainDf_rot)
trainDf_rot
testDf_rot


# In[17]:


# # Create datasets with flow from dataframe
IMG_WIDTH = 200
IMG_HEIGHT = 200
TRAIN_BATCH_SIZE = 20
VAL_BATCH_SIZE = 20

#imgDirectory = "./rotations/"
imgDirectory = "../Images/rotations-png/"
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

train_generator=datagen.flow_from_dataframe(
dataframe=trainDf_rot,
directory=imgDirectory,
x_col="files",
y_col="labels",
subset="training",
batch_size=TRAIN_BATCH_SIZE, # divisibility
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(IMG_WIDTH,IMG_HEIGHT))

validation_generator=datagen.flow_from_dataframe(
dataframe=trainDf_rot,
directory=imgDirectory,
x_col="files",
y_col="labels",
subset="validation",
batch_size=VAL_BATCH_SIZE,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(IMG_WIDTH,IMG_HEIGHT))

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=testDf_rot,
directory=imgDirectory,
x_col="files",
y_col=None,
batch_size=1,
seed=42,
shuffle=False,
class_mode=None,
target_size=(IMG_WIDTH,IMG_HEIGHT))


# # ResNet50 Model
#
# https://github.com/suvoooo/Learn-TensorFlow/blob/master/resnet/Implement_Resnet_TensorFlow.ipynb

def res_identity(x, filters):
    x_skip = x # this will be used for addition with the residual block
    f1, f2 = filters

    #first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    #second block # bottleneck (but size kept same with padding)
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block activation used after adding the input
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    # x = Activation(activations.relu)(x)

    # add the input
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x

def res_conv(x, s, filters):

    x_skip = x
    f1, f2 = filters

    # first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
    # when s = 2 then it is like downsizing the feature map
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # second block
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    #third block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # shortcut
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
    x_skip = BatchNormalization()(x_skip)

    # add
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x


def resnet50():

    input_im = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)) # cifar 10 images size
    x = ZeroPadding2D(padding=(3, 3))(input_im)

    # 1st stage
    # here we perform maxpooling, see the figure above

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    #2nd stage
    # frm here on only conv block and identity block, no pooling

    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

    # 3rd stage

    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    # 4th stage

    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    # 5th stage

    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))

    # ends with average pooling and dense connection

    x = AveragePooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2, activation='softmax', kernel_initializer='he_normal')(x) #multi-class

    # define the model

    model = Model(inputs=input_im, outputs=x, name='Resnet50')

    return model


# In[18]:


# s1 = '../Models/pcc_resnet50_6'
# s1 = 
# s1.split


# In[19]:


cnn_model = resnet50()
# BATCH_SIZE = 1

modelName = 'pcc_resnet50_6' # template is currently pcc_resnet50_X, where X is the iteration of the model
modelStr = os.path.join('..', 'Models', modelName) 

### Hyperparameters ###
n_epochs = 300
init_lr = 7.5e-2
# init_lr = float(sys.argv[1])
decay_rate = 0.99
# decay_rate = float(sys.argv[2])
decay_steps = 100_000
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate = init_lr,
                    decay_steps = decay_steps,
                    decay_rate = decay_rate)


cnn_model.compile(loss='categorical_crossentropy',
                  optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                  metrics = ['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = cnn_model.fit(train_generator,
                        epochs = n_epochs,
                        callbacks = [es],
                        verbose = 0,
                        validation_data=validation_generator)



train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

# learning_rate = history.history['']



# In[20]:


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

ax[0].set_title('Training Accuracy vs. Epochs')
ax[0].plot(train_accuracy, 'o-', label='Train Accuracy')
ax[0].plot(val_accuracy, 'o-', label='Validation Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend(loc='best')
ax[0].grid()

ax[1].set_title('Training/Validation Loss vs. Epochs')
ax[1].plot(train_loss, 'o-', label='Train Loss')
ax[1].plot(val_loss, 'o-', label='Validation Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].set_ylim([-1, 100])
ax[1].legend(loc='best')

# ax[2].set_title('Learning Rate vs. Epochs')
# ax[2].plot(learning_rate, 'o-', label='Learning Rate')
# ax[2].set_xlabel('Epochs')
# ax[2].set_ylabel('Learning Rate')
# ax[2].legend(loc='best')

# ax[3].set_title('Loss vs learning rate')
# # ax[3].plot(learning_rate, 'o-', label='Learning Rate')
# ax[3].plot(learning_rate, train_loss, 'o-', label='Train Loss')
# ax[3].plot(learning_rate, val_loss, 'o-', label='Validation Loss')
# ax[3].set_ylabel('Loss')
# ax[3].set_xlabel('Learning Rate')
# ax[3].legend(loc='best')

plt.suptitle(f'Initial LR = {init_lr} Decay Rate = {decay_rate}')
plt.tight_layout()
plt.savefig(f'{modelName}-train-report_init-lr={init_lr}_decay-rate={decay_rate}_offrots.png')
plt.show()


# fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 10))
# idx = 0

# for i in range(2):
#     for j in range(5):
#         predicted_label = unique_labels[np.argmax(predictions[idx])]
#         ax[i, j].set_title(f"{predicted_label}", fontsize=10)
#         ax[i, j].imshow(test_generator[idx][0])

#         ax[i, j].axis("off")
#         idx += 1

# # plt.tight_layout()
# plt.suptitle("Test Dataset Predictions", fontsize=20)
# plt.show()


# In[34]:


# In[21]:


predictions = cnn_model.predict(test_generator)
test_loss, test_accuracy = cnn_model.evaluate(validation_generator, batch_size=1) # needs to be divisible

y_pred = np.argmax(predictions, axis=1)
y_true = testDf_rot['labels'] # this needs to be checked if you change the input dataframes
y_true = y_true.tolist()
# len(y_pred) == len(y_true)
unique_labels = {value: key for key, value in train_generator.class_indices.items()}

# print("Label Mappings for classes present in the training and validation datasets\n")
# for key, value in unique_labels.items():
#     print(f"{key} : {value}")

# function to return key for any value
def get_key(val):
    for key, value in unique_labels.items():
        if val == value:
            return key

    return "key doesn't exist"

Y_true = []
# for i in range(len(y_true)): # This was the original way to do it -- be careful, this only solved a mismatch and could be wrong
for i in range(len(y_pred)):
    Y_true.append(get_key(y_true[i]))

cf_mtx = confusion_matrix(Y_true, y_pred)

group_counts = ["{0:0.0f}".format(value) for value in cf_mtx.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_mtx.flatten()/np.sum(cf_mtx)]
box_labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
box_labels = np.asarray(box_labels).reshape(2, 2)


# cf_mtx.sum()

y_true = np.array([int(x) for x in y_true]) # cast to np array for type consistency with y_pred
errors = (y_true - y_pred != 0) # everywhere the numbers don't match
y_true_errors = y_true[errors]
y_pred_errors = y_pred[errors]

test_images = test_generator.filenames
test_img_err = np.asarray(test_images)[errors]


# fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(12, 10))
# idx = 0

hits = (y_true - y_pred == 0)
y_true_hits = y_true[hits]
y_pred_hits = y_pred[hits]

test_img_hits = np.asarray(test_images)[hits]


# In[22]:


plt.figure(figsize = (12, 10))
# sns.heatmap(cf_mtx, xticklabels=labels.values(), yticklabels=labels.values(),
#            cmap="YlGnBu", fmt="", annot=box_labels)
sns.heatmap(cf_mtx, cmap="YlGnBu", fmt="", annot=box_labels)
plt.xlabel('Predicted Classes')
plt.ylabel('True Classes')
#plt.show()

plt.title(f"Test Accuracy: {test_accuracy*100:.2f}")
plt.savefig(f'{modelName}-confusion-matrix_init-lr={init_lr}_decay-rate={decay_rate}_offrots.png')

# for i in range(3):
#     for j in range(5):
#         idx = np.random.randint(0, len(test_img_err))
#         true_index = y_true_errors[idx]
#         true_label = unique_labels[true_index]
#         predicted_index = y_pred_errors[idx]
#         predicted_label = unique_labels[predicted_index]
#         ax[i, j].set_title(f"True Label: {true_label} \n Predicted Label: {predicted_label}")
#         ax[i, j].imshow(test_generator[idx][0])
#         ax[i, j].axis("off")

# plt.tight_layout()
# plt.suptitle('Wrong Predictions made on test set', fontsize=15)
# plt.show()


fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(12, 10))
idx = 0

for i in range(3):
    for j in range(5):
        idx = np.random.randint(0, len(test_img_hits))
        true_index = y_true_hits[idx]
        true_label = unique_labels[true_index]
        predicted_index = y_pred_hits[idx]
        predicted_label = unique_labels[predicted_index]
        ax[i, j].set_title(f"True Label: {true_label} \n Predicted Label: {predicted_label}")
        ax[i, j].imshow(test_generator[idx][0])
        ax[i, j].axis("off")

plt.tight_layout()
plt.suptitle('True Predictions made on test set', fontsize=15)
plt.show()

# # End result:
print(f'Init conds:')
print(f'init lr: {init_lr}, decay rate: {decay_rate}, decay steps: {decay_steps}')
print(f"Test Loss:     {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

## save model weights ## 
cnn_model.save(f'../Models/{modelStr}.h5')


# In[27]:


# conv_layers = [None]*53 # 53 layers
# ind = 0
# for layer in cnn_model.layers:
    
#     # check for convolutional layer
#     if 'conv' not in layer.name:
#         continue
        
#     else:
#         # get filter weights
#         filters, biases = layer.get_weights()
#         print(layer.name, filters.shape, layer.output.shape)
#         conv_layers[ind] = layer.name
#         ind += 1
# len(conv_layers)


# In[28]:


# filters, biases = cnn_model.layers[9].get_weights()

# # normalize filter values to 0-1 so we can visualize them
# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min) / (f_max - f_min)


# In[29]:


# # plot first few filters
# n_filters, ix = 6, 1
# for i in range(n_filters):
#     # get the filter
#     f = filters[:, :, :, i]
    
# # plot each channel separately
# for j in range(3):
#     # specify subplot and turn of axis
#     ax = plt.subplot(n_filters, 3, ix)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     # plot filter channel in grayscale
#     plt.imshow(f[:, :, j], cmap='gray')
#     ix += 1
#     # show the figure
#     plt.show()


# In[30]:


# ## are these redundant?? TODO -- Clean!

# trainPath = os.path.join('rotations-png', 'train', '*')
# trainImgs = glob.glob(trainPath)
# testPath = os.path.join('rotations-png', 'test', '*')
# testImgs = glob.glob(testPath)

# maps_model = Model(cnn_model.inputs, outputs=cnn_model.layers[9].output)
# maps_model.summary()
# exImage = load_img(testImgs[0])
# copyImg = exImage.copy()
# exImage = img_to_array(exImage)
# exImage = np.expand_dims(exImage, axis = 0)
# exImage = preprocess_input(exImage)

# feature_maps = maps_model.predict(exImage)
# # feature_maps
# # np.shape(exImage)


# In[31]:


# square = 8
# ix = 1
# for _ in range(square):
#     for _ in range(square):
#         # specify subplot and turn of axis
#         ax = plt.subplot(square, square, ix)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # plot filter channel in grayscale
#         plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
# #         plt.imshow((feature_maps[0, :, :, ix-1]))
#         ix += 1
# # show the figure
# plt.show()


# In[ ]:


# plt.imshow(copyImg)


# In[ ]:


# print(testImgs[0])


# In[ ]:




