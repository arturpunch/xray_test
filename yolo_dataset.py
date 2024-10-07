import shutil
import os
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


cfg = OmegaConf.load("config.yaml")
labels = pd.read_csv('./data/BBox_List_2017.csv')

unique_images = labels['Image Index'].unique()
prev_patient_id = '0'
patient_dict = {}

for i in os.listdir(cfg.original_images_dir):
    if i in unique_images:
        patient_id = i.split('_', 1)[0]
        if patient_id not in patient_dict.keys():
            if np.random.sample() <= cfg.val_rate:
                prefix = 'val'
            else:
                prefix = 'train'
            patient_dict[patient_id] = prefix
        else:
            prefix = patient_dict[patient_id]       
    else:
        prefix = 'test'
    # shutil.copy(f'./data/n/images/{i}', f'./data/images/{prefix}/')
    shutil.move(f'./data/n/images/{i}', f'./data/images/{prefix}/')

print('train', len(os.listdir('./data/images/train/')))
print('test', len(os.listdir('./data/images/test/')))
print('val', len(os.listdir('./data/images/val/')))

trains = list(os.listdir('./data/images/train/'))
vals = list(os.listdir('./data/images/val/'))
tests = list(os.listdir('./data/images/test/'))
   
for n in unique_images:
    l = labels.loc[labels['Image Index'] == n].to_numpy()
    bbox = []
    for row in range(len(l)):
        bbox.append([cfg.labels_dict[l[row][1]], (l[row][2] + l[row][4] / 2) / cfg.image_width, 
                     (l[row][3] + l[row][5] / 2) / cfg.image_height, l[row][4] / cfg.image_width, l[row][5] / cfg.image_height])

    if n in trains:
        prefix = 'train'
    elif n in vals:
        prefix = 'val'
    elif n in tests:
        prefix = False
    else:
        prefix = False
        # raise(Exception)
    
    if prefix:
        with open(f'./data/labels/{prefix}/{n.rsplit(".", 1)[0]}.txt', 'w') as f:
            for b in bbox:
                f.write(f'{b[0]} {b[1]} {b[2]} {b[3]} {b[4]}\n')
