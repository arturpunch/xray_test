from ultralytics import YOLO
import csv
import os
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm

cfg = OmegaConf.load("config.yaml")
model = YOLO(cfg.test_weights)

with open('./data/Data_Entry_2017_v2020.csv') as f:
    reader = csv.reader(f, delimiter=",", quotechar='"')
    meta = [row for row in reader][1:]

tp = [0] * len(cfg.labels_dict.keys())
tn = [0] * len(cfg.labels_dict.keys())
fp = [0] * len(cfg.labels_dict.keys())
fn = [0] * len(cfg.labels_dict.keys())
t = 0

test_images = list(os.listdir(cfg.test_dir))

for m in tqdm(meta):
    if m[0] in test_images:
        classes = m[1].split('|', -1)
        if set(classes).issubset(cfg.labels_dict.keys()) or classes[0] == 'No Finding':
            t += 1
            results = model(f'{cfg.test_dir}{m[0]}', imgsz=[cfg.image_width, cfg.image_height], verbose=False)

            found_classes = [results[0].names[int(c)] for c in results[0].boxes.cls]

            if classes[0] != 'No Finding':
                for c in cfg.labels_dict.keys():
                    if c in found_classes and c in classes:
                        tp[cfg.labels_dict[c] - 1] += 1
                    elif c in found_classes and c not in classes:
                        fp[cfg.labels_dict[c] - 1] += 1
                    elif c not in found_classes and c not in classes:
                        tn[cfg.labels_dict[c] - 1] += 1
                    elif c not in found_classes and c in classes:
                        fn[cfg.labels_dict[c] - 1] += 1
            else:
                for c in cfg.labels_dict.keys():
                    if c in found_classes:
                        fp[cfg.labels_dict[c] - 1] += 1
                    else:
                        tn[cfg.labels_dict[c] - 1] += 1

print('total', t)
for i, c in enumerate(cfg.labels_dict.keys()):
    print(c, tp[i], tn[i], fp[i], fn[i])