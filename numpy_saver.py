from PIL import Image
import os, numpy as np
from tqdm import tqdm
import pickle as pkl

folder = 'images'
read = lambda imname: np.asarray(Image.open(imname).resize((80, 80)))

ims = []
for filename in tqdm(range(82783), total=82783):
    try:
        ims.append(read(os.path.join(folder, str(filename) + '.jpg')))

    except FileNotFoundError:
        ims.append(np.zeros((80, 80, 3), dtype='uint8'))

im_array = np.array(ims, dtype='uint8')

with open('train_images.npy', 'wb') as file:
    np.save(file, im_array)