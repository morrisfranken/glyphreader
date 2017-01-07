'''
Created on 25 Dec 2016

@author: Morris Franken
Loads a batch of images and prepares them for forwarding into a keras deep net.
'''
import numpy as np
from multiprocessing.pool import Pool
from keras.preprocessing import image

def loadImage(path):
    img = image.load_img(path[0], target_size=(299, 299))
    x = image.img_to_array(img)
    x /= 127.5
    x -= 1
    return x

def loadBatch(img_paths):
    with Pool(processes=8) as pool:
        imgs = pool.map(loadImage, zip(img_paths, range(len(img_paths))))
        return np.asarray(imgs)

# Use this for training, instead of loading everything into memory, in only loads chunks
def batchGenerator(img_paths, labels, batch_size):
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:(i + batch_size)]
        batch_labels = labels[i:(i + batch_size)]
        batch_images = loadBatch(batch_paths)
        yield batch_images, batch_labels
