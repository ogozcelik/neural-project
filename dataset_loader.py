"""
Subject:
    The posted dataset is designed for an image captioning task,
    so you are expected to develop a model that generates the caption for an input image.

Dataset Features:
    train_cap: Captions for training images
                (contains 17 indices from the vocabulary for each image,
                note that a single image can have multiple captions provided by different observers)

    train_imid: The indices of training images (since a single image can be captioned several times,
                this image index is required to relate each caption with the corresponding image)

    train_url: Flickr URLs for training images (URLs of actual training images for you to download the data)

    word_code: dictionary for converting words to vocabulary indices

    train_ims: Pretrained feature vector for training images (use is optional,
                but if you use it your project score will be halved!!!)

    test_cap: Captions for testing images

    test_imid: The indices of testing images

    test_url: Flickr URLs for testing images

    test_ims: Pretrained feature vector for testing images (use is optional,
              but if you use it your project score will be halved!!!)

You are supposed to sub-divide the training data
(eee443_project_dataset_train.h5) into (85, 15)% (training, validation)
Test your final trained model/models on the testing data (eee443_project_dataset_test.h5).

"""

import h5py
import numpy as np
from urllib import error

"""with h5py.File('eee443_project_dataset_train.h5', 'r') as f:
    # List all groups
    # print("Keys: %s" % f.keys())
    train_cap = list(f.keys())[0]
    train_imid = list(f.keys())[1]
    train_ims = list(f.keys())[2]
    train_url = list(f.keys())[3]
    word_code = list(f.keys())[4]
    # get the data
    train_cap = list(f[train_cap])  # images
    train_imid = list(f[train_imid])
    train_ims = list(f[train_ims])
    train_url = list(f[train_url])
    word_code = list(f[word_code])

    train_cap = np.array(train_cap)
    train_imid = np.array(train_imid)
    train_ims = np.array(train_ims)
    word_code = np.array(word_code)"""

with h5py.File('eee443_project_dataset_test.h5', 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    test_url = list(f.keys())[3]
    # get the data
    test_url = list(f[test_url])


from tqdm import tqdm
import requests
import multiprocessing as mp

print("Number of cpu : ", mp.cpu_count())


def my_func(caption):
    url = train_url[caption].decode("utf-8")
    res = requests.get(url)

    if res.ok:
        file = open("test_images/" + str(caption) + ".jpg", "wb")
        file.write(res.content)
        file.close()
    else:
        print("Image ", caption, " url is not working!")
        pass


def main():
    pool = mp.Pool(7)
    for _ in tqdm(pool.imap_unordered(my_func, range(len(train_url))), total=len(train_url)-71706):
        pass


if __name__ == "__main__":
    main()

# train_images = np.zeros((80, 80, 3), dtype=np.short)

"""train_captions = []
for i in range(82783):
    train_captions.append([])
"""

"""for ix in tqdm(range(len(train_cap)), total=len(train_cap)):
    caption = train_imid[ix]  # here the caption
    if caption in old_captions:  # if image is already loaded
        train_captions[caption].append(train_cap[ix])
        train_captions[ix].append(train_cap[ix])
    else:  # the image has not been loaded yet, then download it
        old_captions.append(caption)  # append the caption
        url = train_url[caption].decode("utf-8")
        try:
            #img = io.imread(url)
            res = requests.get(url)
        except error.HTTPError as e:
            not_working_urls += 1
            print("Image ", ix, " url is not working!")
            continue
        img = Image.open(BytesIO(res)).resize((80, 80))
        #img = resize(img, (80, 80, 3))
        train_images = np.append(train_images, img, axis=0)
        train_captions[caption].append(train_cap[ix])
        #plt.figure()
        #plt.imshow(img)
        #plt.show()

print(not_working_urls, " urls are broken!")"""

"""for ix in tqdm(range(len(train_cap)), total=len(train_cap)):
    caption = train_imid[ix]  # here the caption
    if caption in old_captions:  # if image is already loaded
        continue
    else:  # the image has not been loaded yet, then download it
        old_captions.append(caption)  # append the caption
        url = train_url[caption].decode("utf-8")

        res = requests.get(url)

        if res.ok:
            file = open("images/" + str(caption) + ".jpg", "wb")
            file.write(res.content)
            file.close()
        else:
            not_working_urls += 1
            print("Image ", ix, " url is not working!")
            continue
        img = Image.open(BytesIO(res)).resize((80, 80))
        #img = resize(img, (80, 80, 3))
        train_images = np.append(train_images, img, axis=0)
        train_captions[caption].append(train_cap[ix])
        #plt.figure()
        #plt.imshow(img)
        #plt.show()"""
