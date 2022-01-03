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

with h5py.File('eee443_project_dataset_train.h5', 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
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
    word_code = np.array(word_code)

    """test_images = list(f[testims])
    test_labels = np.array(list(f[testlbls])).reshape(-1, 1)
    test_labels[np.where(test_labels == 0)] = -1  # change the label 0 to -1
    train_images = list(f[trainims])
    train_labels = np.array(list(f[trainlbls])).reshape(-1, 1)
    train_labels[np.where(train_labels == 0)] = -1  # change the label 0 to -1
    train_images = np.array([np.transpose(i) / 255 for i in train_images]).reshape(-1, 1024)
    test_images = np.array([np.transpose(i) / 255 for i in test_images]).reshape(-1, 1024)"""