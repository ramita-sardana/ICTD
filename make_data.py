from os import replace
from unittest.mock import patch
from osgeo import gdal
import numpy as np
import pickle
from random import sample

def get_array(filename):
    ds = gdal.Open(filename)
    img_array = np.zeros((ds.RasterCount, ds.RasterYSize, ds.RasterXSize))

    for i in range(ds.RasterCount):
        img_array[i,:,:] = np.array(ds.GetRasterBand(i+1).ReadAsArray())
    # print(img_array.shape)
    return img_array

def make_patches(arr_2019, arr_2020, arr_deforestation, size=15, stride=3, threshhold = 0.3, sampling=False):
    #initialize constants and arrays
    bands, ysize, xsize = arr_2019.shape[0], arr_2019.shape[1], arr_2019.shape[2]
    ypatches = int((ysize-size+stride)//stride)
    xpatches = int((xsize-size+stride)//stride)
    patches_2019 = np.zeros((xpatches*ypatches, bands, size, size))
    patches_2020 = np.zeros((xpatches*ypatches, bands, size, size))
    labels = np.zeros((xpatches*ypatches, 1))
    
    #read data
    for i in range(0, ysize-size+1, stride):
        for j in range(0, xsize-size+1, stride):
            patches_2019[(i*xpatches+j)//stride] = arr_2019[:, i:i+size, j:j+size]
            patches_2020[(i*xpatches+j)//stride] = arr_2020[:, i:i+size, j:j+size]
            labels[(i*xpatches+j)//stride, 0] = 1 if (np.count_nonzero(arr_deforestation[0, i:i+size, j:j+size])/size**2) > threshhold else 0
    
    #augment, downsample for train
    if sampling:
        true_indices = np.nonzero(labels)[0]
        false_indices = np.nonzero(1-labels)[0]

        patches_2019_true = patches_2019[true_indices, :, :, :]
        patches_2019_true = np.vstack((patches_2019_true, 
                                        patches_2019_true[:, :, ::-1, :], 
                                        patches_2019_true[:, :, :, ::-1],
                                        patches_2019_true[:, :, ::-1, ::-1]))
        patches_2019_false = patches_2019[false_indices, :, :, :]
        if patches_2019_true.shape[0] > patches_2019_false.shape[0]:
            random_undersample = np.arange(patches_2019_false.shape[0])
        else:
            random_undersample = np.random.choice(patches_2019_false.shape[0], patches_2019_true.shape[0], replace=False)
        patches_2019_false = patches_2019_false[random_undersample, :, :, :]
        patches_2019 = np.vstack((patches_2019_true, patches_2019_false))

        patches_2020_true = patches_2020[true_indices, :, :, :]
        patches_2020_true = np.vstack((patches_2020_true, 
                                        patches_2020_true[:, :, ::-1, :], 
                                        patches_2020_true[:, :, :, ::-1],
                                        patches_2020_true[:, :, ::-1, ::-1]))
        patches_2020_false = patches_2020[false_indices, :, :, :]
        patches_2020_false = patches_2020_false[random_undersample, :, :, :]
        patches_2020 = np.vstack((patches_2020_true, patches_2020_false))

        labels = np.vstack((np.ones((patches_2020_true.shape[0], 1)), 
                            np.zeros((patches_2020_false.shape[0], 1))))
    else:
        #possible downsample for test, val
        pass

    return patches_2019, patches_2020, labels

def create_set_from_list(tiles, train=False, path="./"):
    ret = [[], [], []]
    for i in tiles:
        y = int(300*(i//6))
        x = int(300*(i%6))
        arr_2019 = get_array(path + f"tiles/tile_2019_{x}_{y}.tif")
        arr_2020 = get_array(path + f"tiles/tile_2020_{x}_{y}.tif")
        arr_deforestation = get_array(path + f"tiles/deforestation_{x}_{y}.tif")
        patches_2019, patches_2020, labels = make_patches(arr_2019, arr_2020, arr_deforestation, sampling=train)
        ret[0].append(patches_2019)
        ret[1].append(patches_2020)
        ret[2].append(labels)
    
    ret[0] = np.vstack(ret[0])
    ret[1] = np.vstack(ret[1])
    ret[2] = np.vstack(ret[2])
    return ret


def create_train_test_split():

    all = [i for i in range(24)]
    train = sample(all, 10)
    all = [i for i in all if i not in train]
    val = sample(all, 2)
    test = [i for i in all if i not in val]


    #return train, val and test tuples containing (2019_patches, 2020_patches, labels)
    return create_set_from_list(train, True), create_set_from_list(val), create_set_from_list(test)

train, val, test = create_train_test_split()
with open('data.pkl', 'wb') as f:
    pickle.dump({"train":train, "val":val, "test":test}, f)