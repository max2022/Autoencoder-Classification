#!/usr/bin/env python
# coding: utf-8

# In[26]:


import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from umap import UMAP
import tensorflow as tf
import hdbscan
import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import pickle
import chainer
import chainer.functions as F
from scipy.stats import entropy
import dill
import numpy
import tensorflow as tf
# Display the version
print(tf.__version__)
# other imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Input, Dense, Reshape, Conv2DTranspose,   Activation, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar100, cifar10
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2DTranspose
from keras.models import Sequential
import numpy as np
import os
from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Dropout
from keras.layers import Flatten, Activation
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras import backend as K

from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
get_ipython().run_line_magic('matplotlib', 'inline')

import skimage
from skimage.util import img_as_ubyte
from scipy.stats import entropy
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


# In[27]:


import os
import sys
import glob
import numpy
import numpy as np
from six.moves import cPickle as pickle
from scipy import linalg
from skimage.color import rgb2luv
from skimage import img_as_float


# In[28]:


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/cifar-10-batches-py/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels,         cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names


if __name__ == "__main__":
    """show it works"""
    data_dir ="B_net/datasets/data/cifar10"
    cifar_10_dir = 'cifar-10-batches-py'

    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names =         load_cifar_10_data(data_dir)

    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)

    # Don't forget that the label_names and filesnames are in binary and need conversion if used.

    # display some random training images in a 25x25 grid
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, train_data.shape[0])
            ax[m, n].imshow(train_data[idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()


# In[29]:


x_train, train_filenames, y_train, x_test, test_filenames, y_test, label_names =         load_cifar_10_data(data_dir)


# In[30]:


def get_data():
    cifar_x_train=x_train
    cifar_x_test =x_test
    cifar_y_train = y_train
    cifar_y_test = y_test
    cifar_x_train, cifar_x_test = x_train / 255.0, x_test / 255.0

    # flatten the label values
    cifar_y_train, cifar_y_test = y_train.flatten(), y_test.flatten()
    return cifar_x_train,cifar_y_train,cifar_x_test,cifar_y_test


# In[31]:


cifar_x_train,cifar_y_train,cifar_x_test,cifar_y_test= get_data()


# In[11]:


import tensorflow.keras as K


# In[14]:


cifar_vgg_x_train= K.applications.vgg16.preprocess_input(x_train)
cifar_vgg_y_train= K.utils.to_categorical(cifar_y_train, 10)
cifar_vgg_x_test= K.applications.vgg16.preprocess_input(x_test)
cifar_vgg_y_test= K.utils.to_categorical(y_test, 10)


# In[12]:


plt.figure(figsize=(20,20))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(cifar_x_train[i].reshape(32,32,3), cmap=plt.cm.binary)
    #plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[15]:


plt.figure(figsize=(20,20))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(cifar_vgg_x_train[i].reshape(32,32,3), cmap=plt.cm.binary)
    #plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[67]:


plt.figure(figsize=(20,20))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(cifar_vgg_x_train[i].reshape(32,32,3))
    #plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[9]:


import dill


# In[16]:


with open("models/vgg16_cifar10_transfer_learning.bn", "rb") as f:
    model_vgg = dill.load(f)


# In[17]:


batch_size = 32
score = model_vgg.evaluate(cifar_vgg_x_test, cifar_vgg_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[24]:


def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=False,
                              sqrt_bias=0., min_divisor=1e-8):
    """
    Global contrast normalizes by (optionally) subtracting the mean
    across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).
    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and \
        features indexed on the second.
    scale : float, optional
        Multiply features by this const.
    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing. \
        Defaults to `True`.
    use_std : bool, optional
        Normalize by the per-example standard deviation across features \
        instead of the vector norm. Defaults to `False`.
    sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.
    min_divisor : float, optional
        If the divisor for an example is less than this value, \
        do not apply it. Defaults to `1e-8`.
    Returns
    -------
    Xp : ndarray, 2-dimensional
        The contrast-normalized features.
    Notes
    -----
    `sqrt_bias` = 10 and `use_std = True` (and defaults for all other
    parameters) corresponds to the preprocessing used in [1].
    References
    ----------
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, numpy.newaxis]  # Makes a copy.
    else:
        X = X.copy()

    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = numpy.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = numpy.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, numpy.newaxis]  # Does not make a copy.
    return X


# In[22]:


dirname = "B_net/"
import numpy
def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data


# In[20]:


import numpy
def getBranchydata_data(gcn=1,whitening=1):
    data = np.zeros((50000, 3 * 32 * 32), dtype=np.float32)
    labels = np.zeros((50000), dtype=np.uint8)
    for i, data_fn in enumerate(
            sorted(glob.glob(dirname+'datasets/data/cifar10/data_batch*'))):
        batch = unpickle(data_fn)
        data[i * 10000:(i + 1) * 10000] = batch['data']
        labels[i * 10000:(i + 1) * 10000] = batch['labels']

    data /= 255
    mean = data.mean(axis=0)
    data -= mean

    if gcn==1:
        data = global_contrast_normalize(data,use_std=True)

    # if whitening == 1:
    #     components, meanw, data = preprocessing(data)

    data = data.reshape((-1, 3, 32, 32))
    # for i,image in enumerate(data):
    #     data[i] = rgb2luv(img_as_float(image).transpose((1,2,0))).transpose((2,0,1))


    # for i in range(50000):
    #     d = data[i]
    #     d -= d.min()
    #     d /= d.max()
    #     data[i] = d.astype(np.float32)

    train_data = data
    train_labels = np.asarray(labels, dtype=np.int32)

    test = unpickle(dirname+'datasets/data/cifar10/test_batch')
    data = np.asarray(test['data'], dtype=np.float32)

    data /= 255
    data -= mean

    if gcn==1:
        data = global_contrast_normalize(data,use_std=True)

    # if whitening == 1:
    #     mdata = data - meanw
    #     data = np.dot(mdata, components.T)

    data = data.reshape((-1, 3, 32, 32))
    # for i,image in enumerate(data):
    #     data[i] = rgb2luv(img_as_float(image).transpose((1,2,0))).transpose((2,0,1))


    # for i in range(10000):
    #     d = data[i]
    #     d -= d.min()
    #     d /= d.max()
    #     data[i] = d.astype(np.float32)

    test_data = data
    test_labels = np.asarray(test['labels'], dtype=np.int32)

    return train_data, train_labels, test_data, test_labels


# In[25]:


x_train_branchy,y_train_branchy,x_test_branchy, y_test_branchy = getBranchydata_data()


# In[19]:


fig, ax = plt.subplots(5, 5)
k = 0

for i in range(5):
    for j in range(5):
        ax[i][j].imshow(cifar_x_train[k], aspect='auto')
        k += 1

plt.show()


# In[26]:


def find_classes(x):
    x_train_subclass=[]
    y_train_subclass=[]
    index = np.where(y_train == x)
    #print(index[0])
    for indices in index[0]:
       x_train_subclass.append(x_train[indices])
       y_train_subclass.append(y_train[indices])
    #print(len(x_train_subclass))
    #plt.imshow(x_train_subclass[10].reshape(32,32,3))
    x_train_subclass = np.array(x_train_subclass)
    x_train_subclass= x_train_subclass.reshape(5000,3072)
    #print(len(x_train_subclass), len(y_train_subclass))

    return x_train_subclass,y_train_subclass,x


# In[27]:


reducer2 = UMAP(n_neighbors=15, n_components=2, n_epochs=1000,
                min_dist=0.1, local_connectivity=2, random_state=42,
              )
def Hdb_cluster(x_train_subclass,y_train_subclass,x):
   featured_subclass = reducer2.fit_transform(x_train_subclass)
   print(featured_subclass.shape)
   clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
   cluster_labels = clusterer.fit_predict(featured_subclass)
   print(clusterer.labels_)
   clusterer.labels_.max()
   return cluster_labels


# In[28]:


def extract_features(file, model):
    # load the image as a 224x224 array
    #img = load_img(file, target_size=(32,32))
    # convert from 'PIL.Image.Image' to numpy array

    img = np.array(file)
    #print(img.shape)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,32,32,3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    #print(imgx.shape)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    #print(features.shape)
    return features


# In[29]:


def kmeans_clustering(x_train_subclass,y_train_subclass,x):
      model = VGG16(weights="imagenet", include_top=False,input_tensor= tf.keras.layers.Input(shape=(32, 32, 3)))
      data=[]
      #p = r"E:\Documents\My Projects\Instagram Dashboard\Model Development\flower_features.pkl"

      # lop through each image in the dataset
      for x in range(0,len(x_train_subclass)):
      # try to extract the features and update the dictionary
      #try:
          #print(x.shape)
          feat = extract_features(x_train_subclass[x],model)
          #data.append(x)
          data.append(feat)
      # if something fails, save the extracted features as a pickle file (optional)
      #except:
      #    with open(p,'wb') as file:
      #        pickle.dump(data,file)


      # get a list of the filenames
      #filenames = np.array(list(data.keys()))


      #print(len(data))
      data = np.array(data)


      #print(data.shape)

      # get a list of just the features
      #feat = data.reshape(-1,3072)
      # reshape so that there are 210 samples of 4096 vectors
      #feat = feat.reshape(-1,3072)
      # get the unique labels (from the flower_labels.csv)
      #df = pd.read_csv('flower_labels.csv')
      #label = df['label'].tolist()
      #unique_labels = list(set(label))
      data = data.reshape(-1,512)
      ##change
      pca = PCA(n_components=100, random_state=22)
      pca.fit(data)
      x = pca.transform(data)
      #x = reducer2.fit_transform(data)
      kmeans = KMeans(n_clusters=5,random_state=22)
      kmeans.fit(x)
      labels=kmeans.labels_
      return labels


# In[24]:


import numpy as np
import pandas as pd
def train_exits():
    df = pd.read_csv('Y_train.txt', header= None, usecols=[0], sep='\t')
    train_exits= df.values.tolist()
    train_exits= np.array(train_exits)
    train_exits=train_exits.reshape(50000,1)
    return train_exits


# In[25]:


train_exits= train_exits()


# In[39]:


####here we may think of using normalized data too while finding easy images
def calculate_entropy(model,cifar_x_test):
    cifar_x_test=np.array(cifar_x_test)
    xdata = cifar_x_test.reshape(-1, 32, 32, 3)
    #ydata = cifar_y_test
    softmax_data = model.predict(xdata)
    entropy_value = np.array([entropy(s) for s in softmax_data])
    ret =0
    #print(entropy_value)
    #print(len(entropy_value))
    minimum = entropy_value[0]
    #print(entropy_value[609])
    for idx in range(1,len(entropy_value)):
        if(minimum>entropy_value[idx]):
            minimum = entropy_value[idx]
            #print(minimum)
            ret=idx

    return  ret


# In[27]:


with open("models/cifar10model.bn", "rb") as f:
    model = dill.load(f)
score = model.evaluate( cifar_x_test, cifar_y_test, batch_size=32, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[12]:


overall_acc= []
for idx in range(0,len(cifar_x_train)):
    score = model.evaluate( cifar_x_train[idx:idx+1], cifar_y_train[idx:idx+1], batch_size=32, verbose=1)
    overall_acc.append(score[1])


# In[42]:


x_train_branchy=np.array(cifar_vgg_x_train)
xdata = cifar_vgg_x_train.reshape(-1, 32, 32, 3)
#ydata = cifar_y_test
softmax_data = model_vgg.predict(xdata)
#print(softmax_data)
entropy_value = np.array([entropy(s) for s in softmax_data])
train_entroexits_test=[]
easy_entropy_data_test=[]
easy_entropy_label_test=[]

hard_entropy_data_test=[]
hard_entropy_level_test=[]


def calculate_thresholded_entropy_value(thresholds):
    count =0
    count_hard=0
    #test_set=[]
    #test_ylabel=[]
    #autoencode_set=[]
    #autoencode_label=[]
    train_entroexits=[]

    for idx in range(0,len(entropy_value)):
        #print(entropy_value[idx])
        if(entropy_value[idx]<thresholds):
            count+=1
            train_entroexits.append(0)
            #easy_entropy_data_test.append(cifar_x_test[idx:idx+1])
            #easy_entropy_label_test.append(cifar_y_test[idx:idx+1])

        else:
            count_hard+=1
            train_entroexits.append(1)
            #hard_entropy_data_test.append(cifar_x_test[idx:idx+1])
            #hard_entropy_level_test.append(cifar_y_test[idx:idx+1])



    #print("Last")
    #print(count)
    print("For threshold-------->>>>>",thresholds)
    print("easy data",count)
    print("Hard Data",count_hard)
    print()


    return train_entroexits


# In[40]:


def clustering_bucketing(x_t,labels,x,train_entroexits):
    #print("begin Cluster")
    groups = {}
    for file, cluster in zip(x_t,labels):
        if cluster not in groups.keys():
          groups[cluster] = []
          groups[cluster].append(file)
        else:
          groups[cluster].append(file)
    cluster_list=[]
    for idx in set(labels):
        #print(idx)
        cluster_indexes = "cluster_indexes"
        cluster_indexes= cluster_indexes+str(idx)
        #cluster_indexes= np.where(labels == idx)
        #print(cluster_indexes)
        cluster_list.append(cluster_indexes)

    #print(len(cluster_list))

    real_index = np.where(y_train == x)
    total_hard=[]
    total_easy=[]
    total_real_easy=[]

    for idx,clusters in zip(set(labels),cluster_list):
        #print(clusters)
        #print(idx)
        clusters= np.where(labels == idx)
        #print(len(clusters[0]))
        #print(clusters[0])
        #print(len(clusters[0]))
        clusters_realindexes=[]
        for x in clusters[0]:
           clusters_realindexes.append(real_index[0][x])
        #print(len(clusters_realindexes))
        cluster_easyindex=[]
        cluster_hardindex=[]
        cluster_easyimages=[]
        cluster_hardimages=[]
        #cluster_real_images=[]
        ###entropy is
        ###accuarcy ==1

        for idx in clusters_realindexes:
            #if(train_exits[idx]==0):
            #if(train_entro_exits[idx]==0 and overall_acc[idx]==1):
            if(train_entroexits[idx]==0):
               cluster_easyindex.append(idx)
               cluster_easyimages.append(cifar_vgg_x_train[idx])
            else:
               cluster_hardindex.append(idx)
               cluster_hardimages.append(cifar_vgg_x_train[idx])
        #print("Length of easy images in cluster",len(cluster_easyimages))
        #print("Length of hard images in cluster",len(cluster_hardimages))


        if((len(cluster_easyimages)>=(len(cluster_hardimages)))):

             #total_hard.append(cluster_hardimages)
             ###begin changes
             d= len(cluster_easyimages)-len(cluster_hardimages)
             if(d==0):
                    var_easy=[]
                    total_hard.append(cluster_hardimages)
                    total_real_easy.append(cluster_easyimages)
                    #var_easy.append(cluster_easyimages[r])
                    #total_hard.append(cluster_hardimages)
                    #total_hard.append(cluster_hardimages)
                    #r = random.randint(0, len(cluster_hardimages))
                    r = calculate_entropy(model_vgg,cluster_easyimages)
                    #print("entropy index",r)
                    var_easy.append(cluster_easyimages[r])
                    #print("random number",r)
                    #print(var_easy.shape)
                    easy_peasy1 = var_easy*len(cluster_easyimages)
                    #print("Actual length")
                    #print(len(cluster_easyimages))
                    #print("Easy length")
                    #print(len(easy_peasy1))
                    #total_easy.append(cluster_easyimages[:len(cluster_hardimages)])
                    total_easy.append(easy_peasy1)
                    #total_easy.append(easy_peasy1)
                    #total_easy.append(easy_peasy1)
             else:
                    if(len(cluster_hardimages)!=0):

                        var_easy=[]
                        remaining=d//len(cluster_hardimages)
                        total_real_easy.append(cluster_easyimages)
                        total_hard.append(cluster_hardimages)
                        length1 = len(cluster_easyimages)
                        length2 = len(cluster_hardimages)
                        for itera in range(0,remaining+1):
                            left = length1 - length2
                            if(left<=length2):
                                total_hard.append(cluster_hardimages[:left])
                            else:
                                total_hard.append(cluster_hardimages[:length2])
                                length1=left
                        #total_hard.append(cluster_hardimages)
                        #total_hard.append(cluster_hardimages)
                        #r = random.randint(0, len(cluster_hardimages))
                        r = calculate_entropy(model_vgg,cluster_easyimages)
                        #print("entropy index",r)
                        var_easy.append(cluster_easyimages[r])
                        #print("random number",r)
                        #print(var_easy.shape)
                        easy_peasy1 = var_easy*len(cluster_easyimages)
                        #print("Actual length")
                        #print(len(cluster_easyimages))
                        #print("Easy length")
                        #print(len(easy_peasy1))
                        #total_easy.append(cluster_easyimages[:len(cluster_hardimages)])
                        total_easy.append(easy_peasy1)
                        #total_easy.append(easy_peasy1)
                        #total_easy.append(easy_peasy1)
                    else:
                        var_easy=[]
                        total_real_easy.append(cluster_easyimages)
                        total_hard.append(cluster_easyimages)
                        r = calculate_entropy(model_vgg,cluster_easyimages)
                        var_easy.append(cluster_easyimages[r])
                        easy_peasy1 = var_easy*len(cluster_easyimages)
                        total_easy.append(easy_peasy1)









             #total_real_easy.append(cluster_easyimages[:len(cluster_hardimages)])
        else:
             '''
             total_hard.append(cluster_hardimages[:len(cluster_easyimages)])
             #total_hard.append(cluster_hardimages[:len(cluster_easyimages)])
             #total_hard.append(cluster_hardimages[:len(cluster_easyimages)])
             #total_easy.append(cluster_easyimages)
             var_easy=[]
             r = calculate_entropy(model,cluster_easyimages)
             print("entropy index",r)
             var_easy.append(cluster_easyimages[r])
             easy_peasy1 = var_easy*len(cluster_easyimages)
             print("Actual length")
             print(len(cluster_easyimages))
             print("Easy length")
             print(len(easy_peasy1))
             total_easy.append(easy_peasy1)
             #total_easy.append(easy_peasy1)
             #total_easy.append(easy_peasy1)
             total_real_easy.append(cluster_easyimages)
             '''
             if(len(cluster_easyimages)!=0):
                 var_easy=[]
                 d= len(cluster_hardimages)-len(cluster_easyimages)
                 remaining=d//len(cluster_easyimages)
                 total_real_easy.append(cluster_easyimages)
                 total_hard.append(cluster_hardimages)
                 length1 = len(cluster_hardimages)
                 length2 = len(cluster_easyimages)
                 for itera in range(0,remaining+1):
                    left = length1 - length2
                    #print("Left item",left)

                    if(left<=length2):
                        total_real_easy.append(cluster_easyimages[:left])
                    else:
                        total_real_easy.append(cluster_easyimages[:length2])
                        length1=left
                    #print("loop",length1)

                 #r = calculate_entropy(model,cluster_easyimages)
                 #print("entropy index",r)
                 #var_easy.append(cluster_easyimages[r])
                 #total_hard.append(cluster_hardimages)
                 #total_hard.append(cluster_hardimages)
                 #r = random.randint(0, len(cluster_hardimages))
                 r = calculate_entropy(model_vgg,cluster_easyimages)
                 #print("entropy index",r)
                 var_easy.append(cluster_easyimages[r])
                 #print("random number",r)
                 #print(var_easy.shape)
                 easy_peasy1 = var_easy*len(cluster_hardimages)
                 #print("Actual length 2nd condition")
                 #print(len(cluster_hardimages))
                 #print("Easy length 2nd condition")
                 #print(len(easy_peasy1))
                 #total_easy.append(cluster_easyimages[:len(cluster_hardimages)])
                 total_easy.append(easy_peasy1)
                 #total_easy.append(easy_peasy1)
                 #total_easy.append(easy_peasy1)
             else:
                 var_easy=[]
                 total_real_easy.append(cluster_hardimages)
                 total_hard.append(cluster_hardimages)
                 r = calculate_entropy(model_vgg,cluster_hardimages)
                 var_easy.append(cluster_hardimages[r])
                 easy_peasy1 = var_easy*len(cluster_hardimages)
                 total_easy.append(easy_peasy1)






    #print("End Cluster")

    return total_easy, total_hard,total_real_easy


# In[34]:



def append_cluster(train_entroexits):

    total_easy_all=[]
    total_hard_all=[]
    val_easy_all=[]
    val_hard_all=[]
    total_reasy_all=[]
    val_reasy_all=[]

    for final_class in range(0,10):
        x_t,t_t, x = find_classes(final_class)
        #labels = Hdb_cluster(x_t,t_t, x)
        labels = kmeans_clustering(x_t,t_t, x)
        #print(x)
        totaL_ea,total_ha,total_reasy=clustering_bucketing(x_t,labels,x,train_entroexits)
        #print("total_easy_subclass",len(totaL_ea))
        #print("total_hard_subclass",len(total_ha))
        #print("total_realeasy_subclass",len(total_reasy))

        #print(len(totaL_ea),len(total_ha))
        for idx in range(0,len(totaL_ea)):
            total_easy_all+=totaL_ea[idx]
            val_easy_all+=totaL_ea[idx][:(len(totaL_ea[idx]))//3]

        for idx in range(0,len(total_ha)):
            total_hard_all+=(total_ha[idx])
            val_hard_all+=total_ha[idx][:(len(total_ha[idx]))//3]
        for idx in range(0,len(total_reasy)):
            total_reasy_all+=(total_reasy[idx])
            val_reasy_all+=total_reasy[idx][:(len(total_reasy[idx]))//3]

    return total_hard_all,total_easy_all,total_reasy_all,val_hard_all,val_easy_all,val_reasy_all


# In[16]:


def find_minimum(a,b,c):
    smallest=0
    print(a,b,c)

    if a <= b and a <= c :
        smallest = a
    elif b <= a and b <= c :
        smallest = b
    elif c <=a and c <= b :
        smallest = c
    return smallest


# In[15]:


def autoencoder_preprocess(train_entroexits):
    total_hard_all,total_easy_all,total_reasy_all,val_hard_all,val_easy_all,val_reasy_all=append_cluster(train_entroexits)
    num= find_minimum(len(val_easy_all),len(val_hard_all),len(val_reasy_all))
    num1 = find_minimum(len(total_easy_all),len(total_hard_all),len(total_reasy_all))
    #print(num)
    val_easy_all=val_easy_all[:num]
    val_hard_all=val_hard_all[:num]
    val_reasy_all=val_reasy_all[:num]

    total_easy_all=total_easy_all[:num1]
    total_hard_all=total_hard_all[:num1]
    total_reasy_all=total_reasy_all[:num1]

    total_hard= total_hard_all+total_reasy_all+total_hard_all+total_reasy_all
    total_easy= total_easy_all+total_easy_all+total_easy_all+total_easy_all
    val_hard= val_hard_all+val_reasy_all+val_hard_all+val_reasy_all
    val_easy= val_easy_all+val_easy_all+val_easy_all+val_easy_all
    total_hard=np.array(total_hard)
    total_easy=np.array(total_easy)
    val_hard=np.array(val_hard)
    val_easy=np.array(val_easy)
    total_hard=total_hard.reshape(-1,32,32,3)
    total_easy=total_easy.reshape(-1,32,32,3)
    val_hard=val_hard.reshape(-1,32,32,3)
    val_easy=val_easy.reshape(-1,32,32,3)

    print(total_hard.shape)
    print(total_easy.shape)
    print(val_hard.shape)
    print(val_easy.shape)
    return total_hard,total_easy,val_hard,val_easy




# In[33]:


def conv_block(x, filters, kernel_size, strides=2):
   x = Conv2D(filters=filters,
              kernel_size=kernel_size,
              strides=strides,
              padding='same')(x)
   x = BatchNormalization()(x)
   x = ReLU()(x)
   return x
def deconv_block(x, filters, kernel_size):
   x = Conv2DTranspose(filters=filters,
                       kernel_size=kernel_size,
                       strides=2,
                       padding='same')(x)
   x = BatchNormalization()(x)
   x = ReLU()(x)
   return x


# In[18]:


def denoising_autoencoder():
   dae_inputs = Input(shape=(32, 32, 3), name='dae_input')
   conv_block1 = conv_block(dae_inputs, 32, 3)
   conv_block2 = conv_block(conv_block1, 64, 3)
   conv_block3 = conv_block(conv_block2, 128, 3)
   conv_block4 = conv_block(conv_block3, 256, 3)
   conv_block5 = conv_block(conv_block4, 256, 3, 1)

   deconv_block1 = deconv_block(conv_block5, 256, 3)
   merge1 = Concatenate()([deconv_block1, conv_block3])
   deconv_block2 = deconv_block(merge1, 128, 3)
   merge2 = Concatenate()([deconv_block2, conv_block2])
   deconv_block3 = deconv_block(merge2, 64, 3)
   merge3 = Concatenate()([deconv_block3, conv_block1])
   deconv_block4 = deconv_block(merge3, 32, 3)

   final_deconv = Conv2DTranspose(filters=3,
                       kernel_size=3,
                       padding='same')(deconv_block4)
   ##change
   #final_deconv1= Dropout(0.25)(final_deconv)

   dae_outputs = Activation('sigmoid', name='dae_output')(final_deconv)

   return Model(dae_inputs, dae_outputs, name='dae')


# In[ ]:





# In[49]:


def dae_training():
    # alex net thresholds_array = np.linspace(0.001, 1, num=100)
    thresholds_array = np.linspace(0.000001, 0.0001, num=100)

    for thresholds in (thresholds_array):
        train_entroexits=calculate_thresholded_entropy_value(thresholds)
        total_hard,total_easy,val_hard,val_easy=autoencoder_preprocess(train_entroexits)
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        it_train = datagen.flow(total_hard, total_easy, batch_size=64)
        dae = denoising_autoencoder()
        dae.compile(loss='mse', optimizer='adam')

        #checkpoint = ModelCheckpoint('best_model.h5', verbose=1, save_best_only=True, save_weights_only=True)
        history = dae.fit_generator(it_train,
                epochs=10,
                shuffle=True,
                validation_data=(val_hard.reshape(len(val_hard), 32, 32,3), val_easy.reshape(len(val_easy),32, 32,3)),verbose=0)

        decoded_images = dae.predict(cifar_vgg_x_test.reshape(cifar_vgg_x_test.shape[0],32,32,3))
        autoencode_test = np.array(decoded_images)
        autoencode_test= autoencode_test.reshape(-1,32,32,3)
        batch_size = 32
        score = model_vgg.evaluate(autoencode_test, cifar_vgg_y_test, batch_size=batch_size, verbose=0)
        print("For threshold ",thresholds)
        #print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print()






# In[ ]:


dae_training()


# In[38]:


import tensorflow.keras as K


# In[56]:



def vgg_net():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[57]:


model = vgg_net()


# In[34]:


import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import BatchNormalization


# In[58]:


model.summary()


# In[59]:


y_test = K.utils.to_categorical(cifar_y_test, 10)
y_train = K.utils.to_categorical(cifar_y_train, 10)


# In[60]:


it_train = datagen.flow(cifar_x_train, y_train, batch_size=64)


# In[61]:


cifar_x_train.shape


# In[52]:


opt = SGD(lr=0.001, momentum=0.9)
model =model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[62]:


model.summary()


# In[ ]:



#create data generator
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
# prepare iterator
it_train = datagen.flow(cifar_x_train, y_train, batch_size=64)
# fit model

history = model.fit_generator(it_train,epochs=400, shuffle = True , validation_data=(cifar_x_test, y_test), verbose=1)
# evaluate model
_, acc = model.evaluate(cifar_x_test, y_test, verbose=0)
print('> %.3f' % (acc * 100.0))


# In[64]:


print('> %.3f' % (acc * 100.0))


# In[ ]:
