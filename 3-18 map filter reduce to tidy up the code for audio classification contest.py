#!/usr/bin/env python
# coding: utf-8

# In[257]:


# Part I ---
# Data Preprocessing
# generate derived images or extracted features
# plus file manipulation to set up folder and subfolders
#
#

import wave
import os
import glob
import shutil
import pandas as pd
import numpy as np
from pandas import read_csv
import librosa
import librosa.display
import sklearn.preprocessing
from scipy import signal
import matplotlib.pyplot as plt
from random import seed
from random import random, randint

def image_derived(data, sr): 
    # print ('data Shape', data.shape, 'sampling rate', sr)  
    spec = librosa.feature.melspectrogram(y=data, sr=sr, fmax=8192)   # show in amplitude
    # print ('spec shape', spec.shape)
    librosa.display.specshow(spec,y_axis='mel', x_axis='s', sr=sr)
    db_spec = librosa.power_to_db(spec, ref=np.max)
    # print ('db_spec shape', db_spec.shape)
    # librosa.display.specshow(db_spec,y_axis='mel', x_axis='s', sr=sr)    # show in db
    # plt.show()
        
    b, a = signal.butter(3, 0.05)
    y = signal.filtfilt(b, a, data)
    y1 = np.asfortranarray(y)    
    spec = librosa.feature.melspectrogram(y=y1, fmax=1536)   # show in amplitude
    # spec = librosa.feature.melspectrogram(y=y, sr=sr)   # show in amplitude
        
    # print ('spec shape', spec.shape)
    librosa.display.specshow(spec,y_axis='mel', x_axis='s', sr=sr)
    db_spec = librosa.power_to_db(spec, ref=np.max)
    # print ('db_spec shape', db_spec.shape)
    # librosa.display.specshow(db_spec,y_axis='mel', x_axis='s', sr=sr)    # show in db
    # plt.show()
    
    return(db_spec)

def feature_extracted(data, sr): 
    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = 512
    y_harmonic, y_percussive = librosa.effects.hpss(data) # Separate harmonics and percussives into two waveforms
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr) # Beat track on the percussive signal
    
    mfcc = librosa.feature.mfcc(y=data, sr=sr, hop_length=hop_length, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)  # And the first-order differences (delta features)       
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames) # sync with beats
    
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr) # Compute chroma features from the harmonic signal
    # Aggregate chroma features between beat events # We'll use the median value of each feature between beat frames
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median) # use the median value of each feature between beat frame
    
    # Finally, stack all beat-synchronous features together   
    beat_features = np.vstack([beat_mfcc_delta, beat_chroma]).T
    mean_beat_features = np.mean(beat_features, axis=0) 
    std_beat_features = np.std(beat_features, axis=0) 
    features = np.hstack((mean_beat_features, std_beat_features))  
    # print('feature_2', pd.DataFrame(features_2).shape)
    return (features)

def image_gen(x):
    # print (x)
    filename=x[0] # wave file name
    label=x[1] # file label
    
    image_png= [] # aPseudoImage      # retunr thre orginal .wav file
    if not os.path.exists(T_path+'/'+filename):
        print ('ERROR - Skip file %s as it can not be found' %(filename))
    elif label < 0 or label > 7:
        print ('ERROR - Skip file %s as the label %d is not in range' %(filename, label) )
    else:
        print ('Handle file:%s with label %d' %(filename, int(label)))
        # create a derived image
        data, sr = librosa.load(T_path+'/'+filename) 
        data_reduced = data[:sr*11]
        image_data=image_derived(data_reduced, sr)
        # dispatch the image to the subdirectories such as trainDir/label3 or validDir/label5
        fig = plt.figure(figsize=[1.0, 1.0])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        librosa.display.specshow(image_data)    
        if randint(1,4) == 2: # pick 25% of training samples for validation
            path = TV_path+'/label'+str(label)
        else:
            path = TT_path+'/label'+str(label)
        if not os.path.exists(path):
            os.makedirs(path)
        filename  = filename.replace('.wav','.png')
        plt.savefig(path+'/'+filename, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close('all')
        print('destination:', path+'/'+filename)
    return (filename)
    
def feature_gen(x):
    filename=x[0] # wave file name
    label=x[1] # file label  
    feature_vector=[]
    if not os.path.exists(T_path+'/'+filename):
        print ('Skip file %s as it can not be found' %(filename))
    elif label < 0 or label > 7:
        print ('Skip file %s as the label %d is not in range' %(filename, label) )
    else:
        print ('Handle file:%s with label %d' %(filename, int(label)))
        # extract features
        data, sr = librosa.load(T_path+'/'+filename) 
        feature_vector=feature_extracted(data, sr)
        recordX = np.hstack((filename, label, feature_vector))
        # print(recordX)
    return (recordX)
    
home_dir='/Users/YCLee/YC Documents/RapidMiner/aidea voice classification'
L_file=home_dir+'/train.csv'  # the lable file
labels=read_csv(L_file, sep=',', header=0).values
# dic={row[0]:row[1] for _, row in DataSet.iterrows()} # convert dataframe to a dictionary
T_path=home_dir+'/train_set' # where each training record (an image or wave file) is stored
# scaler = sklearn.preprocessing.StandardScaler()  # use the same scaler for normalization later
TT_path=home_dir+'/trainDir' #keep subdolder like trainDir/label3 for training
TV_path=home_dir+'/validDir' #keep subdolder like validDir/label5 for va;idation
seqNo=0 # for global count of files# scaler = sklearn.preprocessing.StandardScaler()  # use the same scaler for normalization later

# feature extraction
feature_array=list(map(feature_gen, labels)) # the array holding all extracted feature vectors
print('feature_array', pd.DataFrame(feature_array).shape)
pd.DataFrame(feature_array).to_csv(home_dir+'/train_feature_data_x.csv', sep=',', header=False, index=False)

# image drivation
# seed(1)
# image_array=list(map(image_gen, labels)) # the array holding all extracted feature vectors
# print('image_array', pd.DataFrame(image_array).shape)


# In[258]:


FA = feature_array
print('FA', pd.DataFrame(FA).shape)


# In[ ]:





# In[259]:


# Part II ---
# Training and Validation
# ----------------
# 2D CNN based approach - build the model and run
# either compile simple model from scratch or compile model with pretrained model
#
from keras import layers 
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications import VGG16
from sklearn.metrics import classification_report, confusion_matrix

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

import matplotlib.pyplot as plt

def simple_model():
    model = models.Sequential()
    # (72, 72) for now, adjust later
    model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(72, 72, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(8, activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return (model)

def pretrained_model(): 
    # simple example of transfer learning  
    # conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(72, 72, 3))
    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(102, 102, 3))
    model = models.Sequential() 
    model.add(conv_base)
    print('Number of trainable weights BEFORE freezing the conv base:', len(model.trainable_weights))
    conv_base.trainable = False 
    print('Number of trainable weights -AFTER freezing the conv base:', len(model.trainable_weights))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(8, activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return (model)
    
def fit_model(m_def, train_dir, validation_dir):
    model = m_def
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        )
    test_datagen = ImageDataGenerator(rescale=1./255)
        
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(72, 72), # needs to be consistent with inpit_shape
        batch_size=32,
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(72,72),
        batch_size=32,
        class_mode='categorical')
    
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=10)
  
    test_loss, test_acc = model.evaluate_generator(validation_generator, steps=6) 
    print('test acc:', test_acc)
    
    # Need to revisit below
    # Y_pred = model.predict_generator(validation_generator, 7) 
    ## validation_steps = 1+ num_of_test_samples // batch_size
    # y_pred = np.argmax(Y_pred, axis=1)
    # print('Confusion Matrix')
    # print(confusion_matrix(validation_generator.classes, y_pred))
    # print('Classification Report')
    # target_names = ['L_0', 'L_1', 'L_2', 'L-3', 'L_4', 'L_5', 'L_6', 'L_7']
    # print(classification_report(validation_generator.classes, y_pred, target_names=target_names))   
    return (history)

def plot_learning_curve(hist):
    # plot the learning curve
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('SPEC - Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('SPEC - Training and validation loss')
    plt.legend()
    plt.show()
    
def run():
    m=simple_model()
    # m=pretrained_model()
    TT_dir = '/Users/YCLee/YC Documents/RapidMiner/aidea voice classification/trainDir'
    TV_dir = TT_dir.replace('train','valid')
    h=fit_model(m, TT_dir, TV_dir)
    return (h)

h=run()    
plot_learning_curve(h)


# In[354]:


# Part II ---
# Training and Validation
# ----------------
# Traditional feature-based approach - build the model and run
# use neuron network first
#

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost.sklearn import XGBClassifier as xgbc
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

# --------- Algorithm DL -----------
def DL_model():
    model = Sequential()
    model.add(Dense(256, input_dim=104, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    # model.add(Dense(25, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    return model
    
def byDL(X_train, y_train, k_fold, split):
    X = X_train
    print('y_train shape', pd.DataFrame(y_train).shape)
    y = y_train
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10) # patient early stopping
    if  k_fold > 0:
        estimator = KerasClassifier(build_fn=DL_model, epochs=50, batch_size=20, verbose=0)
        kfold = KFold(n_splits=k_fold, shuffle=True)
        results = cross_val_score(estimator, X, y, cv=kfold)
        print("Cross-validated by kfold-%-2d: %.2f%% (%.2f%%)" % (k_fold, results.mean()*100, results.std()*100))
    clf = DL_model()
    history = clf.fit(X, y, 
                      validation_split=split, 
                      epochs=50, batch_size=20, verbose=0, callbacks=[es])   
    return (clf, history) 

# --------- Algorithm XGB from SK Grid  -----------
def XGB_model():
    xgb_model = xgbc()
    parameters = {
            'nthread':[4], # when use hyperthread, xgboost may become slower
            'objective':['reg:squarederror'],
            'learning_rate': [.03, 0.07], # [.03, .07] #so called `eta` value
            'max_depth': [5, 6], # [5, 6]
            'min_child_weight': [4],
            'silent': [1],
            'subsample': [0.7],
            'colsample_bytree': [0.7],
            'n_estimators': [500]}
    predictor = GridSearchCV(
            xgb_model,
            parameters,
            cv = 2,
            n_jobs = 5,
            verbose=True)
    return(predictor)
    
def byXGB(X_train, y_train, k_fold, split):
    X = X_train
    print('y_train shape', pd.DataFrame(y_train).shape)
    y = y_train.reshape(-1) # need to flatten y_train
    print('reshape(y_train) shape', pd.DataFrame(y).shape)
    if  k_fold > 0:
        # estimator = KerasClassifier(build_fn=XGB_model)
        # kfold = KFold(n_splits=k_fold, shuffle=True)
        xgb_model = XGB_model()
        kfold = KFold(n_splits=k_fold, random_state=7)
        results = cross_val_score(xgb_model, X, y, cv=kfold)
        print("Cross-validated by kfold-%-2d: %.2f%% (%.2f%%)" % (k_fold, results.mean()*100, results.std()*100))

    predictor=XGB_model()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=9)
    # predictor.fit(X, y)
    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    print('Model to be used has accuracy=', acc)
    # validation_split=split, epochs=50, batch_size=20, verbose=0
        # ) # no history here to track the fitting process
    print(predictor.best_params_)
    return (predictor, None)    

##########
###
## Generic code below
#
def train_Xvalid_model(byAlgo, X_train, y_train, k_fold, split):
    # if k_fold == 0, bypass X-validation and simply generate model with history
    model, history=byAlgo(X_train, y_train, k_fold, split)
    return(model, history)

def plot_learning_curve(hist):
    print(history.history.keys())
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(acc) + 1)
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'r--', label='Validation loss')
    plt.plot(epochs, loss, 'k', label='Training loss')
    plt.plot(epochs, val_loss, 'r^', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.figure()
    plt.show()
    # plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.plot(epochs, acc, 'k', label='Training acc')
    plt.plot(epochs, val_acc, 'r--', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()
    plt.figure()
    plt.show()
    
def load_data():
    # 193 mean features: chroma(12), mfcc(20), delta(20), chroma_s(12), mfcc_s(20), delta_s(20)
    All_features = [ 
        'cmx00',  'cmx01',  'cmx02',  'cmx03',  'cmx04',  'cmx05',  'cmx06',  'cmx07',  'cmx08',  'cmx09',
        'cmx10',  'cmx11',
        'fmx00',  'fmx01',  'fmx02',  'fmx03',  'fmx04',  'fmx05',  'fmx06',  'fmx07',  'fmx08',  'fmx09',
        'fmx10',  'fmx11',  'fmx12',  'fmx13',  'fmx14',  'fmx15',  'fmx16',  'fmx17',  'fmx18',  'fmx19',  
        'dmx00',  'dmx01',  'dmx02',  'dmx03',  'dmx04',  'dmx05',  'dmx06',  'dmx07',  'dmx08',  'dmx09',
        'dmx10',  'dmx11',  'dmx12',  'dmx13',  'dmx14',  'dmx15',  'dmx16',  'dmx17',  'dmx18',  'dmx19',
        'csx00',  'csx01',  'csx02',  'csx03',  'csx04',  'csx05',  'csx06',  'csx07',  'csx08',  'csx09',
        'csx10',  'csx11',
        'fsx20',  'fsx21',  'fsx22',  'fsx23',  'fsx24',  'fsx25',  'fsx26',  'fsx27',  'fsx28',  'fsx29',
        'fsx30',  'fsx31',  'fsx32',  'fsx33',  'fsx34',  'fsx35',  'fsx36',  'fsx37',  'fsx38',  'fsx39',  
        'dsx20',  'dsx21',  'dsx22',  'dsx23',  'dsx24',  'dsx25',  'dsx26',  'dsx27',  'dsx28',  'dsx29',
        'dsx30',  'dsx31',  'dsx32',  'dsx33',  'dsx34',  'dsx35',  'dsx36',  'dsx37',  'dsx38',  'dsx39',
        ]
    # load training dataset
    dataframe = pd.read_csv(home_dir+'/train_feature_data_x.csv', header=None).sort_values(by=[0])
    dataset = dataframe.values
    Xarray = dataset[:,2:].astype(float)
    # print('Train: dataset shape', Xarray.shape)
    Y = dataset[:,1].astype(str)
    ID_list = dataset[:,0].astype(str)
    X = pd.DataFrame(data=Xarray, columns=All_features)
    print('Train: dataset shape', X.shape)
    features_to_remove = [
            # empty for now
        ]
    X_reduced = X.drop(features_to_remove, axis=1)
    y = pd.DataFrame(data=Y, columns=['label'])
    # print(y.head())
    ids = pd.DataFrame(data=ID_list, columns=['wav'])
    print(y.label.value_counts())
    return(X_reduced, y, ids)

# InteractiveShell.ast_node_interactivity = "all"
X1, y1, id_list = load_data()
# pd.DataFrame(X).to_csv(home_dir+'/RM/train_X.csv', sep=',', header=True, index=False)
# pd.DataFrame(y).to_csv(home_dir+'/RM/train_y.csv', sep=',', header=True, index=False)
# pd.DataFrame(id_list).to_csv(home_dir+'/RM/train_id.csv', sep=',', header=True, index=False)
        
# main set up
print("Original shape", X1.shape, y1.shape)
# if use predictionDL
X_train = X1.values
y_train = y1.values

# algo = byDL
algo = byXGB
if algo == byDL:
    # need to enumerate y_train, the label, and normalize the X_train, training records
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    y = np_utils.to_categorical(encoded_Y)
    print('cate_y', pd.DataFrame(cate_y).shape)
    # if use DL for multi-class prediction
    scaler = MinMaxScaler() 
    scaler = StandardScaler()
    X = scaler.fit_transform(X_train) 
    # REMEBER to scale the test data as well - scaled_test = scaler.transform(X_test) 
else: # others such as byXGB
    X=X_train
    y=y_train

# valiadtion_split_ratio, k_fold = 0.2, 2
# bstV, history = train_Xvalid_model(algo, X, y, k_fold, valiadtion_split_ratio) # 3-fold cross-validation
# print('Learning curve by valiadtion split=', valiadtion_split_ratio)
# plot_learning_curve(history)

valiadtion_split_ratio, k_fold = 0.05, 10
bstV, history = train_Xvalid_model(algo, X, y, k_fold, valiadtion_split_ratio)
if algo == byDL:
    print('Learning curve by valiadtion split=', valiadtion_split_ratio)
    plot_learning_curve(history)


# Below are all the examples for illustratio purposes

# In[150]:


# example of map(func, list(s))
#
#

import functools, operator

a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
b = ['a', 'b', 'c', 'd', 'e', 'f', 'gxxx', 'hyyy', 'i', 'j']
c = ['A', 'BC', 'DEF']
d = [[1, 'a'], [2, 'b'], [3, 'c']]

squared = []
for x in a: squared.append(x ** 2)    
print('loop', squared)
print('lambda-map', list(map((lambda x: x **2), a)))             
def sq(x): return x ** 2
print('def-map', list(map(sq, a)))     
def cub(x): return x ** 3
print ('consecutive func', list(zip(map(sq, a), map(cub, a))))
def form_id(int1, str1, str2): return (str1+'_'+str(int1)+'--'+str2)
print('id strings', list(map(form_id, a, b, c)) )

def conca(x): 
    print (x)
    return str(x[0])+'_'+x[1]
print('combine components', list(map(conca, d)) )


# In[132]:


# example of filter(func, list
#
#

import functools, operator

a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
b = ['a', 'b', 'c', 'd', 'e', 'f', 'gxxx', 'hyyy', 'i', 'j']
c = ['A', 'BC', 'DEF']

odd_squared = []
for x in a: 
    if x%2 != 0:
        odd_squared.append(x**2)
print('loop', odd_squared)
def odd(x): return ( x%2 != 0)
print('def-filter', list(filter(odd, a)))   
print('lambda-filter', list(filter((lambda x: x%2 != 0), a)))  

def form_id(int1, str1, str2): return (str1+'_'+str(int1)+'-'+str2)
print('id strings', list(map(form_id, a, b, c)))


# In[133]:


# example of reduce(func, list(s))
#
#

import functools, operator
from functools import reduce

a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
b = ['a', 'b', 'c', 'd', 'e', 'f', 'gxxx', 'hyyy', 'i', 'j']
c = ['A', 'BC', 'DEF']

add_all = 0
for x in a: 
    add_all = add_all+x
print('loop', add_all)
def add_(x, y): return (x+y)
 
print('lambda-reduce', reduce((lambda x, y: x + y**2), a))
print('def-reduce', reduce(add_, a))

def concat(str1, str2): return (str1+'_'+str2)
print('id strings', reduce(concat, b))


# In[237]:


# ---------------------
# plot the learning curve
#

import matplotlib.pyplot as plt

def plot_learning_curve(hist):
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('SPEC - Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('SPEC - Training and validation loss')
    plt.legend()
    plt.show()
    
plot_learning_curve(h)


# In[ ]:




