# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 20:52:26 2023

@author: Nidhi
"""

import pandas as pd
import numpy as np
import os
import glob

import PIL
from PIL import Image
import imageio

import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import seaborn as sn
import numpy as np
import pathlib

train = pathlib.Path("C:\\Users\\Nidhi\\Desktop\\brain\\train")
test = pathlib.Path("C:\\Users\\Nidhi\\Desktop\\brain\\test")

image_count_train = len(list(train_dir.glob('*/*.jpg')))
image_count_test = len(list(test_dir.glob('*/*.jpg')))
print("Train Image Count: {} \n Test Image Count: {}".format(image_count_train,image_count_test))

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train,
    validation_split=None,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test,
  validation_split=None,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 2

model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(255, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

hist = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8
)

model.summary()

tf.keras.utils.plot_model(model,
                          show_shapes=True,
                          expand_nested=True)

def plot_metrics(history):
  metrics = ['loss', 'accuracy']
  for n, metric in enumerate(metrics):
    try:
      name = metric.replace("_"," ").capitalize()
      plt.plot(history.epoch, history.history[metric], label='Train')
      plt.plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Val')
      plt.xlabel('Epoch')
      plt.ylabel(name)
      if metric == 'loss':
        plt.ylim([0, plt.ylim()[1]])
      elif metric == 'auc':
        plt.ylim([0.8,1])
      else:
        plt.ylim([0,1])
      plt.legend()
      plt.show()  
    except:
      pass
  
plot_metrics(hist)


def list_files(dir,full_dir):
    r = []
    r1 = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            rr = os.path.join(root, name)
            r.append(rr)
    dd = {'local_path':r}
    df = pd.DataFrame(dd)
    return df


def proccess(img1):
  img = tf.keras.preprocessing.image.load_img(
      img1, target_size=(img_height, img_width)
  )
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  pred = class_names[np.argmax(score)]
  score1 = 100 * np.max(score)

  return pred, score1


def new_col(col):
    if col['Pred'] == 'yes' and col['Actual'] == 'yes':
        return 1
    elif col['Pred'] == 'no' and col['Actual'] == 'no':
      return 1
    else:
      return 0
  
    
def proccess1(df):
  aa = []
  bb = []
  cc = []

  for a,b in df.iterrows():
    img = b['local_path']
    pred, value = proccess(img)
    pat = b['local_path']
    
    val = pat.split()
    
    aa.append(pred)
    bb.append(value)
    cc.append(val)
  vals = {"Pred":aa,"Accurarcy":bb,'Actual':cc}
  df_test1 = pd.DataFrame(vals)
  df_test1 = pd.concat([df,df_test1], axis=1)

  df_test1['Check'] = df_test1.apply(lambda col: new_col (col),axis=1)

  return df_test1


fullpath = 'C:\\Users\\Nidhi\\Desktop\\brain\\test'
path = "test"

df_test = list_files(fullpath, path)
df_test1 = proccess1(df_test)

def new_col(col):
    if col['Pred'] == 'yes':
        return 1
    else:
      return 0
  
    
def new_col2(col):
    if col['Actual'] == 'yes':
        return 1
    else:
      return 0
  
    
df_test1['Pred1'] = df_test1.apply(lambda col: new_col (col),axis=1)
df_test1['Actual1'] = df_test1.apply(lambda col: new_col2 (col),axis=1)
df_test1.head()
df_test1.head()

form = df_test1.Check.value_counts()[1] / df_test1.Check.count()
print('Accuracy is : {}'.format(form))

cm = tf.math.confusion_matrix(labels=df_test1['Actual1'].to_numpy(), predictions=df_test1['Pred1'].to_numpy()).numpy()
ls = ['Non-Tumorous', 'Tumorous'] # your y labels()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ls)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(xticks_rotation=50, ax = ax)
plt.show()

acc = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1]+ cm[1][1] + cm[1][0])
TPR =  (cm[1][1]) / (cm[1][1] + cm[1][0])
FPR = (cm[0][1]) / (cm[0][1] + cm[0][0])
print("ACC: {}\nTPR: {}\n FPR: {}".format(acc,TPR,FPR))

df_test1.head()

df_test1[df_test1["Check"] == 0][['Actual',"Pred"]].value_counts()

df_test1

df_test2= df_test1.head(10)
df_test3= df_test1.tail(10)
df_test4 = pd.concat([df_test2,df_test3])
df_test4

for a,b in df_test4.iterrows():
    img_path = (b['local_path'])
    im = imageio.imread(img_path)

    print("Actual: {} \nPrediction: {}".format(b['Actual'], b['Pred']))
    plt.imshow(im)
    plt.show()
    print('==============================================')