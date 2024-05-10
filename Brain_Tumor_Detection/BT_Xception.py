# -*- coding: utf-8 -*-
"""Xception.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UeE4Eftlz0eTCRLrFKMxjNa6ixQEUgvR
"""

from google.colab import drive
drive.mount('/content/gdrive')

import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize

from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras

datadir = '/content/gdrive/MyDrive/Tumor'
categories = ['no', 'yes']
x = []
y = []

for category in categories:
    path = os.path.join(datadir, category)
    for image in os.listdir(path):
        images = cv2.imread(os.path.join(path, image))
        color = cv2.imread(path, cv2.COLOR_BGR2RGB)
        images = resize(images, (224, 224, 3), mode = 'constant', preserve_range = True)
        x.append(images)
        y.append(categories.index(category))

no = 0
yes = 0

for i in range(len(x)):
    if y[i] == 0:
        no= no + 1
    elif y[i] == 1:
        yes = yes + 1

print("Total Images: ", len(x))
print("Total Images No Tumor: ", no)
print("Total Images Tumor: ", yes)

target_dict={k: v for v, k in enumerate(np.unique(y))}
target_val=  [target_dict[y[i]]
              for i in range(len(y))]

X2, X_test, y2, y_test = train_test_split(x, y, test_size=0.1,random_state=42,shuffle=True,stratify=y) # Test
X_train, X_val, y_train, y_val = train_test_split(X2, y2, test_size=0.2,random_state=42,shuffle=True,stratify=y2) #Training(80%) and Validation(20%)

print('Train size:', len(X_train))
print('Validation size:', len(X_val))
print('Test size:', len(X_test))

classes = np.unique (y_train)
classes_num = len(classes)

print ('Outputs: ', classes_num)
print ('Classes: ', classes)

!pip install tf-explain # to use pre trained

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
from tf_explain.core.activations import ExtractActivations
from tensorflow.keras.applications.xception import decode_predictions
# %matplotlib inline

base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    include_top=False)
base_model.trainable = False #Setting trainable to False empties the list of trainable weights of the layer or model.

base_model.summary()

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ]
)

inputs = keras.Input(shape=(299,299,3))
x = data_augmentation(inputs)
x = keras.applications.xception.preprocess_input(x)
x = base_model(x)
x = layers.Flatten()(x)
x = layers.Dense(128)(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(loss="binary_crossentropy",
            optimizer="adam",
              metrics=["accuracy"])

x_test = np.array(X_test)
x_train = np.array(X_train)
y_test = np.array(y_test)
y_train = np.array(y_train)
x_val = np.array(X_val)
y_val = np.array(y_val)

print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)
print (x_val.shape, y_val.shape)

history = model.fit(x_train, y_train, epochs=60, batch_size = 64, validation_data=(x_val, y_val))

model.save('/content/gdrive/MyDrive/Br35BT/modelBr35HDataAug2.h5')

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.title('Loss')
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend();
plt.subplot(1,2,2)
plt.title('Accuracy')
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend();

print ('Loss and Accuracy:')
model.evaluate(x_test, y_test)

from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

cmat=confusion_matrix(y_test,y_pred,labels=[0,1])
cm_df = pd.DataFrame(cmat)

cmat_df = pd.DataFrame(cmat,
                     index = ['No Tumor','Tumor' ],
                     columns = ['No Tumor','Tumor' ])

plt.figure(figsize=(8,6))
sns.heatmap(cmat_df, annot=True,fmt="d",cmap=plt.cm.Blues )
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

tp = cmat[1][1] # True positive
fp = cmat[0][1] # False positive
tn = cmat[0][0] # True negative
fn = cmat[1][0] # False negative

accuracy = ((tp+tn)*100)/np.sum(cmat)
precision = (tp*100)/(tp+fp)
sensibility = (tp*100/(tp+fn))
specificity = (tn*100)/(fp+tn)

print('Metrics')
print('Accuracy:..........>',accuracy,"%")
print('Precision:..........>',precision,"%")
print('Recall:.....>',sensibility,"%")
print('Specificity:....>',specificity,"%")

plt.show()

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,5))
plt.plot(fpr, tpr, color='darkorange',  label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC (Receiver Operating Characteristic)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

import numpy as np
from PIL import Image


# Load and preprocess the image
img = Image.open("/content/gdrive/MyDrive/Tumor/yes/y1200.jpg")
img = img.resize((299, 299))
img_arr = np.array(img)
img_arr = np.expand_dims(img_arr, axis=0)
img_arr = img_arr.astype("float32") / 255

# Make predictions
preds = model.predict(tf.constant(img_arr))
# Plot the image and the predicted class
plt.imshow(img)
plt.axis('off')
plt.title(f"Predicted class: {preds}")
plt.show()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
import numpy as np
from PIL import Image

# Load and preprocess the image
img = Image.open("/content/gdrive/MyDrive/Tumor/yes/y12.jpg")
img = img.resize((299, 299))
img_arr = np.array(img)
img_arr = np.expand_dims(img_arr, axis=0)
img_arr = img_arr.astype("float32") / 255

# Make predictions
preds = model.predict(tf.constant(img_arr))
# Plot the image and the predicted class
plt.imshow(img)
plt.axis('off')
plt.title(f"Predicted class: {preds}")
plt.show()