

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("C:/Users/Lenovo/XAI/Chest_Xray"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.utils import shuffle

df = pd.read_csv('C:/Users/Lenovo/XAI/Chest_Xray/data/Data_Entry_2017.csv')

diseases = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']
#Number diseases
for disease in diseases :
    df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)



import os
labels = df[diseases].to_numpy()
all_image_paths = {os.path.basename(x): x for x in
                   glob(os.path.join('..', 'input','data','images*','images','*.png'))}
print('Images found:', len(all_image_paths))


df['Path'] = df['Image Index'].map(all_image_paths.get)
files_list = df['Path'].tolist()

# #test to perfect
# labelB = df['Emphysema'].tolist()

labelB = (df[diseases].sum(axis=1)>0).tolist()
labelB = np.array(labelB, dtype=int)

from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path, shape):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=shape)
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)/255
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, shape):
    list_of_tensors = [path_to_tensor(img_path, shape) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

train_labels = labelB[:89600][:, np.newaxis]
valid_labels = labelB[89600:100800][:, np.newaxis]
test_labels = labelB[100800:][:, np.newaxis]


img_shape = (64, 64)
train_tensors = paths_to_tensor(files_list[:89600], shape = img_shape)
valid_tensors = paths_to_tensor(files_list[89600:100800], shape = img_shape)
test_tensors = paths_to_tensor(files_list[100800:], shape = img_shape)

import time

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.layers import BatchNormalization
from keras import regularizers, applications, optimizers, initializers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0

# VGG16
# resnet50.ResNet50
# inception_v3.InceptionV3 299x299
# inception_resnet_v2.InceptionResNetV2 299x299

base_model = EfficientNetB0(weights='imagenet',
                                include_top=False,
                                input_shape=train_tensors.shape[1:])

add_model = Sequential()
add_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
add_model.add(Dropout(0.2))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dropout(0.2))
add_model.add(Dense(50, activation='relu'))
add_model.add(Dropout(0.2))
add_model.add(Dense(1, activation='sigmoid'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

model.summary()

from keras import backend as K

def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))

def precision_threshold(threshold = 0.5):
    def precision(y_true, y_pred):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(y_pred)
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall

def fbeta_score_threshold(beta = 1, threshold = 0.5):
    def fbeta_score(y_true, y_pred):
        threshold_value = threshold
        beta_value = beta
        p = precision_threshold(threshold_value)(y_true, y_pred)
        r = recall_threshold(threshold_value)(y_true, y_pred)
        bb = beta_value ** 2
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        return fbeta_score

import keras.backend as K
import tensorflow as tf

model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
              loss='binary_crossentropy',
              metrics=['accuracy',
                      precision_threshold(threshold = 0.5),
                       recall_threshold(threshold = 0.5),
                       fbeta_score_threshold(beta=0.5, threshold = 0.5)])

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import numpy as np

epochs = 20
batch_size = 32

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto')
log = CSVLogger('saved_models/log_pretrained_CNN.csv')
checkpointer = ModelCheckpoint(filepath='saved_models/pretrainedVGG.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)

start = time.time()
train_datagen = ImageDataGenerator(
                        featurewise_center=False,  # set input mean to 0 over the dataset
                        samplewise_center=False,  # set each sample mean to 0
                        featurewise_std_normalization=False,  # divide inputs by std of the dataset
                        samplewise_std_normalization=False,  # divide each input by its std
                        zca_whitening=False,  # apply ZCA whitening
                        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                        horizontal_flip=True,  # randomly flip images
                        vertical_flip=False
)



# Training with data augmentation. If shift_fraction=0., also no augmentation.
history = model.fit_generator(
    train_datagen.flow(train_tensors,train_labels, batch_size = batch_size),
    steps_per_epoch = len(train_tensors) // batch_size,
    validation_data = (valid_tensors, valid_labels),
    validation_steps = len(valid_tensors) // batch_size,
    epochs = epochs,
    callbacks=[checkpointer, log, earlystop], verbose=1
)
print("training time: %.2f minutes"%((time.time()-start)/60))

prediction = model.predict(test_tensors)

import matplotlib.pyplot as plt

plt.figure(1, figsize = (15,8))

plt.subplot(222)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('efficient-net model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])

plt.show()

import matplotlib.pyplot as plt

plt.figure(1, figsize = (15,8))

plt.subplot(222)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('efficient-net model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])

plt.show()

plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])

plt.title('efficient-net model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])

plt.title('efficient-net model precision')
plt.ylabel('f1_score')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

threshold = 0.5
beta = 0.5

pre = K.eval(precision_threshold(threshold = threshold)(K.variable(value=test_labels),
                                   K.variable(value=prediction)))
rec = K.eval(recall_threshold(threshold = threshold)(K.variable(value=test_labels),
                                   K.variable(value=prediction)))
#fsc = K.eval(fbeta_score_threshold(beta = beta, threshold = threshold)(K.variable(value=test_labels),
                                  # K.variable(value=prediction)))

print ("Precision: %f %%\nRecall: %f "% (pre, rec))

K.eval(binary_accuracy(K.variable(value=test_labels),
                       K.variable(value=prediction)))

labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
         'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
         'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
         'Pneumothorax', 'none']

matrices=[]

for i in range(0,15):
    fpr, tpr, thresholds = roc_curve(true_y[:, i], preds_sig[:, i])
    J = tpr - fpr
    ix = argmax(J)
    best_thresh = thresholds[ix]
    pred = np.where(preds_sig[:, i] > best_thresh, 1, 0)
    matrices.append(confusion_matrix(y_pred=pred, y_true=true_y[:, i]))

plt.figure(figsize=(24,20))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.5)

for i,j,k,m in zip(labels, matrices, range(0,15), range(1,16)):
    plt.subplot(5,5,m)
    sn.set(font_scale=1.2)
    plt.title('{}'.format(labels[k], 4, 4), y=1.0)
    df_cm = pd.DataFrame(matrices[k], index=None, columns=None)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 15}, fmt='g')
    sn.set(font_scale=0.8)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plt.show()

def gen_heatmap(input_image,target_class):
    target_layer = multi_disease_model.get_layer(find_target_layer())
    gradModel = Model(inputs = [multi_disease_model.inputs],
                      outputs = [target_layer.output, multi_disease_model.output])
    with tf.GradientTape() as tape:
        convOutputs, pred = gradModel(input_image)
        loss = pred[:,target_class]
    grads = tape.gradient(loss, convOutputs)
    # use automatic differentiation to compute the gradients

    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    # compute the guided gradients

    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
    heatmap = cv2.resize(cam.numpy(), IMG_SIZE)
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + epsilon  # normalize the heatmap such that all values lie in the range
    heatmap = numer / denom                            # [0, 1], scale the resulting values to the range [0, 255]
    heatmap = (heatmap * 255).astype("uint8")          # and then convert to an unsigned 8-bit integer
    return heatmap

fig, axs = plt.subplots(4, 4,figsize=(16,16))
i = 0
j = 0
for k in range(len(all_labels)):
    idx = img_idx[k]
    target = np.argmax(pred_test[idx,:]) # select target class as the one with highest probability
    heatmap = gen_heatmap(img_array[k:k+1],target)
    axs[i,j].imshow(img_array[k,:,:,0].astype("uint8"))
    axs[i,j].imshow(heatmap,alpha=0.5)
    axs[i,j].set_title(all_labels[target]+" : "+str(pred_test[idx,target]))
    j+=1
    if j==4:
        i += 1
        j = 0
fig.savefig("xray_samples.png")