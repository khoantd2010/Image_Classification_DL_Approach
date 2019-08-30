# from google.colab import drive
# drive.mount('/content/drive')
# pip install keras-sequential-ascii
# ref: https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
import keras
import seaborn as sn
import pandas  as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras.utils import np_utils

# Delare Variables: 
num_classes = 10
weight_decay = 1e-4    
batch_size = 64
number_epochs = 200
# Learning rate elastics
def learning_rates(epoch):
    learning_rate = 0.001
    if epoch > 75:
        learning_rate = 0.0005
    if epoch > 100:
        learning_rate = 0.0003
    return learning_rate
# Loading CIDAR-10 dataset 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
 
#Convert and Pre-processing -> Normalization: z-score
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)
 
def CNN_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    #training
    opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    return model
  
cnn = CNN_model()    
cnn.summary()
 
#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(x_train) 

cnn6L = cnn.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                    steps_per_epoch=x_train.shape[0] // batch_size,epochs= number_epochs,\
                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(learning_rates)])

#testing
scores = cnn.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\n Test result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

# Virtualizing model structure
sequential_model_to_ascii_printout(cnn)

# Plotting
plt.figure(0)
plt.plot(cnn6L.history['acc'],'r')
plt.plot(cnn6L.history['val_acc'],'g')
plt.xticks(np.arange(0,201,20.0))
plt.rcParams['figure.figsize'] = (10,8)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])

plt.figure(1)
plt.plot(cnn6L.history['loss'],'r')
plt.plot(cnn6L.history['val_loss'],'g')
plt.xticks(np.arange(0,201,20.0))
plt.rcParams['figure.figsize'] = (10,8)
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train','validation'])

plt.show()
# Extra virtualization - heatmap
from sklearn.metrics import classification_report, confusion_matrix
Y_pred = cnn.predict(x_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)
 
for ix in range(10):
    print(ix, confusion_matrix(np.argmax(y_test,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print(cm)
  
df_cm = pd.DataFrame(cm, range(10),range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True) 
plt.show()
