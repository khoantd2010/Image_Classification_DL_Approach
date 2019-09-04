# from google.colab import drive
# drive.mount('/content/drive')
# pip install keras-sequential-ascii
# ref: https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
import keras
import pandas  as pd
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras_sequential_ascii import sequential_model_to_ascii_printout
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
from time import time
# Timing calculation Class
class TimingCallback(keras.callbacks.Callback):
  def __init__(self):
    self.logs=[]
  def on_epoch_begin(self,epoch, logs={}):
    self.starttime=time()
  def on_epoch_end(self,epoch, logs={}):
    self.logs.append(time()-self.starttime)
# Early stop execution --> add into fit-generator if you'd like to apply
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
# TensorBoard
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=logdir)
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
y_true = np.asarray(y_test) # convert from list to array for y_test
#Convert and Pre-processing -> z-score
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

    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

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
 
# Start timing record
cb = TimingCallback()
# change verbose = 2 if you get error while running on GGColab
cnn6L = cnn.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                    steps_per_epoch=x_train.shape[0] // batch_size,epochs= number_epochs,\
                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(learning_rates),cb])
#save to disk
model_json = cnn.to_json()
with open('./models/cnn_c10_6l.json', 'w') as json_file:
    json_file.write(model_json)
cnn.save_weights('./models/model_c10_6l.h5')
import time
 
#testing
start_eva = time.time()
scores = cnn.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\n Test result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
end_eva = time.time()
# Virtualizing model structure
sequential_model_to_ascii_printout(cnn)

# Plotting
plt.figure(0)
plt.plot(cnn6L.history['acc'],'r')
plt.plot(cnn6L.history['val_acc'],'g')
plt.xticks(np.arange(0,201,20))
plt.rcParams['figure.figsize'] = (10,8)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])
# plt.savefig('./HINH/ACC/Accuracy_C10_6L.eps')
plt.figure(1)
plt.plot(cnn6L.history['loss'],'r')
plt.plot(cnn6L.history['val_loss'],'g')
plt.xticks(np.arange(0,201,20))
plt.rcParams['figure.figsize'] = (10,8)
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train','validation'])
# plt.savefig('./HINH/LOSS/Loss_C10_6L.eps')
plt.show()
# HeatMap Implementation
from sklearn.metrics import classification_report, confusion_matrix
start_inf = time.time()
Y_pred = cnn.predict(x_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)
end_inf = time.time()
for ix in range(10):
    print(ix, confusion_matrix(np.argmax(y_test, axis=1), y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print("Confusion Matrix:\n %s" % cm)

df_cm = pd.DataFrame(cm, range(10), range(10))
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.4)  # for label size
try:
    heatmap = sns.heatmap(df_cm, annot=True, annot_kws={"size": 12}, fmt='d')  # font size
except ValueError:
    raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# plt.savefig('./HINH/HEAT/HeatMap_C10_6L.eps')
plt.show()
# Precision - Recall - F1 Scores info
sr = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
sr.to_csv('./REPORT/Report_C10_6L.csv', index=True)
print ('\n[info] Classification Report\n')
print (classification_report(y_true, y_pred))
# Time info
print('\n Training time:\n')
print(cb.logs)
print('Total training time = %.3f' % sum(cb.logs))
print('Evaluation time = %.3f' % (end_eva - start_eva))
print('Prediction time = %.3f' % (end_inf - start_inf))
# Load and run TensorBoard in ipynb
# load_ext tensorboard
# tensorboard --logdir=.\logs\fit
