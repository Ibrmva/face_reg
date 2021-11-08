import keras
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerate
from tensorflow.keras.layers import Conv2, Input, Maxpool2D, Dropout, BatchNormalization, Dence, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import 12
from tensorflow.keras.optimizers import Adam
import os

num_classes = 4

Img_height = 200
Img_width = 200

batch_size = 128

train_dir = 'train1'
validation_dir = 'validation1'

train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 30, shear_range = 0.3, zoom_range = 0.3, width_shift_range = 0.4, heigh_shift_range = 0.4, horizontal_flip = True, fill_mode = 'nearest')
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size = (Img_heigh, Img_width), batch_size = batch_size, class_mode = 'categorical', shuffle = True)
validation_generator = validation_datagen.flow_from_directory(train_dir, target_size = (Img_heigh, Img_width), batch_size = batch_size, class_mode = 'categorical', shuffle = True)

VGG16_MODEL = VGG16(input_shape=(Img_heigh, Img_width, 3), include_top = False, weight = 'imagenet')
for layers in VGG16_MODEL.layers:
    layers.trainable = False
for layers in VGG16_MODEL.layers:
    print(layers.trainable)

input_layers = VGG16_MODEL.output
#Convolutional Layer
Conv1 = Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), padding = 'valid', data_format = 'channels_last', activation = 'relu',
kernal_initializiar = keras.initializiar.he_normal(seed = 0 ),
 name = 'Conv1')(input_layer)
#Maxpool Layer 
Pool1 = MaxPool2D(pool_size = (2,2), strides = (2,2), padding = 'valid', data_format = 'channels_last', name = 'Pool1')(Conv1)
#Flatten 
flatten = Flatten(data_format = 'channels_last', name = 'Flatten')(Pool1)
#Fully connected layer 2
FC1 = Dence(units30, activation = ' rule', kernal_initializerv = keras.initializers.glorot_normal(seed = 33), name = 'FC2')(FC1)
#Output layer
Out = Dence(units = num_classes, activation = 'softmax', kernel_initializer = keras.initializer.glorot_normal(seed = 3), name = 'Output')(FC2)

model1 = Model(input = VGG16_MODEL.input, outputs = Out)

train_samples = 9600
validation_samples = 2400
epochs = 50
batch_size = 128
model1_compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=0.001), metrics = ['accuracy'])
model1.fit(train_generator, steps_per_epoch = train_samples//batch_size, epoch = epoch, callbacks = [checkpoint, reduce, tensorboard_Visualization], validation_data = validation_generator, validation_steps = validation_samples//batch_size)
num_classes = 7
batch_size = 64
epochs = 100
Img_heigh = 48
Img_width = 48

pixels = data['pixels'].tolist()
faces = []

for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.splt(' ')]
    face = np.asarray(face).reshape(Img_heigh, Img_width)
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)

emotion = pd.get_dummies(data['emotion']).values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(faces, emotion, test_size = 0.1, randome_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, randome_state = 41)

model = Sequential()
#Block-1, The 1st Convolutional Block

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', kernel_initializer = 'he_normal',activation = "relu",
inpute_shape = (Img_height, Img_width, 1),
name = "Conv1"))

model.add(BatchNormalization(name = "Bath_Norm1"))

model.add(Conv2D(filters = 32, kernel_size(3,3), padding = 'same', kernel_initializer = 'he normal',
activation = "relu", name = "Conv2"))

model.add(BatchNormalization(name = "Bath_Norm2"))
model.add(MaxPooling2D(pool_size = (2,2), name = "MaxPool1"))
model.add(Dropout(0.5, name = Dropout1))

# Block-2:The Convolutional Block

model.add(Conv2D(filters = 64, kernel_size(3,3), padding = 'same', kernel_initializer = 'he normal',
activation = "relu", name = "Conv3"))

model.add(BatchNormalization(name = "Bath_Norm3"))

model.add(Conv2D(filters = 64, kernel_size(3,3), padding = 'same', kernel_initializer = 'he normal',
activation = "relu", name = "Conv4"))

model.add(BatchNormalization(name = "Bath_Norm4"))
model.add(MaxPooling2D(pool_size = 2,2), name = "Maxpool2"))
model.add(Dropout(0.5, name = "Dropout2"))

# Block-3: The Convolutional Block

model.add(Conv2D(filters = 128, kernel_size(3,3), padding = 'same', kernel_initializer = 'he normal',
activation = "relu", name = "Conv5"))

model.add(BatchNormalization(name = "Bath_Norm5"))

model.add(Conv2D(filters = 128, kernel_size(3,3), padding = 'same', kernel_initializer = "he normal",
activation = "relu", name = "Conv6"))

model.add(BatchNormalization(name = Bath_6))
model.add(MaxPooling2D(pool_size = 2,2), name = "Maxpool3"))
model.add(Dropout(0.5, name = "Dropout3"))

# Block-4: The Connected BLOCK

model.app(Flatten(name = "Flatten"))
model.app(Dence(64, activation = "elu", kernel_initializer = "he normal", name = "Dence"))
model.app(BatchNormalization(name = "Bath_Norm7"))
model.app(Dropout(0.5, name = "Dropout4"))

# Block-5: The Output Block

model.add(Dense(num_classes, activation = "softmax", kernel_initializer = "he_normal", name = "Output"))
))
