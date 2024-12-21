import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to the project directory
path = "C://rooftop-detection//"

# Load labels
labels = pd.read_csv(path + "labels.csv", delimiter=",", header=None)

# Initialize the list to store images
L = []
train = np.array([])

# Get the list of all images
images = glob.glob(os.path.join(path, "images", "*.*"))

# Resize each image and store them in the list
for i in range(len(images)):
    image_path = os.path.join(path, "images", labels.iloc[i, 0] + ".jpg")
    im = Image.open(image_path)
    im_rz = im.resize((64, 64), Image.ANTIALIAS)  # Resize to 64x64
    L.append(np.array(im_rz))

# Convert the list of images into a numpy array
data = np.array(L)

# Convert labels into categorical format
y = pd.get_dummies(labels.iloc[:, 1]).to_numpy()

# Split the dataset into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(data, y, train_size=0.8, random_state=42)

# Define the batch size, number of classes, and number of epochs
batch_size = 128
nb_classes = y.shape[1]  # Infer the number of classes from labels
epochs = 150

# Preprocess the data (normalize pixel values)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Build the Convolutional Neural Network (CNN)
model = Sequential()

# First convolutional block
model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(110, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

# Second convolutional block
model.add(Conv2D(84, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(84, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

# Third convolutional block
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

# Fourth convolutional block
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

# Fully connected layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))

# Compile the model
opt = optimizers.RMSprop(learning_rate=0.001, decay=1e-7)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Define a checkpoint to save the best model during training
filepath = os.path.join(path, "weights.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Data augmentation
data_generator = ImageDataGenerator(
    rotation_range=7,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    vertical_flip=True
)
data_generator.fit(x_train)

# Train the model with data augmentation
model.fit(
    data_generator.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=callbacks_list
)
