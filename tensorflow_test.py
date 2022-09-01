import sys
sys.path.append('/home/Kosta404/path_to_your_py_packages/')

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

train_images = np.load(r"solar_train_data.npz")['data'] #, np.load(r"solar_train_data.npz")['data']
train_labels = train_images[:,0]
train_feature_vector = np.delete(train_images, 0,1)

print(train_images)
print(train_labels)

test_images = np.load(r"solar_test_data.npz")['data']
test_labels = test_images[:,0]
test_feature_vector = np.delete(test_images, 0,1)

#print(train_images.files)
#print(test_images.files)
#print(test_images['data'])

#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images_ML, test_images_ML = train_images / 255.0, test_images / 255.0
train_images_ML = [[i[1:].reshape(256,256)  for i in train_images_ML]] 

test_images_ML = [[i[1:].reshape(256,256)  for i in test_images_ML]] 
class_names = ['filament', 'spot', 'flare', 'prominence', 'calm']

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(np.reshape(train_images[i][1:], (256, 256)))
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images_ML, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)