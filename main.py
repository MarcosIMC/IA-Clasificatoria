import tensorflow as tf
import matplotlib.pyplot as plt
#Comentar la línea inferior si se ejecuta en local (PyCharm)
from google.colab import drive
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Rescaling, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import LeakyReLU

#Comentar la línea inferior si se ejecuta en local (PyCharm)
drive.mount('/content/drive')

image_size = (150,150)
batch_size = 32

#DataAugmentaiton
train_da = ImageDataGenerator(
    rotation_range = 20,
    #width_shift_range = 0.2,
    #height_shift_range = 0.2,
    #shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest')

train_ds = train_da.flow_from_directory('/content/drive/My Drive/Colab Notebooks/datasets/seg_train',
                                                               #validation_split = 0.2,
                                                               #subset = 'training',
                                                               seed = 1337,
                                                               target_size = image_size,
                                                               batch_size = batch_size,
                                                               class_mode = 'categorical')

val_ds = tf.keras.preprocessing.image_dataset_from_directory('/content/drive/My Drive/Colab Notebooks/datasets/seg_train',
                                                               #validation_split = 0.2,
                                                               #subset = 'training',
                                                               seed = 1337,
                                                               image_size = image_size,
                                                               batch_size = batch_size,
                                                               label_mode = 'categorical')

#train_ds = train_ds.prefetch(buffer_size = 32)
#val_ds = val_ds.prefetch(buffer_size = 32)

#Model
model = keras.Sequential()
model.add(Rescaling(scale=(1./127.5),
                    offset=-1,
                    input_shape=(150, 150, 3)))
model.add(Conv2D(16, kernel_size=(6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (6, 6),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(loss = tf.keras.losses.categorical_crossentropy,
              optimizer = tf.keras.optimizers.Adam(1e-3),
              metrics = ['accuracy'])

#Training
epochs = 200
es = EarlyStopping(monitor = 'val_accuracy', mode = 'max', verbose = 1, patience = 7, restore_best_weights = True)
h = model.fit_generator(
    train_ds,
    epochs = epochs,
    validation_data = val_ds,
    callbacks = [es]
)

#View
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.plot(h.history['loss'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation', 'loss'], loc = 'upper right')
plt.show()