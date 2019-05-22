import os, math, json
import numpy as np
from matplotlib import pyplot as plt
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.applications import VGG16, InceptionV3
from keras import metrics, losses
from sklearn.metrics import classification_report



# Varibles and parametes used in the model
classes = ['Negative', 'Positive']
num_classes = 2
batch_size = 64
epochs = 1 
steps_per_epochs = 300
learning_rate = 0.0001
metric = ["accuracy"]
loss = "categorical_crossentropy"


train_path = '/home/sofus/deep/data/train/'
test_path = '/home/sofus/deep/data/test/'
val_path = '/home/sofus/deep/data/val/'


# Create data sets, validations set from data paths
train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_path,
    classes=classes,
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    val_path,
    classes=classes,
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_path,
        classes=classes,
        batch_size=batch_size,
        class_mode='categorical')




"""
 Import models for training
 make sure that the top layer is not include so
 a layer can be added for the needed classification task
"""
# model = IncetptionV3(weights = "imagenet", include_top=False, classes=num_classes)
model = VGG16(weights='imagenet', include_top=False)

# Add layers and a layer for the classification of the 2 classes
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x) # Added 3 layes to enhance perfomance
predictions = Dense(num_classes, activation='softmax')(x) # final layer with softmax

# Final model
final_model = Model(input=model.input, output=predictions)
final_model.compile(loss = loss,
        optimizer = optimizers.SGD(lr=learning_rate, momentum=0.9),
        metrics=metric)


# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5",
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto', period=1)

# Fit the model for training on the image generators
history = final_model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epochs,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=300)


"""
# serialize model to JSON
model_json = final_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen, steps=300)
 
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
 
# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys())
"""

# Plot the different training
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

