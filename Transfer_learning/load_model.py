import os
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image




# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
 # evaluate loaded model on test data
 loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
 score = loaded_model.evaluate(X, Y, verbose=0)
 print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
    


# dimensions of our images
img_width, img_height = 224, 224
for i in range(0, 50):
    # predicting images
    img = image.load_img('/home/sofus/deep/data/val/Positive/184{:02d}_1.jpg'.format(i), target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = loaded_model.predict(images, batch_size=10)
    print(np.argmax(classes, axis=1))
