import os
import numpy
from keras.models import load_model
from keras.preprocessing import image
base_dir = '/Users/heathivie/Downloads/cats_and_dogs_small'
# load model
model = load_model('ch_5_cat_dogs.h5')
# summarize model.
model.summary()
file = test_dir = os.path.join(base_dir, 'test/cats/banana.jpeg')
f = image.load_img(file, target_size=(150, 150, 3))
x = image.img_to_array(f)
# the first param is the batch size

y = x.reshape((1, 150, 150, 3)).astype('float32')

classes = model.predict_classes(y)
print(classes)
