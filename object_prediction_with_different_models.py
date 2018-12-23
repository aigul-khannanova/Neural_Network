from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications import Xception, InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


model = {'InceptionResNetV2': InceptionResNetV2, 'Xception': Xception, 'InceptionV3': InceptionV3, 'ResNet50': ResNet50, 'VGG16': VGG16}

for key, model_type in model.items():

    model = model_type(weights='imagenet')
    img_path = 'test_pictures/imgonline-com-ua-Resize-vRSMPiWDERer.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted for {}: '.format(key), decode_predictions(preds, top=3)[0])