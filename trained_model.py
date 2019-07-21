from keras.models import model_from_json
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import io
import tensorflow as tf
from keras.optimizers import SGD
import os, ModelType

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

LABELS_MAKE = {0: 'Audi', 1: 'BMW', 2: 'Benz', 3: 'Toyota', 4: 'Volkswagen'}
LABELS_MODEL = {0: 'Accord_213', 1: 'Audi A4L_2', 2: 'Audi A5 convertible_22', 3: 'Audi A5 coupe_23', 4: 'Audi A5 hatchback_24', 5: 'Audi A8L_27', 6: 'Audi Q5_5', 7: 'Aveo sedan_1917', 8: 'BAW E Series hatchback_342', 9: 'BMW 3 Series_68', 10: 'BMW 5 Series_70', 11: 'BMW X1_69', 12: 'BMW X5_105', 13: 'Benz E Class couple_154', 14: 'Benz E Class_127', 15: 'Benz GLA Class_175', 16: 'Benz GLK Class_126', 17: 'Besturn B70_367', 18: 'Bora_500', 19: 'Camry_1386', 20: 'Citroen C5_1259', 21: 'Civic_251', 22: 'Corolla_1323', 23: 'Cruze sedan_1908', 24: 'EXCELLE  GT_196', 25: 'Fiesta sedan_586', 26: 'Fit_215', 27: 'Golf_501', 28: 'KIA K2 sedan_1181', 29: 'Lacrosse_192', 30: 'New Focus hatchback_590', 31: 'New Focus sedan_591', 32: 'Octavia_911', 33: 'Pajero_1866', 34: 'Peugeot 307 hatchback_259', 35: 'Peugeot 308CC_293', 36: 'Regal_190', 37: 'Roewe 550_1229', 38: 'Sail sedan_1915', 39: 'Teana_854', 40: 'V3 Lingyue_533', 41: 'Volkswagen CC_508', 42: 'Volvo C30_1701', 43: 'Volvo XC60_1706', 44: 'Volvo XC90 abroad version_1707', 45: 'Wrangler_735', 46: 'Xiali N7_1071', 47: 'Yuexiang sedan_410', 48: 'c-Elysee sedan_1251', 49: 'smart fortwo_1692'}

####################################
# TEST IMAGES PROCESSING FUNCTIONS
# IMAGES OUTSIDE STANFORD DATA SCOPE
####################################
def fetch_prepare_image_from_url(url, car_model="N/A"):
    # load an image from file
    img = io.imread(url)
    img = img.astype("float32")
    img /= float(255)
    img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    print("Image resize to: " + str(img.shape))

    # show image
    plt.imshow(img)
    plt.title(car_model)
    plt.show()

    return img


# re-sizing images
def resize_image(img_name, car_model="N/A"):
    # load an image from file
    img = Image.open(img_name)
    img = img.resize((IMAGE_HEIGHT, IMAGE_WIDTH), Image.ANTIALIAS)
    img = np.array(img)

    # show image
    #plt.imshow(img)
    #plt.title(car_model)
    #plt.show()

    return img

def predict(graph, model, img_name, LABEL):

    image = resize_image(img_name=img_name, car_model=img_name)

    images = np.array([image])
    print(images.shape)

    with graph.as_default():
        prediction = model.predict(images)

    print(prediction)

    predicted = np.argmax(prediction, axis=1)

    pcar = LABEL.get(predicted[0])
    print("Model predicted given image as: " + pcar)

    return pcar


def load_model(modelType:ModelType):
    #keras.backend.clear_session()

    label = None
    if modelType == ModelType.ModelType.car_model:
        label = LABELS_MODEL
    elif modelType == ModelType.ModelType.car_make:
        label = LABELS_MAKE
    else:
        return

    # load the pre-trained Keras model

    path = os.path.join(modelType.value, 'model.json')
    json_file = open(path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    path = os.path.join(modelType.value, 'model.h5')
    model.load_weights(path)
    print("Loaded model and weights from disk for: " + modelType.name)
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.0001, momentum=0.9),
                  metrics=["accuracy"])
    graph = tf.get_default_graph()
    return model, graph, label
