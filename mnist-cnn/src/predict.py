import numpy as np

def predict(model, image, process=False):
    if process==True:
        np_img = np.asarray(image)
        np_img = np.expand_dims(np_img, 0)
    pred = np.argmax(model.predict(np_img), axis=1)
    return pred

