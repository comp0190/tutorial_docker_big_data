import numpy

def process_dataset(x_train, y_train, x_test, y_test):
    """
    Args:
        x_train: np.ndarray: trainset of 60 000 MNIST 28x28 images of digits
        y_train: np.ndarray: train labels identifying the trainset digits
        x_test: np.ndarray: testset of 10 000 images of 28x28 digits
        y_test: np.ndarray: test labels identifying the testset digits
    """
    res_x_train = np.expand_dims(x_train, 1)
    res_x_test = np.expand_dims(x_test, 1)
    res_y_train = keras.utils.to_categorical(y_train)
    res_y_test = keras.utils.to_categorical(y_test)
    return {
        'images_train': res_x_train,
        'labels_train': res_y_train,
        'images_test': res_x_test,
        'labels_test': res_y_test}