# load dataset
from tensorflow import keras
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import sys
sys.path.append('./src/')

# load config file
import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# prepare dataset
import preprocess_dataset
processed_dataset = preprocess_dataset.process_dataset(x_train, y_train, x_test, y_test)

# train mocdel
import model
cnn = model.initiate_model()

cnn.compile(loss=config['loss'], optimizer=config['optimizer'], metrics=[config['metrics']])
cnn.fit(
    processed_dataset['images_train'],
    processed_dataset['labels_train'],
    batch_size=config['batch_size'],
    epochs=config['epochs'],
    validation_split=config['validation_split'])

# save model
import  os
cnn.save(os.path.join(config['model_path'], config['model_name']))


