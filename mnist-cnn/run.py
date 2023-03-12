import sys
sys.path.append('./src/')

# load config file
import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# prepare dataset
import preprocess_dataset
processed_dataset = preprocess_dataset.process_dataset(x_train, y_train, x_test, y_test)

# model training
import model
cnn = model.initiate_model()
model = keras.models.load_model(config)