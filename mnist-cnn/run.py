import sys
sys.path.append('./src/')

# load config file
import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# prepare dataset
import preprocess_dataset
processed_dataset = preprocess_dataset.process_dataset(x_train, y_train, x_test, y_test)

# load model
import model
cnn = model.initiate_model()
model = keras.models.load_model(os.path.join(config['model_path'], config['model_name']))

y_predict = predict(cnn, processed_dataset['images_test'])
y_true = np.argmax(processed_dataset['labels_test'], 1)

evaluate_model(y_true, y_predict, output_type='print2file', output_path='./results.txt')