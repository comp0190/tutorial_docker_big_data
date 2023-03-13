import sys

sys.path.append('./src/')

import argparse
import os

if __name__ == '__main__':
    # add parser to read config file path
    parser = argparse.ArgumentParser(
        # Turn off help, so we print all options in response to -h
        add_help=False
    )
    parser.add_argument("-c", "--config",
                        help="Specify a configuration file",
                        metavar="FILE")
    args = parser.parse_args()

    # load dataset
    from tensorflow import keras
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # load config file
    import yaml

    with open(args.config, 'r') as file:
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
    cnn.save(os.path.join(config['model_path'], config['model_name']))


