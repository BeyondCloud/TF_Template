import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs_if_not_exist
from utils.logger import Logger
from utils.utils import get_args

# python main.py --config ./configs/example.json
def main():
    # capture the conf -ig path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)
    print(config.summary_dir)
    # create the experiments dirs
    create_dirs_if_not_exist([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    
    # create an instance of the model you want
    model = ExampleModel(config)
    with tf.Session() as sess:
        #load model if exists
        model.load(sess)
        # create your data generator
        data = DataGenerator(config)
        # create tensorboard logger
        logger = Logger(sess, config)
        # create trainer and pass all the previous components to it
        trainer = ExampleTrainer(sess, model, data, config, logger)

        # here you train your model
        trainer.train()


if __name__ == '__main__':
    main()
