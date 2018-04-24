import tensorflow as tf
from models.example_model import ExampleModel
from utils.config import process_config
from utils.utils import get_args
from data_loader.data_generator import DataGenerator
try:
    args = get_args()
    config = process_config(args.config)
except:
    print("missing or invalid arguments")
    exit(0)
model = ExampleModel(config)
with tf.Session() as sess:
    #load model if exists
    model.load(sess)
    data = DataGenerator(config)
    batch_x, batch_y =next(data.next_batch(config.batch_size))
    feed_dict = {model.x: batch_x, model.y: batch_y, model.is_training: False}
    y_bar = sess.run([model.y_bar],feed_dict=feed_dict)
    print(y_bar)
    print(batch_y)
    