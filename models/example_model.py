from base.base_model import BaseModel
import tensorflow as tf
import numpy as np
def my_activation(x):
    return (tf.exp(x)-1)/(tf.exp(x)+1)
class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 1])

        # network architecture
        out = tf.layers.dense(self.x, 50, activation=tf.nn.relu, name="dense1")
        out= tf.layers.dense(out, 50, activation=tf.nn.relu, name="dense2")
        out= tf.layers.dense(out, 50, activation=tf.nn.relu, name="dense3")
        self.y_bar = tf.layers.dense(out, 1, name="dense4")

        with tf.name_scope("loss"):
            self.sqm = tf.reduce_mean(tf.squared_difference(self.y_bar,self.y),1)
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.sqm,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(self.y_bar, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

