#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNISTModel class definition 
to implement a neural network that recognize MNIST 
handwritten digits.
@datecreated: 2022-02-09
@lastupdated: 2022-02-09
@author: Jose Luis Bracamonte Amavizca
"""
# Meta informations.
__author__ = 'Jose Luis Bracamonte Amavizca'
__version__ = '0.0.1'
__maintainer__ = 'Jose Luis Bracamonte Amavizca'
__email__ = 'luisjba@gmail.com'
__status__ = 'Development'

import datetime
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist

class MNISTModel():
    """The MINIST Model"""

    def __init__(self, 
            epochs:int=200,
            batch_size:int=128,
            verbose:int = 1
        ) -> None:
        self.epochs:int = epochs
        self.batch_size:int = batch_size
        self.verbose:int = verbose
         # number of outputs = number of digits
        self.nb_classes:int = 10
        # how much TRAIN is reserved for VALIDATION
        self.validation_split:float = 0.2
        # Load Data
        # X_train is 60000 rows of 28x28 values; we  --> reshape it to
        # 60000 x 784.
        self.reshaped:int = 784
        self.load_data()
        # Initialize NN model
        self.model = tf.keras.models.Sequential()
    

    def load_data(self) -> None:
        # Loading MNIST dataset.
        # verify
        # You can verify that the split between train and test is 60,000, 
        # and10,000 respectively.
        # Labels have one-hot representation.is automatically applied
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = mnist.load_data()

        self.X_train = self.X_train.reshape(60000, self.reshaped)
        self.X_test = self.X_test.reshape(10000, self.reshaped)
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        # Normalize inputs to be within in [0, 1].
        self.X_train /= 255
        self.X_test /= 255
        # One-hot representation of the labels.
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train, self.nb_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test, self.nb_classes)

    def get_sample_images(self, n:int=9, cols:int=3) -> tuple:
        """Get the  fig and axs for image ploting
        
        :param n: the number of images to take
        :param cols: the number of columns per row
        
        :return (fig,axs): the plot objects
        """
        last_blank_cols = n % cols
        plot_rows = int(n/cols) + last_blank_cols
        fig, axs = plt.subplots(plot_rows, cols, figsize=(16,16), tight_layout=True)
        fig.suptitle('MNIST images', fontsize=18, fontweight='bold')
        fig.tight_layout()
        # Get the sample number
        m = len(self.X_train)
        # delimite the n to be maximun the number of m samples
        n = min(m,n)
        # select random indexes with size = n
        indexes = np.random.choice(range(m), size=n, replace=False)
        for i_plt, i_data in enumerate(indexes):
            ax = axs.flat[i_plt]
            gray_image = self.X_train[i_data].reshape(28,28)
            digit = self.Y_train[i_data].argmax()
            ax.imgshow(gray_image, cmap="gray_r")
            ax.set_title("Digit: {}".format(digit))
            ax.axis('off')
            #ax.axes.get_xaxis().set_visible(False)
            #ax.axes.get_yaxis().set_visible(False)

        # Clear the last blank columns
        for i in range(last_blank_cols):
            axs.flat[-(i+1)].axis('off') # clear existing plot
        return fig, axs

    def build(self):
        self.model.add(
            keras.layers.Dense(self.nb_classes,
                    input_shape=(self.reshaped,),
                    name='dense_layer',
                    activation='softmax'
                )
            )

    def compile(self, 
            optimizer:str="SGD", 
            loss:str="categorical_crossentropy",
            metrics:list=['accuracy']) -> None:
        """Compile the model"""
        self.model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)

    @property
    def log_dir(self) -> str:
        return "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def train(self):
        fit_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        self.model.fit(
            x = self.X_train, 
            y = self.Y_train,
            batch_size = self.batch_size,
            epochs = self.epochs,
            verbose = self.verbose,
            validation_data = (self.X_test, self.Y_test),
            callback = [fit_callback]
        )

