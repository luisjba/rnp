#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNISTModel class definition 
to implement a neural network that recognize MNIST 
handwritten digits.

Usage:
  nn_mnist_model.py help
  nn_mnist_model.py version
  nn_mnist_model.py net LAYERS
  nn_mnist_model.py board

Arguments:
  LAYERS        The Net Layer, ex. "[128,'relu'],[128,'relu']"

Options:
  -h, help         Show help
  -v, version      Version

@datecreated: 2022-02-09
@lastupdated: 2022-02-17
@author: Jose Luis Bracamonte Amavizca
"""
# Meta informations.
__author__ = 'Jose Luis Bracamonte Amavizca'
__version__ = '0.0.1'
__maintainer__ = 'Jose Luis Bracamonte Amavizca'
__email__ = 'luisjba@gmail.com'
__status__ = 'Development'

from cProfile import label
import datetime, ast
from statistics import mode
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
            verbose:int = 1,
            auto_load_data=False
        ) -> None:
        self.epochs:int = epochs
        self.batch_size:int = batch_size
        self.verbose:int = verbose
         # number of outputs = number of digits
        self.nb_classes:int = 10
        # X_train is 60000 rows of 28x28 values; we  --> reshape it to
        # 60000 x 784.
        self.input_shape:int = 784
        # how much TRAIN is reserved for VALIDATION
        self.validation_split:float = 0.2
        # Load Data
        self.load_data()
        # Initialize NN model
        self.model:tf.keras.Model = tf.keras.models.Sequential()
        # To store the train history
        self.history = []
        if auto_load_data:
            self.load_data()
    

    def load_data(self) -> None:
        # Loading MNIST dataset.
        # verify
        # You can verify that the split between train and test is 60,000, 
        # and10,000 respectively.
        # Labels have one-hot representation.is automatically applied
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = mnist.load_data()

        self.X_train = self.X_train.reshape(60000, self.input_shape)
        self.X_test = self.X_test.reshape(10000, self.input_shape)
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        # Normalize inputs to be within in [0, 1].
        self.X_train /= 255
        self.X_test /= 255
        # One-hot representation of the labels.
        self.Y_train = keras.utils.to_categorical(self.Y_train, self.nb_classes)
        self.Y_test = keras.utils.to_categorical(self.Y_test, self.nb_classes)

    def get_sample_images(self, n:int=18, cols:int=6) -> tuple:
        """Get the  fig and axs for image ploting
        
        :param n: the number of images to take
        :param cols: the number of columns per row
        
        :return (fig,axs): the plot objects
        """
        last_blank_cols = n % cols
        plot_rows = int(n/cols) + last_blank_cols
        fig, axs = plt.subplots(plot_rows, cols, figsize=(6,6), tight_layout=True)
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
            ax.imshow(gray_image, cmap="gray_r")
            ax.set_title("Digit: {}".format(digit), fontsize=14)
            ax.axis('off')
            #ax.axes.get_xaxis().set_visible(False)
            #ax.axes.get_yaxis().set_visible(False)

        # Clear the last blank columns
        for i in range(last_blank_cols):
            axs.flat[-(i+1)].axis('off') # clear existing plot
        return fig, axs

    def add_acc_plot(self, fig:Figure, ax:Axes) -> tuple:
        """Add the accuracy plot """
        acc = self.history.history["accuracy"]
        val_acc = self.history.history['val_accuracy']
        epochs = self.history.params["epochs"]
        plot_tile = "Accuracy -> T:{:.2f}%, V:{:.2f}% ".format(acc[-1]*100, val_acc[-1]*100)
        if len(self.model.layers) > 1:
            plot_tile = "{} {}".format(plot_tile, self.descriptive_name)
        ax.set_title(plot_tile)
        epochs_line = range(epochs)
        ax.plot(epochs_line, acc, label="Train")
        ax.plot(epochs_line, val_acc, label="Validation")
        ax.legend(loc="lower right")

    def add_loss_plot(self, fig:Figure, ax:Axes) -> tuple:
        """Add the loss plot """
        loss = self.history.history["loss"]
        val_loss = self.history.history['val_loss']
        epochs = self.history.params["epochs"]
        plot_tile = "Loss -> T:{:.2f}%, V:{:.2f}% ".format(loss[-1]*100, val_loss[-1]*100)
        if len(self.model.layers) > 1:
            plot_tile = "{} {}".format(plot_tile, self.descriptive_name)
        ax.set_title(plot_tile)
        epochs_line = range(epochs)
        ax.plot(epochs_line, loss, label="Train")
        ax.plot(epochs_line, val_loss, label="Validation")
        ax.legend(loc="upper right")

    def add_learning_curve_plots(self, fig:Figure, ax_acc:Axes, ax_loss:Axes):
        """Adding the Learning Cuerve to the corresponding sub plot
        
        :param ax_acc:Axes The Axes object to plot the Accuracy data
        :param ax_loss:Axes The Axes object to plot the Loss data

        :return fig:Figure The Figure object 
        :return (ax_acc,ax_loss: The tuple of axes for Accuracy and Loss
        """
        self.add_acc_plot(fig, ax_acc)
        self.add_loss_plot(fig, ax_loss)
        return fig, (ax_acc,ax_loss)

    def get_learning_curve_plots(self):
        """Generate the plots for Learning cuerve

        :return fig:Figure The Figure object 
        :return (ax_acc,ax_loss: The tuple of axes for Accuracy and Loss
        """
        fig, axs = plt.subplots(1, 2, figsize=(16,8))
        ax_acc, ax_loss = axs.flat[:2]
        # Superior Title
        fig.suptitle("Learning Curve", fontsize=18, fontweight='bold')
        fig.tight_layout()
        self.add_learning_curve_plots(fig, ax_acc, ax_loss)
        return fig, (ax_acc,ax_loss)


    def build(self, hidden_layers:list=[]) -> None:
        """Build the model with the corresponding layers.
        
        :param hidden_layers: A list of elements with the number of units in 
                            the first position and activation in the second.
                            Example -> [[128, 'relu'], [64, 'relu']]
        
        """
        self.model:tf.keras.Model = tf.keras.models.Sequential()
        for i, h_layer in enumerate(hidden_layers):
            l_units, l_activation = h_layer[:2]
            layer_kargs = {
                "name": "dense_layer-{}".format(i + 1),
                "activation" : l_activation
            }
            # Connect to the input layer for the first hidden layer
            if i == 0:
                layer_kargs["input_shape"] = (self.input_shape,)
            self.model.add(keras.layers.Dense(l_units, **layer_kargs))
        # adding the output layer
        layer_kargs = {
            "name": "dense_layer-{}".format(len(hidden_layers) + 1),
            "activation" : "softmax"
        }
        # Connect to the input layer if there aren't hidden layers
        if len(hidden_layers) == 0:
            layer_kargs["input_shape"] = (self.input_shape,)
            layer_kargs["name"] =  'dense_layer'
        self.model.add(keras.layers.Dense(self.nb_classes, **layer_kargs))

    def summary(self):
        """Return the summary of the Model"""
        return self.model.summary()

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
        """Return the log dir"""
        return "logs/fit/{}{}".format(self.descriptive_name ,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    @property
    def descriptive_name(self):
        """Return the desciptive name base on the layers configuration"""
        if len(self.model.layers) < 2:
            return ""
        name = ""
        for layer in self.model.layers[1:-1]:
            l_conf = layer.get_config()
            name = "{}__{}_{}".format(name, l_conf["units"], l_conf["activation"])
        return name


    def fit(self):
        tf.config.set_soft_device_placement(True)
        fit_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        self.history = self.model.fit(
            x = self.X_train, 
            y = self.Y_train,
            batch_size = self.batch_size,
            epochs = self.epochs,
            verbose = self.verbose,
            validation_data = (self.X_test, self.Y_test),
            callbacks = [fit_callback]
        )

    def evaluate(self) -> tuple:
        """Eavaluate the model
        
        :return (loss, acc): The loss and accuracy of the model base on the test set
        """
        return self.model.evaluate(self.X_test, self.Y_test)

def parse_srt_representation_of_layers(srt_rpr_layers:str) -> list:
    """Parse a string representation of layers for a 
    Neural Network

    return list: list of layer with format [[units,activation]]
    """
    layers = []
    if srt_rpr_layers is not None:
        try:
            parsed_layers = ast.literal_eval(srt_rpr_layers)
            layer_item = []
            for layer in parsed_layers:
                if type(layer) is list:
                    # get the first two items for units and activation from list
                    layer_item = layer[:2]
                elif type(layer) is int and len(layer_item) == 0:
                    # Append the units in first position
                    layer_item.append(layer)
                elif type(layer) is str and len(layer_item) == 1:
                    # Append the activation in the second position
                    layer_item.append(layer)
                # Append a layer if completed    
                if len(layer_item) >=2 :
                    layers.append(layer_item)
                    layer_item = [] # Reset to find the next layer
        except Exception as e:
            print("Invalid format of 'LAYERS' with value:{}".format(srt_rpr_layers))
    return layers


def run_default_net(args):
    
    hidden_layers = parse_srt_representation_of_layers(args.get('LAYERS'))
    mnist_model = MNISTModel(
        epochs=200,
        batch_size=128
    )
    mnist_model.load_data()
    print("{} train samples. Each sample is a flaten vector of {} elements".format(mnist_model.X_train.shape[0],mnist_model.X_train.shape[1] ))
    print("{} test samples".format(mnist_model.X_test.shape[0]))
    print("{} Labels are One-hot encodded as columns".format(mnist_model.Y_train.shape[1]))
    # Build the model.
    mnist_model.build(hidden_layers)
    # Summary of the model
    mnist_model.summary()
    # Compiling the model.
    mnist_model.compile()
    # Training the model.
    mnist_model.fit()
    #evaluate the model
    test_loss, test_acc = mnist_model.evaluate()
    print('Test accuracy:', test_acc)

def run_tensorboard(args):
    import os
    tensorBoardPath = "logs/fit/"
    os.system('tensorboard --logdir=' + tensorBoardPath)
    return

def _tb_launch_task(args):
    import threading
    t = threading.Thread(target=run_tensorboard, args=([args]))
    t.start()

if __name__ == '__main__':
    def help():
        print(__doc__)
    def version():
        print('Version %s' % __version__)
    from docopt import docopt
    args = docopt(__doc__, version=__version__)
    if args.get('help'):
        help()
    elif args.get('version'):
        version()
    elif  args.get('net'):
        run_default_net(args)
    elif  args.get('board'):
        _tb_launch_task(args)
    else:
        print("Invalid command")
        help()
