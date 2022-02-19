# Deep Neural Network

This respository is itended for execrsises related
with experimentation and research with Neural Networks in
my learning proccess for the Subject of **Deep Neural Network**
of the Mater Degree in Data Science by the "Universidad de Sonora".

# Python environment for Acarus

Download the [environment.yml](https://raw.githubusercontent.com/luisjba/rnp/master/environment.yml) 
file thta contains the requiered python package to install in out environment.

```bash
curl https://raw.githubusercontent.com/luisjba/rnp/master/environment.yml --output environment.yml
Collecting package metadata (repodata.json): done
Solving environment: done

Downloading and Extracting Packages
ipython-8.0.1        | 1.1 MB    | ################################################################################################################################################ | 100% 
libffi-3.4.2         | 57 KB     | ################################################################################################################################################ | 100%
.
.
.
.
tomli-2.0.1          | 16 KB     | ################################################################################################################################################ | 100% 
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate tf_acarus
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

Create the python environment with conda form the [environment.yml](environment.yml) file.

```bash
conda env create --file=environment.yml --name tf_acarus
```
Once you run the previos commnd, you have to wait to conda install and prepare the environment
with the packages present in the `environment.yml` file.

To activate the new `tf_acarus` environmet just created, use the `ativate` conda command and
pass the envoronment name as argument.

```bash
conda activate tf_acarus
```
To deactivate the `tf_acarus` environment, use `conda deactivate` and the base environment 
will be activated.

Pip need to be upgraded in order to install aditional and latest essential packages.


```bash
python -m pip install -U pip
# Other essential packages
conda install -c conda-forge matplotlib -y
conda install -c conda-forge scikit-learn -y
conda install -c conda-forge opencv -y
conda install -c conda-forge pandas -y
```

To finish, we have to install Tensorflow

```bash
python -m pip install tensorflow

```

To show the list of Pyhon Packages installed in the environment, run `conda list`.


# MNIST Handwriting digits

TensorFlow version 2.7.0 is used to implement the NN building. The code implementation was made as module in the file [`nn_mnist_model.py`](nn_mnist_model.py) and usage
for the basic NN example is in the jupyter notebook [`nn_mnist.ipynb`](nn_mnist.ipynb). The module have been implemented to be used in a terminal and accept parameters.

## Usage in console with paramaters

To show the available options, you can pass the `--help` option to the `nn_mnist_model.py` file to get the help and detailed options.

```bash
$ python nn_mnist_model.py --help

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

```

### Basic usage to train a simple net

The defaul net architecture is a one Dense connected Layer with 7,850 trainable parameters.

```bash
$ python nn_mnist_model.py net
```

If you want to improve your net architecture, you can pass the `LAYERS` parameter after the `net` command. For example, to add two hidden layer with 128 units and `relu` activation, you can use the following text parameter as  representation of the hidden layers architecture.

```bash
$ python nn_mnist_model.py net "[128,'relu'],[128,'relu']"
```

You can add as much layers as you want on each list item comma separated in the argument.

### Launch TensorBoard

The module have the implementation to run the TensorBoard server and visualiza 
the train reults.

```bash
$ python nn_mnist_model.py net board

NOTE: Using experimental fast data loading logic. To disable, pass
    "--load_fast=false" and report issues on GitHub. More details:
    https://github.com/tensorflow/tensorboard/issues/4784

Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.8.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

By default, the board try to use the port 6006 as [http://localhost:6006/](http://localhost:6006/) if available, if not, will be 
check for availability of the next one until one is free and available tu 
provide the web service.

To stop the server, press `CTRL+C` to quit.