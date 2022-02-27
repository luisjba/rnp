# Environment for Tensorflow for mac M1 

## Installation of miniforce via homebriew

```bash
brew install miniforge
```

## Python env

To install TensorFlow on Apple’s M1 machines, first download the environment.yml file from [https://raw.githubusercontent.com/mwidjaja1/DSOnMacARM/main/environment.yml](https://raw.githubusercontent.com/mwidjaja1/DSOnMacARM/main/environment.yml)

```bash
curl https://raw.githubusercontent.com/mwidjaja1/DSOnMacARM/main/environment.yml --output environment.yml
```

Create the python environment with conda

```bash
conda env create --file=environment.yml --name tf_m1
```

Activate the new `tf_m1` environment in conda

```bash
conda activate tf_m1
```
Now, you will see the environment in paranthesis in you promt `(tf_acarus) [user@host]$ `


# Install Tensorflow-metal Pluggable Device into Mac M1

Create a directory to store all the necesary files

```bash
mkdir env_tf_arm64_m1
cd env_tf_arm64_m1
```

Download and install the conda env for arm64 Apple silicon

```bash
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
```

Install the Tensorflow dependencies

```bash
conda install -c apple tensorflow-deps
```

## Install Tensorflow

```bash
python -m pip install -U pip
python -m pip install tensorflow-macos
```

## Install Tensorflow-metal plugin

```bash
python -m pip install tensorflow-metal
```


## Install other essential libraries

```bash
conda install -c conda-forge matplotlib -y
conda install -c conda-forge scikit-learn -y
conda install -c conda-forge opencv -y
conda install -c conda-forge pandas -y
```