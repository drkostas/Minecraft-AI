# Minecraft AI

[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](
https://github.com/drkostas/Minecraft-AI/blob/master/LICENSE)

## Table of Contents

+ [About](#about)
+ [Requirements](#installing)
+ [Running the code](#run)
+ [License](#license)

## About <a name = "about"></a>

The goal of this project is to experiment using Deep Q-learning to play Minecraft.

To get started, follow their respective instructions.

## Requirements <a name = "installing"></a>

Install required libraries:

** Windows **
Follow the instructions [here](https://www.oracle.com/java/technologies/downloads/#java8-windows)

** Macos **
```ShellSession
$ brew tap AdoptOpenJDK/openjdk
$ brew install --cask adoptopenjdk8
```
** Debian-based systems (Ubuntu) **
```ShellSession
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install openjdk-8-jdk

# Verify installation
java -version # this should output "1.8.X_XXX"
# If you are still seeing a wrong Java version, you may use
# the following line to update it
# sudo update-alternatives --config jav
```

Before running the programs, you should first create a conda environment, load it, and install the requirements
like so:

```ShellSession
$ conda create -n minerl python=3.8
$ conda activate minerl
$ pip install -r requirements.txt
```


## Running the code <a name = "run"></a>

In order to run the code, first, make sure you are in the correct virtual environment:

```ShellSession
$ conda activate minerl

$ which python
/home/drkostas/anaconda3/envs/minerl/bin/python

```

Run a minimal example:

```ShellSession
$ python example.py
```



## License <a name = "license"></a>

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
