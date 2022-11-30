# Minecraft AI

[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](
https://github.com/drkostas/Minecraft-AI/blob/master/LICENSE)

## Table of Contents

+ [About](#about)
+ [Requirements](#installing)
    + [OS Requirements](#osinstalling)
    + [Python Requirements](#pyinstalling)
+ [Running the code](#run)
    + [Minecraft Client](#runminecraft)
    + [RL Agent](#runrl)
+ [License](#license)

## About <a name = "about"></a>

The goal of this project is to experiment using Deep Q-learning to play Minecraft.

To get started, follow their respective instructions.

## Requirements <a name = "installing"></a>

This project has only been tested on Ubuntu 22.04. In theory any non arm based OS should work, but it has not been tested.


### OS Requirements <a name = "osinstalling"></a>
First, you need to install the follwioing libraries:

1. GCC and CMake
    ```ShellSession
    sudo apt-get update
    sudo apt install -y gcc
    sudo apt-get install -y make
    ```

2. CUDA 11.0
    ```ShellSession
    wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
    sudo sh cuda_11.0.2_450.51.05_linux.run
    ```

3. [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)

4. Java 8
    ```ShellSession
    sudo apt-get install -y openjdk-8-jdk
    echo -e "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64" >> ~/.bashrc
    source ~/.bashrc
    ```

### Python Requirements <a name = "pyinstalling"></a>

1. Now you need to create a conda environment and install the required python packages:
    ```ShellSession
    conda create -n minerl python=3.6
    conda activate minerl
    pip install -r requirements.txt
    ```

2. Finally, you need to bootstrap the Malmo library 
   and set the MALMO_XSD_PATH environment variable as follows:
    ```ShellSession
    python3 -c "import malmo.minecraftbootstrap; malmo.minecraftbootstrap.download();"
    echo -e "export MALMO_XSD_PATH=$PWD/MalmoPlatform/Schemas" >> ~/.bashrc
    source ~/.bashrc
    cd MalmoPlatform/Minecraft
    (echo -n "malmomod.version=" && cat ../VERSION) > ./src/main/resources/version.properties
    cd ../..
    ```

## Running the code <a name = "run"></a>

In order to run the code, first, make sure you are in the correct virtual environment:
```ShellSession
$ conda activate minerl

$ which python
/home/drkostas/anaconda3/envs/minerl/bin/python
```

### Minecraft <a name = "runminecraft"></a>

Run the following command to start the Minecraft client:
```ShellSession
cd MalmoPlatform/Minecraft
./launchClient.sh -port 9000
```

### RL Agent <a name = "runrl"></a>

To run the simple (random) agent run the following command:
```ShellSession
python basic_agent.py -c configs/lava_maze.yml
```



## License <a name = "license"></a>

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
