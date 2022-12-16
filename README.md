# Minecraft AI

[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](
https://github.com/drkostas/Minecraft-AI/blob/master/LICENSE)

## Table of Contents

+ [About](#about)
+ [Requirements](#installing)
    + [OS Requirements](#osinstalling)
    + [Python Requirements](#pyinstalling)
+ [Domain Problem](#domain)
+ [Running the code](#run)
    + [Minecraft Client](#runminecraft)
    + [RL Agent](#runrl)
+ [Results](#res)
+ [Conclusion](#conc)
+ [License](#license)

## About <a name = "about"></a>

In this project, we used reinforcement learning to train a PPO agent to solve maze missions in Minecraft using the Malmo
library. The agent was tested using different action spaces rewards, and compression techniques. Our results showed that it was able
to successfully navigate the mazes and complete the missions. This project demonstrates the potential of reinforcement learning for
solving complex problems in gaming environments and has potential applications in a wide range of fields. It represents an important
step towards realizing the full potential of reinforcement learning for solving complex problems in virtual environments.

The Model structure:<br>
![Model Structure](https://github.com/drkostas/Minecraft-AI/blob/master/img/model.png?raw=true)

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
   and set the `MALMO_XSD_PATH` environment variable as follows:
    ```ShellSession
    python3 -c "import malmo.minecraftbootstrap; malmo.minecraftbootstrap.download();"
    echo -e "export MALMO_XSD_PATH=$PWD/MalmoPlatform/Schemas" >> ~/.bashrc
    source ~/.bashrc
    cd MalmoPlatform/Minecraft
    (echo -n "malmomod.version=" && cat ../VERSION) > ./src/main/resources/version.properties
    cd ../..
    ```

## Domain Problem <a name = "domain"></a>
The problem we are working on is using reinforcement
learning to train an agent to solve maze missions in
Minecraft using the Malmo library. In this problem, the
agent is placed in a maze within the Minecraft game world
and must navigate the maze to reach the goal. The rules
of the game are defined by the Minecraft environment and
the specific maze mission that the agent is attempting to
solve. To frame this problem as an MDP (Markov Decision
Process), we need to define the state space, action space, and
reward structure. The state space in this problem is the set
of all possible configurations of the Minecraft game world
that the agent can encounter while navigating the maze.
This includes the location of the agent, the layout of the
maze, and any other relevant information that the agent can
observe.

The action space in this problem is the set of all possible
actions that the agent can take at any given time. In the
context of Minecraft, these actions might include moving
in different directions, turning left and right, interacting
with objects in the game world, or using items in the
agent’s inventory. The reward structure in this problem is
the set of rewards that the agent receives for taking different
actions and achieving different goals within the maze. These
rewards might include positive rewards for reaching the
goal, negative rewards for encountering obstacles or hazards, and other rewards for achieving intermediate goals or
completing certain tasks.

We examined different options for the state space, action
space, and reward structure in this problem. For the state
space, we used the raw pixel data from the Minecraft game
screen as input to the agent, as well as using pre-processed
representations of the game state that extracted relevant
features and abstracted away irrelevant details. An example
of what the agent sees can be found in Figure 1. For the
action space, we considered using a discrete set of actions
that the agent could take. For the reward structure, we
considered using a variety of different rewards for different
actions and goals, including both intrinsic rewards that were
defined by the game environment and extrinsic rewards that
were defined by the objective of the maze mission.

The environment that we designed was a maze variant
that forced the agent to rely on the visual input. The agent’s
goal was to touch the tower of emerald blocks in the shortest
time possible and without dying. There were two obstacles
to this goal; First there were small walls that were scattered
around the environment. These walls would stop the agent
from being able to move in a direction while still allowing
the agent to have a sight line on the target. The second set
of obstacles was fires that were generated throughout the
environment. These fires would spread as the episode ran,
creating a dynamic environment. The reward system was
set up so if the agent successfully made it to the tower, it
would receive a large positive reward while if the episode
timed out, or the agent walked into fire it would receive a
large negative reward. In addition, the agent would get a
small negative reward at every step to incentivize finding
the target faster.

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

The missions are places in the `missions` folder and the configuration files in the `configs` folder.
To train the agent run the following command:
```ShellSession
python train_agent.py -c configs/mazes.yml
```

The results are going to be saved in the `checkpoints` and `logs` folders.

An example of what the agent sees:<br>
![What the agent sees](https://github.com/drkostas/Minecraft-AI/blob/master/img/agent-view.png?raw=true)

The maze outline:<br>
![Maze outline](https://github.com/drkostas/Minecraft-AI/blob/master/img/maze.png?raw=true)


## Results <a name = "res"></a>

The PPO model showed promising levels of learning when
implemented in the correct conditions. Each model was run
for 50 epoch which took approximately 5 hours of run time.
Our initial agent interface allowed the agent to take six
different actions: move forward, move backward, move left,
move right, turn left, and turn right. Early tests of this action
space mainly proved successful, however, the model rarely
turned to observe its surroundings. Instead, the model used
the image to avoid obvious obstacles in front of itself while
moving in semi-random directions until it reached the pillar.
This behavior would likely be fixed given extensive training
times, however turning only provided potential delayed
rewards, inhibiting its regular use. To force the model to
rely on the image further we restricted the action space to
four actions: move forward, backward, turn right, and turn
left. Therefore, the model is now required to turn to move in
a specific direction. This quickened the model learning and
led to the rewards shown in the figure below:<br>
![AVG Reward - No PCA](https://github.com/drkostas/Minecraft-AI/blob/master/img/nopca.png?raw=true)

Training with the modified action space shows clear
improvement over time. However, there is a brief but significant dip that appears during epochs 30 to 32. While it
is unclear what caused this dip, it recovered quickly and
continued the improvement trend. After 50 training epochs
the average reward is consistently close to the maximum
allowed by the environment. This shows that the agent has
not only learned to move around the walls in an efficient
manner, but learned to avoid walking into areas where fire
is spreading.

Using the PCA to reduce the dimensionality of the image
proved to interfere with models’ ability to train. Figure
6 shows that the model makes no improvement over the
50 training epochs. It is likely that while PCA is able to
retain a significant amount of the information, the reduction
method removes trends that the agent uses to learn. It is
possible applying PCA in a different manner or introducing
a different dimensionality reduction technique could lead to
an improved result. Figure 6:<br>
![AVG Reward - With PCA](https://github.com/drkostas/Minecraft-AI/blob/master/img/pca.png?raw=true)


## Conclusion <a name = "conc"></a>
The PPO model showed a promising ability to complete
tasks in a simple dynamic environment using a convolutional network as its only input. 
When the data is directly fed the model learns at a mostly consistent rate until it has
found the optimal behavior. This model could be further
tested by complicating the environment. The map could be
expanded, or the site lines could be restricted, requiring a
more complex and comprehensive decision-making process.

In addition, other dynamic variables could be introduced,
including hostile non player characters.
The attempt to use dimensionality reduction to reduce
training time proved to be a failure with the current method.
PCA appeared promising as it is a faster method than other
reduction techniques and allows for tuning of the data
reduction amount. However, it appears that the method
fails to capture the patterns the model requires in order
to learn. Future work could try different techniques for
dimensionality reduction or improve the application of PCA
through tuning.

## License <a name = "license"></a>

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
