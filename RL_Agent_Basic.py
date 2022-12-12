import argparse
from pathlib import Path
import random
import gym
import json
from typing import Dict
from yaml_config_wrapper import Configuration
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from RLcraft import MalmoMazeEnv
import numpy as np
#from ray.rllib.agents.ppo import PPO

from ray.rllib.algorithms.ppo import PPO


class CustomEnv(gym.Env):
    def __init__(self, config: Dict):
        # Set a random seed for the environment
        maze_seed = random.randint(1, 9999)
        config['env_config']['mazeseed'] = maze_seed
        env_config = config['env_config']

        self.env = MalmoMazeEnv(
            mazeseed=maze_seed,
            width=env_config["width"],
            height=env_config["height"],
            xml=env_config["mission_file"],
            millisec_per_tick=env_config['millisec_per_tick'],
            max_loop=env_config['max_loop'],
            mission_timeout_ms=env_config['mission_timeout_ms'],
            step_reward=env_config['step_reward'],
            win_reward=env_config['win_reward'],
            lose_reward=env_config['lose_reward'],
            action_space=env_config['action_space'],
            client_port=env_config['client_port'],
            time_wait=env_config['time_wait'])
        self.observation_space = gym.spaces.Box(high=355,
                                                low=0,
                                                shape=(env_config["height"],
                                                       env_config["width"],
                                                       3),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.env.action_space))
        print(self.action_space)

    def reset(self):
        x = self.env.reset()
        return x

    def step(self, action):
        print(action)
        x = self.env.step(action)

        info = {
            # "obs":x[3].observations,
            # "rewards":x[3].rewards,
            # "frames":x[3].number_of_video_frames_since_last_state,
            "rewards": x[3].rewards[0].getValue()
        }

        return x[0], x[1], x[2], info

    @staticmethod
    def process_obs(np_obs, info):
        """ Process the observation from the environment. """
        # obs is a numpy array of shape (height, width, 3)
        # info is a dictionary but we have to transform it to use it
        info_obs = json.loads(info.observations[-1].text)
        floor_data = info_obs['floor10x10']
        time_data = info_obs['TotalTime']
        xpos_data = info_obs['XPos']
        ypos_data = info_obs['YPos']
        zpos_data = info_obs['ZPos']
        yaw_data = info_obs['Yaw']  # where the player is facing
        hp_data = info_obs['Life']
        obs = {}
        obs['rgb'] = np_obs  # Eg: (240, 320, 3) np array
        obs['floor'] = floor_data  # Eg: ['air', 'air', 'beacon', ...]
        obs['time'] = time_data  # Eg: 18196 (time passed)
        obs['xpos'] = xpos_data  # Eg: 3.5
        obs['ypos'] = ypos_data  # Eg: 227.0
        obs['zpos'] = zpos_data  # Eg: 3.5
        obs['yaw'] = yaw_data  # Eg: 270.0
        obs['hp'] = hp_data  # Eg: 20.0 (max)
        return obs


def get_args():
    parser = argparse.ArgumentParser()
    # Required Args
    required_args = parser.add_argument_group('Required Arguments')
    config_file_params = {
        'type': argparse.FileType('r'),
        'required': True,
        'help': "The configuration yml file"
    }
    required_args.add_argument('-c', '--config-file', **config_file_params)
    args = parser.parse_args()
    return args


def get_train_name(name, c):
    """ Get the name of the training session. """
    e = c['env_config']
    hiddens = [str(h) for h in c['model']['fcnet_hiddens']]
    hiddens = '+'.join(hiddens)
    name = f"{name}_{e['width']}width_{e['millisec_per_tick']}ticks_"\
           f"{e['mission_timeout_ms']}timeout_{e['step_reward']}step_"\
           f"{e['win_reward']}win_{e['lose_reward']}lose_{len(e['action_space'])}actions_"\
           f"{e['time_wait']}wait_{e['max_loop']}loop_{hiddens}hiddens"
    return name


def main():
    """ Train the agent. """
    # Read command arguments
    args = get_args()
    # Load YML config file
    c = Configuration(config_src=args.config_file)
    # Load configs from config class
    general_config = c.get_config('general')['config']
    train_config = c.get_config('train')[0]['config']
    env_config = train_config['env_config']
    height, width = env_config['height'], env_config['width']
    train_config['model']['conv_filters'] = (height, width, 3)
    train_name = get_train_name(name=general_config['name'], c=train_config)
    print("Training session name: ", train_name)

    algo = PPO(env=CustomEnv, config=train_config)
    for _ in range(5):
        (algo.train())


if __name__ == '__main__':
    main()
