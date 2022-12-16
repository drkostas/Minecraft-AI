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
import os
import time
import matplotlib.pyplot as plt
from PIL import Image

import pickle

#from ray.rllib.agents.ppo import PPO
from sklearn.decomposition import PCA

from ray.rllib.algorithms.ppo import PPO


class CustomEnv(gym.Env):
    def __init__(self, config: Dict):
        # Set a random seed for the environment
        maze_seed = random.randint(1, 9999)
        config['mazeseed'] = maze_seed
        self.env_config = config

        self.env = MalmoMazeEnv(
            mazeseed=maze_seed,
            width=self.env_config["width"],
            height=self.env_config["height"],
            xml=self.env_config["xml"],
            millisec_per_tick=self.env_config['millisec_per_tick'],
            max_loop=self.env_config['max_loop'],
            mission_timeout_ms=self.env_config['mission_timeout_ms'],
            step_reward=self.env_config['step_reward'],
            win_reward=self.env_config['win_reward'],
            lose_reward=self.env_config['lose_reward'],
            action_space=self.env_config['action_space'],
            client_port=self.env_config['client_port'],
            time_wait=self.env_config['time_wait'])
        self.max_path_length = 200
        self.observation_space = gym.spaces.Box(high=355,
                                                low=0,
                                                shape=(self.env_config["height"],
                                                       self.env_config["width"],
                                                       3),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.env.action_space))

    def reset(self):
        maze_seed = random.randint(1, 9999)
        self.env.mazeseed = maze_seed
        x = self.env.reset()
        img = self.pca_image_compress(x)
        img = Image.fromarray(img, 'RGB')
        img.save()
        return x

    def pca_image_compress(self, img):

        img_r = Image.fromarray(img, 'RGB')
        img_r.save('out_n.png')

        pca = PCA(n_components=140)
        img_s = np.reshape(img.transpose((0, 2, 1)),
                           (self.env_config["height"], -1))
        img_r = Image.fromarray(img_s, 'L')
        img_r.save('out_t.png')

        pca.fit(img_s)

        pca_t = pca.transform(img_s)
        pca_recovered = pca.inverse_transform(pca_t)
        img_r = Image.fromarray(pca_recovered, 'L')
        img_r.save('out_r.png')
        x = pca_recovered.reshape((140, 140, 3))

        # temp = pca.inverse_transform(img_t)
        # img_r = np.reshape(temp, (self.env_config["height"],self.env_config["width"],3))
        img_r = Image.fromarray(x, 'RGB')
        img_r.save('out.png')
        return img_r

    def step(self, action):
        # print(self.observation_space)
        x = self.env.step(action)
        # TODO: Option to use the observations from the info (next 2 lines)
        # observations = self.process_obs(x[0], x[3])
        # reward = x[1]
        # while(len(x[3].rewards)==0):
        #    print("___________________________________________")
        #    print(len(x[3].rewards))
        #    x = self.env.step(action)
        # print(len(x[3].rewards))
        # info = {
        # "obs":x[3].observations,
        # "rewards":x[3].rewards,
        # "frames":x[3].number_of_video_frames_since_last_state,
        #    "rewards": x[3].rewards[0].getValue()
        # }
        img = Image.fromarray(x[0], 'RGB')
        img = self.pca_image_compress(img)
        img.save('out.png')
        img = np.array(img)
        # TODO: Is this structured required by rrllib or can we change it?
        return x[0], x[1], x[2], {}

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

    # TODO: We should use a variation of this to encode the block observations
    # @staticmethod
    # def gridProcess(state):
    #     msg = state.observations[-1].text
    #     observations = json.loads(msg)
    #     grid = observations.get(u'floor10x10', 0)
    #     Xpos = observations.get(u'XPos', 0)
    #     Zpos = observations.get(u'ZPos', 0)
    #     obs = np.array(grid)
    #     obs = np.reshape(obs, [16, 16, 1])
    #     obs[(int)(5 + Zpos)][ (int)(10 + Xpos)] = "human"

    #     # for i in range(obs.shape[0]):
    #     #     for j in range(obs.shape[1]):
    #     #         if obs[i,j] ==""
    #     obs[obs == "carpet"] = 0
    #     obs[obs == "sea_lantern"] = 1
    #     obs[obs == "human"] = 3
    #     obs[obs == "fire"] = 4
    #     obs[obs == "emerald_block"] = 5
    #     obs[obs == "beacon"] = 6
    #     obs[obs == "air"] = 7
    #     # print("Here is obs", obs)
    #     return obs


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
    actions = []
    for a in e['action_space']:
        a = str(a).replace(' ', '')
        actions.append(a)
    actions = '+'.join(actions)
    hiddens = [str(h) for h in c['model']['fcnet_hiddens']]
    hiddens = '+'.join(hiddens)
    name = f"{name}_{e['width']}width_{e['millisec_per_tick']}ticks_"\
           f"{e['mission_timeout_ms']}timeout_{e['step_reward']}step_"\
           f"{e['win_reward']}win_{e['lose_reward']}lose_{actions}actions_"\
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
    train_configs = c.get_config('train')

    for train_config in [train_configs[2]]:
        print()
        print("# ------ New Training ------ #")
        train_config = train_config['config']
        env_config = train_config['env_config']

        # Set the name of the training agent
        height, width = env_config['height'], env_config['width']
        train_config['model']['conv_filters'] = [[8, 6, 4],
                                                 [16, 6, 4],
                                                 [32, 6, 4]]

        train_name = get_train_name(
            name=general_config['name'], c=train_config)
        print("Training session name: ", train_name)

        # Create checkpoint directory
        save_freq = general_config['save_freq']
        checkpoint_path = os.path.join(
            general_config['checkpoint_path'], train_name)
        log_path = os.path.join(general_config['log_path'], train_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        # Create the environment
        algo = PPO(env=CustomEnv, config=train_config)

        policy = algo.get_policy()
        print(policy.model.base_model.summary())
        # Train the agent
        train_epochs = int(general_config['train_epochs'])
        start_time = time.time()
        last_eval = 0
        print("#--------- Starting Training--------- #")
        for epoch in range(train_epochs):
            info = algo.train()

            if epoch % save_freq == 0:
                algo.save_checkpoint(checkpoint_path)

                print(f"Checkpoint saved.")
                print(f"{(time.time()-start_time)/60:0.1f} minutes elapsed.")
                # TODO: Also print the average, min, max reward, (and loss??)
                with open(f'{log_path}/epoch{epoch}.pkl', 'wb') as f:
                    pickle.dump(info, f)
                print(
                    f"Checkpoint saved (epoch {epoch} - {(time.time()-start_time)/60:0.1f} minutes elapsed).")
        # Save data for final epoch just to be safe
        algo.save_checkpoint(checkpoint_path)
        print(f"Final Checkpoint saved.")
        print(f"{(time.time()-start_time)/60:0.1f} minutes elapsed.")
        # TODO: Also print the average, min, max reward, (and loss??)
        with open(f'{log_path}/epoch{epoch}.pkl', 'wb') as f:
            pickle.dump(info, f)
        print(f"Final Log saved.")
        print(f"Total time elapsed: {(time.time()-start_time)/60:0.1f} minutes.")


if __name__ == '__main__':
    main()
