import argparse
from pathlib import Path
import random
import json
from typing import Dict
import time
from yaml_config_wrapper import Configuration
from RLcraft import MalmoMazeEnv


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
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument("--num_gpus",
                               type=int,
                               required=False,
                               default=0,
                               help="number of gpus to use for training")
    args = parser.parse_args()
    return args


def create_env(config: Dict):
    """ Create a custom OpenAI gym environment (custom MalmoMazeEnv).
    """
    maze_seed = random.randint(1, 9999)
    env = MalmoMazeEnv(
            mazeseed=maze_seed,
            xml=config["xml"],
            width=config["width"],
            height=config["height"],
            millisec_per_tick=config['millisec_per_tick'],
            mission_timeout_ms=config['mission_timeout_ms'],
            step_reward=config['step_reward'],
            win_reward=config['win_reward'],
            lose_reward=config['lose_reward'],
            action_space=config['action_space'],
            client_port=config['client_port'],
            time_wait=config['time_wait'],
            max_loop=config['max_loop'])
    return env


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
    obs['rgb'] = np_obs
    obs['floor'] = floor_data
    obs['time'] = time_data
    obs['xpos'] = xpos_data
    obs['ypos'] = ypos_data
    obs['zpos'] = zpos_data
    obs['yaw'] = yaw_data
    obs['hp'] = hp_data
    return obs


def main():
    """ Run a the game with a random agent. """
    # Read command arguments
    args = get_args()
    # Load YML config file
    c = Configuration(config_src=args.config_file)
    # Load configs from config class
    general_config = c.get_config('general')
    train_config = c.get_config('train')[0]['config']
    env_config = train_config['env_config']

    run = True
    while run:
        print("Loading environment ...")
        env = create_env(env_config)
        print("Resetting environment ...")
        print(env.reset())
        print("The world is loaded.")
        print("Press Enter and F5 key in Minecraft to show third-person view.")
        input("Press any key to start simulation")
        done = False
        while not done:
            action = env.action_space.sample()
            # Actions: 0 -> move (frwd), 1 -> right, 2 -> left
            np_obs, reward, done, info = env.step(action)
            observations = process_obs(np_obs, info)
            done = True
            print(observations)
            env.render()
        user_choice = input(
            "Enter 'N' to exit, 'Y' to run new episode [Y/n]: ").lower()
        if user_choice.lower() in ['n', 'no']:
            run = False
        else:
            time.sleep(5)
        env.close()

    print("Done.")


if __name__ == '__main__':
    main()
