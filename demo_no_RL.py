import argparse
from pathlib import Path
import random
import json

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
    xp_data = info_obs['XP']
    obs = {}
    obs['rgb'] = np_obs
    obs['floor'] = floor_data
    obs['time'] = time_data
    obs['xpos'] = xpos_data
    obs['ypos'] = ypos_data
    obs['zpos'] = zpos_data
    obs['yaw'] = yaw_data
    obs['xp'] = xp_data
    return obs

def main():
    """ Run a the game with a random agent. """
    # Read command arguments
    args = get_args()
    # Load YML config file
    c = Configuration(config_src=args.config_file)
    # Load configs from config class
    c_general = c.get_config('general')[0]
    c_tuner = c.get_config('tuner')[0]
    # Load the values from the config
    run_config = c_tuner['config']
    env_config = run_config['env_config']
    c_general = c_general['config']

    run = True
    while run:
        # Generate a seed for maze
        print("Generating new seed ...")
        maze_seed = random.randint(1, 9999)
        print("Loading environment ...")
        env = MalmoMazeEnv(
            width=env_config["width"],
            height=env_config["height"],
            mazeseed=maze_seed,
            xml=env_config["mission_file"],
            millisec_per_tick=env_config['millisec_per_tick'],
            max_loop=c_general['max_loop'],
            mission_timeout_ms=c_general['mission_timeout_ms'],
            step_reward=c_general['step_reward'],
            win_reward=c_general['win_reward'],
            lose_reward=c_general['lose_reward'],
            action_space=c_general['action_space'],
            client_port=env_config['client_port'],
            time_wait=c_general['time_wait'])
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
        env.close()

    print("Done.")


if __name__ == '__main__':
    main()
