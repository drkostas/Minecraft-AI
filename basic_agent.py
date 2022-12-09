import argparse
from pathlib import Path
import random

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
                               help="number of gpus to use for trianing")
    args = parser.parse_args()
    return args


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
    run_config = c_tuner['config']['env_config']
    c_general = c_general['config']

    run = True
    while run:
        # Generate a seed for maze
        print("Generating new seed ...")
        maze_seed = random.randint(1, 9999)
        print("Loading environment ...")
        xml = Path(run_config["mission_file"]).read_text()
        env = MalmoMazeEnv(
            xml=xml,
            width=run_config["width"],
            height=run_config["height"],
            millisec_per_tick=run_config["millisec_per_tick"],
            mission_timeout_ms=run_config['mission_timeout_ms'],
            step_reward=run_config['step_reward'],
            win_reward=run_config['win_reward'],
            lose_reward=run_config['lose_reward'],
            action_space=run_config['action_space'],
            client_port=run_config['client_port'],
            time_wait=run_config['time_wait'],
            max_loop=run_config['max_loop'])
        print("Resetting environment ...")
        print(env.reset())
        print("The world is loaded.")
        print("Press Enter and F5 key in Minecraft to show third-person view.")
        input("Press any key to start simulation")
        done = False
        while not done:
            action = env.action_space.sample()
            # Actions: 0 -> move (frwd), 1 -> right, 2 -> left
            obs, reward, done, info = env.step(action)
            done = True
            print(len(obs))
            # obs is a numpy array of shape (height, width, 3)
            env.render()
        user_choice = input(
            "Enter 'N' to exit, 'Y' to run new episode [Y/n]: ").lower()
        if user_choice.lower() in ['n', 'no']:
            run = False
        env.close()

    print("Done.")


if __name__ == '__main__':
    main()
