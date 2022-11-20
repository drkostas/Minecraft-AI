import argparse
from pathlib import Path

from yaml_config_wrapper import Configuration
from RLcraft import MalmoMazeEnv


def create_env(config):
    """ Create a custom OpenAI gym environment (custom MalmoMazeEnv). """
    xml = Path(config["mission_file"]).read_text()
    env = MalmoMazeEnv(
        xml=xml,
        width=config["width"],
        height=config["height"],
        millisec_per_tick=config["millisec_per_tick"])
    return env


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
    run_config = c_tuner['config']
    c_general = c_general['config']

    run = True
    while run:
        # Generate a seed for maze 
        print("Generating new seed ...")
        maze_seed = random.randint(1, 9999)
        print("Loading environment ...")
        xml = Path(run_config["env_config"]["mission_file"]).read_text()
        env = MalmoMazeEnv(
            xml=xml,
            width=800,
            height=600,
            millisec_per_tick=50,
            mazeseed=maze_seed,
            step_reward=c_general['step_reward'],
            win_reward=c_general['win_reward'],
            lose_reward=c_general['lose_reward'],
            action_space=c_general['action_space'],
            client_port=c_general['client_port'],
            time_wait=c_general['time_wait'],
            max_loop=c_general['max_loop'])
        print("Resetting environment ...")
        env.reset()
        print("The world is loaded.")
        print("Press Enter and F5 key in Minecraft to show third-person view.")
        input("Press any key to start simulation")
        done = False
        while not done:
            action = env.action_space.sample()
            # Actions: 0 -> move (frwd), 1 -> right, 2 -> left
            obs, reward, done, info = env.step(action)
            # obs is a numpy array of shape (height, width, 3)
            env.render()
        user_choice = input("Enter 'N' to exit, 'Y' to run new episode [Y/n]: ").lower()
        if user_choice.lower() in ['n', 'no']:
            run = False
        env.close()
    
    print("Done.")


if __name__ == '__main__':
    main()
