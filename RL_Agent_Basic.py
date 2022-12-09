import argparse
from pathlib import Path
import random
import gym
from yaml_config_wrapper import Configuration
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from RLcraft import MalmoMazeEnv
import numpy as np
#from ray.rllib.agents.ppo import PPO

from ray.rllib.algorithms.ppo import PPO


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



    maze_seed = random.randint(1, 9999)

    # xml = Path(run_config["env_config"]["mission_file"]).read_text()
    #env = MalmoMazeEnv(
    env_config = {
        "xml":run_config["env_config"]["mission_file"],
        "width":320,
        "height":240,
        "millisec_per_tick":50,
        "mazeseed":maze_seed,
        "mission_timeout_ms":c_general['mission_timeout_ms'],
        "step_reward":c_general['step_reward'],
        "win_reward":c_general['win_reward'],
        "lose_reward":c_general['lose_reward'],
        "action_space":c_general['action_space'],
        "client_port":c_general['client_port'],
        "time_wait":c_general['time_wait'],
        "max_loop":c_general['max_loop']
    }
    config = {

    "env_config": env_config,
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 1,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "tf",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [64, 64],
        "conv_filters": None,
        "fcnet_activation": "relu",
        "grayscale": True,
        "conv_filters":get_filter_config((env_config["height"],env_config["width"],3))

    },
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    "disable_env_checking":True,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
    }


    print("Resetting environment ...")

    algo = PPO(env=CustEnv, config=config)
    for _ in range(5):
        (algo.train())

    """
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
            mission_timeout_ms=c_general['mission_timeout_ms'],
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

    print("Done.")"""


class CustEnv(gym.Env):
    def __init__(self, config):
        self.env = MalmoMazeEnv(
            xml=config["xml"],
            width=config["width"],
            height=config["height"],
            millisec_per_tick=config["millisec_per_tick"],
            mazeseed=config["mazeseed"],
            mission_timeout_ms=config['mission_timeout_ms'],
            step_reward=config['step_reward'],
            win_reward=config['win_reward'],
            lose_reward=config['lose_reward'],
            action_space=config['action_space'],
            client_port=config['client_port'],
            time_wait=config['time_wait'],
            max_loop=config['max_loop'])
        self.observation_space = gym.spaces.Box(high=355,low=0, shape=(config["height"],config["width"],3), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.env.action_space))
        print(self.action_space)
    def reset(self):
        x = self.env.reset()
        return x
    def step(self, action):
        print(action)
        x = self.env.step(action)
        #for i in x[3].rewards:
        #    print(i)

        info = {
            #"obs":x[3].observations,
            #"rewards":x[3].rewards,
            #"frames":x[3].number_of_video_frames_since_last_state,
            "rewards":x[3].rewards[0].getValue()
        }


        return x[0],x[1],x[2],info
if __name__ == '__main__':
    main()
