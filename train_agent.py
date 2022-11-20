import ray
import ray.tune as tune
from ray.rllib import rollout
from ray.tune.registry import get_trainable_cls
import gym
from functools import partial
import argparse
from pathlib import Path
import random
import time
import os

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
    # log_path = c_general['log_path']
    checkpoint_path = os.path.join(c_general['checkpoint_path'], c.tag)
    # terminal_reward = int(c_general['terminal_reward'])

    run = True
    while run:

        # Generate a seed for maze
        print("Generating new seed ...")
        maze_seed = random.randint(1, 9999)

        # Run agent with trained checkpoint
        print("An agent is running ...")
        tune.register_env(c.tag, create_env)
        cls = get_trainable_cls("DQN")
        agent = cls(env=c.tag, config=run_config)
        # agent.optimizer.stop()
        if os.path.exists(checkpoint_path):
            agent.restore(checkpoint_path)
        env1 = agent.workers.local_worker().env
        obs = env1.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.compute_action(obs)
            obs, reward, done, info = env1.step(action)
            total_reward += reward
        env1.close()
        agent.stop()
        print("Done with reward ", total_reward)

        #
        # Simulate same result with wide screen
        #

        xml = Path(args.mission_path).read_text()
        env2 = MalmoMazeEnv(
            xml=xml,
            width=800,
            height=600,
            millisec_per_tick=50,
            mazeseed=maze_seed)
        env2.reset()
        print("The world is loaded.")
        print("Press Enter and F5 key in Minecraft to show third-person view.")
        input("Enter keyboard to start simulation")
        for action in env1.action_history:
            time.sleep(0.5)
            obs, reward, done, info = env2.step(action)
        user_choice = input("Enter 'N' to exit [Y/n]: ").lower()
        if user_choice in ['n', 'no']:
            run = False
        env2.close()

    ray.shutdown()


if __name__ == '__main__':
    main()
