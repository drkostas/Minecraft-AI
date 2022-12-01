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
    """ Create a custom OpenAI gym environment (custom MalmoMazeEnv). 
        MalmoMazeEnv Required Args:
        action_space
        step_reward
        win_reward
        lose_reward
        mission_timeout_ms
    """
    xml = Path(config["mission_file"]).read_text()
    env = MalmoMazeEnv(
        xml=xml,
        width=config["width"],
        height=config["height"],
        millisec_per_tick=config["millisec_per_tick"],
        mission_timeout_ms=config['mission_timeout_ms'],
        step_reward=config['step_reward'],
        win_reward=config['win_reward'],
        lose_reward=config['lose_reward'],
        action_space=config['action_space'],
        client_port=config['client_port'],
        time_wait=config['time_wait'],
        max_loop=config['max_loop'])
    return env


def stop_check(trial_id, result):
    return result["episode_reward_mean"] >= 85


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
    print(checkpoint_path, c.tag, run_config)

    tune.register_env(c.tag, create_env)
    ray.init()
    tune.run(run_or_experiment="DQN",
             config=run_config,
             stop=stop_check,
             checkpoint_freq=1,
             checkpoint_at_end=True,
             local_dir=c_general['log_path'])
    ray.shutdown()


if __name__ == '__main__':
    main()
