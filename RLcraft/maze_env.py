import gym
import malmo.MalmoPython as MalmoPython
import random
import time
import numpy as np
from enum import Enum

CLIENT_PORT = 9000                  # malmo port
TIME_WAIT = 0.05                    # time to wait for retreiving world state (when MsPerTick=20)
MAX_LOOP = 50                       # wait till TIME_WAIT * MAX_LOOP seconds for each action

class AgentActionSpace(gym.spaces.Discrete):
    def __init__(self):
        actions = []
        actions.append("move")
        actions.append("right")
        actions.append("left")
        self.actions = actions
        gym.spaces.Discrete.__init__(self, len(self.actions))

    def sample(self):
        return random.randint(1, len(self.actions)) - 1

    def __getitem__(self, action):
        return self.actions[action]

    def __len__(self):
        return len(self.actions)

class MalmoMazeEnv(gym.Env):
    """
    A class implementing OpenAI gym environment to
    run Project Malmo 0.36.0 Python API for solving
    maze.
    init parameters
    ---------------
    xml : str (required)
        Mission setting (XML string) used in Project Malmo.
    width : int (required)
        Frame width for agent
    height : int (required)
        Frame height for agent.
    millisec_per_tick : int (optional)
        Millisec between each ticks. Set lower value to speed up.
        Default is 50 (which is normal Minecraft game speed).
    mazeseed : str/int (optional)
        Seed value for maze. To create the same maze, set same value of seed.
        Default is "random".
    enable_action_history : bool (optional)
        Set if the action is recorded in "action_history" attribute.
        Default is False.
    """
    def __init__(self,
        xml,
        width,
        height,
        millisec_per_tick = 50,
        mazeseed = "random",
        enable_action_history=False):
        # Set up gym.Env
        super(MalmoMazeEnv, self).__init__()
        # Initialize self variables
        self.xml = xml
        self.height = height
        self.width = width
        self.shape = (self.height, self.width, 3)
        self.millisec_per_tick = millisec_per_tick
        self.mazeseed = mazeseed
        self.enable_action_history = enable_action_history
        # none:0, move:1, right:2, left:3
        self.action_space = AgentActionSpace()
        # frame
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.shape, dtype=np.float32)
        # Create AgentHost
        self.agent_host = MalmoPython.AgentHost()
        # Create MissionRecordSpec
        self.my_mission_record = MalmoPython.MissionRecordSpec()
        self.my_mission_record.recordRewards()
        self.my_mission_record.recordObservations()
        # Create ClientPool
        self.pool = MalmoPython.ClientPool()
        client_info = MalmoPython.ClientInfo('127.0.0.1', CLIENT_PORT)
        self.pool.add(client_info)

    """
    Public methods
    """

    def reset(self):
        # Create MissionSpec
        xml = self.xml
        xml = xml.format(
            PLACEHOLDER_MSPERTICK=self.millisec_per_tick,
            PLACEHOLDER_WIDTH=self.width,
            PLACEHOLDER_HEIGHT=self.height,
            PLACEHOLDER_MAZESEED=self.mazeseed)
        my_mission = MalmoPython.MissionSpec(xml,True)
        # Start mission
        self.agent_host.startMission(my_mission,
            self.pool,
            self.my_mission_record,
            0,
            'test1')
        # Wait till mission begins
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(TIME_WAIT * self.millisec_per_tick / 20)
            world_state = self.agent_host.getWorldState()
        # Get reward, done, and frame
        frame, _, _ = self._process_state(False)
        if frame is None:
            self.last_obs = np.zeros(self.shape, dtype=np.float32)
        else:
            self.last_obs = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(self.shape)
        # Record action history
        if self.enable_action_history:
            self.action_history = []
        return self.last_obs

    def render(self, mode=None):
        if self.last_obs is None:
            self.last_obs = np.zeros(self.shape, dtype=np.float32)
        return np.flipud(self.last_obs)

    def step(self, action):
        # Take corresponding actions
        """ none:0, move:1, right:2, left:3 """
        if self.action_space[action] == "move":
            self.agent_host.sendCommand("move 1")
        elif self.action_space[action] == "right":
            self.agent_host.sendCommand("turn 1")
        elif self.action_space[action] == "left":
            self.agent_host.sendCommand("turn -1")

        # Get reward, done, and frame
        frame, reward, done = self._process_state()
        if reward is None:
            reward = 0
        # Clean up
        if done:
            frame2, reward2 = self._comsume_state()
            if frame2 is not None:
                frame = frame2
            reward = reward + reward2
        # Return observations
        if frame is None:
            self.last_obs = np.zeros(self.shape, dtype=np.uint8)
        else:
            self.last_obs = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(self.shape)
        # Record action history
        if self.enable_action_history:
            self.action_history.append(action)
        return self.last_obs, reward, done, {}

    def close(self):
        self.agent_host.sendCommand("quit")
        _, _, _ = self._process_state()
        _, _ = self._comsume_state()

    """
    Internal methods
    """

    # Extract frames, rewards, done_flag
    def _process_state(self, get_reward=True):
        reward_flag = False
        reward = 0
        frame_flag = False
        frame = None
        done = False
        loop = 0
        while True:
            # get world state
            time.sleep(TIME_WAIT * self.millisec_per_tick / 20)
            world_state = self.agent_host.getWorldState()
            # reward (loop till command's rewards are all retrieved)
            if (not reward_flag) and (world_state.number_of_rewards_since_last_state > 0):
                reward_flag = True;
                reward = reward + world_state.rewards[-1].getValue()
            # frame
            if world_state.number_of_video_frames_since_last_state > 0:
                frame = world_state.video_frames[-1]
                frame_flag = True
            # done flag
            done = not world_state.is_mission_running
            # (Do not quit before comsuming)
            # if done:
            #     break;
            # exit loop when extraction is completed
            if get_reward and reward_flag and frame_flag:
                break;
            elif (not get_reward) and frame_flag:
                break;
            # exit when MAX_LOOP exceeds
            loop = loop + 1
            if loop > MAX_LOOP:
                reward = None
                break;
        return frame, reward, done

    def _comsume_state(self):
        reward_flag = True
        reward = 0
        frame = None
        loop = 0
        while True:
            # get next world state
            time.sleep(TIME_WAIT * self.millisec_per_tick / 5)
            world_state = self.agent_host.getWorldState()
            # reward (loop till command's rewards are all retrieved)
            if reward_flag and not (world_state.number_of_rewards_since_last_state > 0):
                reward_flag = False;
            if reward_flag:
                reward = reward + world_state.rewards[-1].getValue()
            # frame
            if world_state.number_of_video_frames_since_last_state > 0:
                frame = world_state.video_frames[-1]
            if not reward_flag:
                break;
        return frame, reward