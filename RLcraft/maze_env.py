import gym
import malmo.MalmoPython as MalmoPython
import random
import time
import numpy as np
from pathlib import Path


class AgentActionSpace(gym.spaces.Discrete):
    def __init__(self, action_space):
        actions = list(action_space)
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
        action_space,
        step_reward,
        win_reward,
        lose_reward,
        mission_timeout_ms,
        millisec_per_tick = 50,
        mazeseed = "random",
        enable_action_history=False,
        client_port=5000,
        time_wait=0.1,
        max_loop=50,
        name='test1',
        client_address='127.0.0.1',
        max_xml=159):
        # Set up gym.Env
        super(MalmoMazeEnv, self).__init__()
        # Initialize self variables
        if '*' in xml:
            num = random.randint(0, max_xml)
            xml = xml.replace("*", str(num))
        self.xml = Path(xml).read_text()
        self.height = height
        self.width = width
        self.shape = (self.height, self.width, 3)
        self.millisec_per_tick = millisec_per_tick
        self.mazeseed = mazeseed
        self.enable_action_history = enable_action_history
        self.mission_timeout_ms = mission_timeout_ms
        self.step_reward = step_reward
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.time_wait = time_wait
        self.max_loop = max_loop
        self.name=name
        # load action space
        self.action_space = AgentActionSpace(action_space)
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
        client_info = MalmoPython.ClientInfo(client_address, client_port)
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
            PLACEHOLDER_MAZESEED=self.mazeseed,
            PLACEHOLDER_STEP_REWARD=self.step_reward,
            PLACEHOLDER_WIN_REWARD=self.win_reward,
            PLACEHOLDER_LOSE_REWARD=self.lose_reward,
            PLACEHOLDER_MISSION_TIMEOUT_MS=self.mission_timeout_ms)
        my_mission = MalmoPython.MissionSpec(xml,True)
        # Start mission
        self.agent_host.startMission(my_mission,
            self.pool,
            self.my_mission_record,
            0,
            self.name)
        # Wait till mission begins
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(self.time_wait * self.millisec_per_tick / 20)
            world_state = self.agent_host.getWorldState()
        # Get reward, done, and frame
        frame, _, _, _ = self._process_state(False)
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
        """ Available sendCommand() commands:
            move: walks forwards/backward
              0 -> Nothing, 1 -> Forward, -1 -> Backwards
            strafe: walks left/right
              0 -> Nothing, 1 -> Right, -1 -> Left
            turn: turns the camera left/right without moving
              0 -> Nothing, 1 -> Right, -1 -> Left """
              
        self.agent_host.sendCommand(self.action_space[action])

        # Get reward, done, and frame
        frame, reward, done, world_state = self._process_state()
        if reward is None:
            reward = self.step_reward
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
        return self.last_obs, reward, done, world_state

    def close(self):
        self.agent_host.sendCommand("quit")
        _, _, _, _ = self._process_state()
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
            time.sleep(self.time_wait * self.millisec_per_tick / 20)
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
            if loop > self.max_loop:
                reward = None
                break;
        return frame, reward, done, world_state

    def _comsume_state(self):
        reward_flag = True
        reward = 0
        frame = None
        while True:
            # get next world state
            time.sleep(self.time_wait * self.millisec_per_tick / 5)
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