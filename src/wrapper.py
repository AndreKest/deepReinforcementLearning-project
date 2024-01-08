from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium import spaces

import numpy as np
from gym_environments import normalize_angle


class WrapperCenter(gym.Env):
    """
    This class is used to wrap an environment to adapt the observation space and reward function.
    The observation space is a Box with the distance and the angle to the center of the next gate.
    The reward function gives a reward for getting closer to the next gate and punishment for getting further away.
    The reward function gives a extra reward for passing a gate.
    """
    def __init__(self, env) -> None:
        super(WrapperCenter, self).__init__()
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.pi]),
            high=np.array([np.inf, np.pi]),
        )

        self.last_dist = np.inf

        self.lst_rewards = []
        self.cumulative_reward = 0

        self.reset()

    def reset(self, **kwargs):
        """ Reset the environment """
        obs, _ = self.env.reset()
        self.last_dist = np.inf

        next_gate = self.env.unwrapped.track.get_gates()[self.env.unwrapped.track.get_gate_idx()]
        gate_center = self.env.unwrapped.calc_gate_center(next_gate)
        
        obs = np.array([
            np.linalg.norm(obs[0:2] - gate_center), # distance to the center of the next gate
            normalize_angle(obs[2] - np.arctan2(gate_center[1] - obs[1], gate_center[0] - obs[0])) # angle to the center of the next gate
        ])

        self.cumulative_reward = 0
    
        return obs, {}
    
    def step(self, action):
        """ Step the environment """
        obs, reward, done, truncated, _ = self.env.step(action)

        # Get distance and angle to gate
        try:
            next_gate = self.env.unwrapped.track.get_gates()[self.env.unwrapped.track.get_gate_idx()]
        except IndexError:
            next_gate = self.env.unwrapped.track.get_gates()[-1]
        gate_center = self.env.unwrapped.calc_gate_center(next_gate)
        distance_to_center = np.linalg.norm(obs[0:2] - gate_center)
        angle_to_center = normalize_angle(obs[2] - np.arctan2(gate_center[1] - obs[1], gate_center[0] - obs[0]))

        obs = np.array([
            distance_to_center, # distance to the center of the next gate
            angle_to_center # angle to the center of the next gate
        ])

        ###### Reward Shaping ######

        reward = self.reward_shaping(reward, distance_to_center)

        self.cumulative_reward += reward

        ############

        ###### Check truncate ######
        # b_tmp = not self.env.check_inside_box()
        # truncated = truncated or b_tmp
        ############################
        
        self.last_dist = distance_to_center

        if done or truncated:
            self.lst_rewards.append(self.cumulative_reward)
            self.cumulative_reward = 0

        return obs, reward, done, truncated, {}

    def render(self):
        """ Render the track """
        return self.env.render()
    
    def reward_shaping(self, reward, distance_to_center):
        """
        Function to shape the reward

        Args:
            reward: Reward from the environment
            distance_to_center: Distance to the center of the next gate

        Returns:
            Shaped reward
        """
        r_tmp_gate = self.reward_gate(reward)

        r_tmp_distance = self.reward_distance(reward, distance_to_center)

        r_tmp_box = self.reward_box_boundary()

        reward = r_tmp_gate + r_tmp_distance + r_tmp_box

        return reward

    def reward_box_boundary(self):
        """
        Reward for being inside the box

        Returns:
            Reward for being inside the box 0 or punishment for being outside the box -10
        """
        if self.env.unwrapped.check_inside_box():
            return 0
        else:
            return -10

    def reward_distance(self, reward, distance_to_center):
        """ 
        Calculate the reward 
            - for getting closer to the next gate +1
            - for getting further away from the next gate -1

        Args:
            reward: Reward from the environment
            distance_to_center: Distance to the center of the next gate

        Returns:
            Reward for getting closer to the next gate 1 or punishment for getting further away from the next gate -1

        """
        if reward == 0:
            if distance_to_center < self.last_dist:
                return 1 # get reward for getting closer to the next gate
            else:  
                return -1 # get punishment for getting further away from the next gate
        else:
            return 0

    def reward_gate(self, reward):
        """
        Calculate the reward (how middle the agent drives through the gates)
            - for passing a gate +10

        Args:
            reward: Reward from the environment
        
        Returns:
            Reward for passing a gate 10 if passed middle area of the gate, 2 if passed above or below the middle area of the gate, 0 otherwise

        """
        if reward != 0:
            prev_gate = self.env.unwrapped.track.get_gates()[self.env.unwrapped.track.get_gate_idx()-1]
            gate_center = self.env.unwrapped.calc_gate_center(prev_gate)

            state = self.env.unwrapped.track.state

            if (state[1] > (gate_center[1] - 0.25)) and (state[1] < (gate_center[1] + 0.25)):
                return 10
            elif state[1] > gate_center[1] + 0.25:
                return 2
            elif state[1] < gate_center[1] - 0.25:
                return 2
        else:
            return 0

class WrapperXYTheta(gym.Env):
    """
    This class is used to wrap an environment to adapt the observation space and reward function.
    The observation space is a Box with the x,y coordiante and the angle of the agent.
    The reward function gives a reward for getting closer to the next gate and punishment for getting further away.
    The reward function gives a extra reward for passing a gate.
    """
    def __init__(self, env) -> None:
        super(WrapperXYTheta, self).__init__()
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi]),
            high=np.array([np.inf, np.inf, np.pi]),
        )

        self.last_dist = np.inf

        self.lst_rewards = []
        self.cumulative_reward = 0

        self.reset()

    def reset(self, **kwargs):
        obs, _ = self.env.reset()
        self.last_dist = np.inf
        obs = np.array([
            obs[0], # x coordinate
            obs[1], # y coordinate
            obs[2] # angle
        ])

        self.cumulative_reward = 0
        
        return obs, {}
    
    def step(self, action):
        obs, reward, done, truncated, _ = self.env.step(action)

        try:
            next_gate = self.env.unwrapped.track.get_gates()[self.env.unwrapped.track.get_gate_idx()]
        except IndexError:
            next_gate = self.env.unwrapped.track.get_gates()[-1]

        gate_center = self.env.unwrapped.calc_gate_center(next_gate)
        distance_to_center = np.linalg.norm(obs[0:2] - gate_center)

        obs = np.array([
            obs[0], # x coordinate
            obs[1], # y coordinate
            obs[2] # angle
        ])

        ###### Reward Shaping ######

        reward = self.reward_shaping(reward, distance_to_center)

        self.cumulative_reward += reward

        ############################

        ###### Check truncate ######
        # b_tmp = not self.env.check_inside_box()
        # truncated = truncated or b_tmp
        ############################

        if done or truncated:
            self.lst_rewards.append(self.cumulative_reward)
            self.cumulative_reward = 0

        self.last_dist = distance_to_center
        return obs, reward, done, truncated, {}

    def render(self):
        return self.env.render()
    
    def reward_shaping(self, reward, distance_to_center):
        """
        Function to shape the reward

        Args:
            reward: Reward from the environment
            distance_to_center: Distance to the center of the next gate

        Returns:
            Shaped reward
        """
        r_tmp_gate = self.reward_gate(reward)

        r_tmp_distance = self.reward_distance(reward, distance_to_center)

        r_tmp_box = self.reward_box_boundary()

        reward = r_tmp_gate + r_tmp_distance + r_tmp_box

        return reward

    def reward_box_boundary(self):
        """
        Reward for being inside the box

        Returns:
            Reward for being inside the box 0 or punishment for being outside the box -10
        """
        if self.env.unwrapped.check_inside_box():
            return 0
        else:
            return -10

    def reward_distance(self, reward, distance_to_center):
        """ 
        Calculate the reward 
            - for getting closer to the next gate +1
            - for getting further away from the next gate -1

        Args:
            reward: Reward from the environment
            distance_to_center: Distance to the center of the next gate

        Returns:
            Reward for getting closer to the next gate 1 or punishment for getting further away from the next gate -1

        """
        if reward == 0:
            if distance_to_center < self.last_dist:
                return 1 # get reward for getting closer to the next gate
            else:  
                return -1 # get punishment for getting further away from the next gate
        else:
            return 0

    def reward_gate(self, reward):
        """
        Calculate the reward (how middle the agent drives through the gates)
            - for passing a gate +10

        Args:
            reward: Reward from the environment
        
        Returns:
            Reward for passing a gate 10 if passed middle area of the gate, 2 if passed above or below the middle area of the gate, 0 otherwise

        """
        if reward != 0:
            prev_gate = self.env.unwrapped.track.get_gates()[self.env.unwrapped.track.get_gate_idx()-1]
            gate_center = self.env.unwrapped.calc_gate_center(prev_gate)

            state = self.env.unwrapped.track.state

            if (state[1] > (gate_center[1] - 0.25)) and (state[1] < (gate_center[1] + 0.25)):
                return 10
            elif state[1] > gate_center[1] + 0.25:
                return 2
            elif state[1] < gate_center[1] - 0.25:
                return 2
        else:
            return 0    


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import pandas as pd

    import gymnasium as gym

    import stable_baselines3 as sb3
    from stable_baselines3.common.monitor import Monitor

    from wrapper import WrapperXYTheta as _wrapper
    from gym_environments import GymRaceEnvContinous

    from race import CurvyRace

    
    # Create environment
    
    env = GymRaceEnvContinous(track=CurvyRace())
    monitor = Monitor(env, filename=None, allow_early_resets=True)
    wrapped = _wrapper(monitor)

    model = sb3.SAC("MlpPolicy", wrapped, verbose=0, learning_rate=0.001, buffer_size=100000, tau=0.005, gamma=0.8, batch_size=200, learning_starts=1000)

    model.learn(total_timesteps=300000, progress_bar=True)

    plt.figure()
    plt.plot(monitor.get_episode_rewards())
    plt.xlabel("Episode")
    plt.ylabel("Gates")
    plt.savefig("./gates.png")

    plt.figure()
    print(len(wrapped.lst_rewards))
    plt.plot(wrapped.lst_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.savefig("./cum_reward.png")

    model.save("./model.zip")

    print("Done")
