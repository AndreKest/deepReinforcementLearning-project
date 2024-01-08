from race import RaceEnv

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt
from IPython import display

import io


def normalize_angle(angle):
    """ Normalize an angle to [-pi, pi] """
    return (angle + np.pi) % (2 * np.pi) - np.pi

class GymRaceEnvContinous(gym.Env):
    def __init__(self, track, render_mode="human"):
        super(GymRaceEnvContinous, self).__init__()
        self.track = track
        self.max_steps = self.track.get_max_steps()
        limits = self.track.get_action_limits()
        self.render_mode = render_mode
        self.action_space = spaces.Box(low=np.array([0, -limits[1]]), high=np.array([limits[0], limits[1]]), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array([-np.inf for _ in range(3)]), high=np.array([np.inf for _ in range(3)]), dtype=np.float64)

        self.current_step = 0
        self.reset()

        self.box_boundaries = self.get_box_boundaries()

    def reset(self, **kwargs):
        """ Reset the environment """
        self.current_step = 0
        obs = self.track.reset()
        
        obs = np.array([
            obs[0], # x
            obs[1], # y 
            normalize_angle(obs[2]), # phi
            ])
        
        return obs, {}
    
    def step(self, action):
        """ Step the environment """
        self.current_step += 1
        obs, reward, done = self.track.step(action)
        
        obs = np.array([
            obs[0], # x
            obs[1], # y
            normalize_angle(obs[2]), # phi
        ])

        truncated = self.current_step >= self.max_steps
        return obs, reward, done, truncated, {}
    
    def render(self):
        """
        Render the track
        """
        if self.render_mode == "human":
            display.display(plt.gcf())
            display.clear_output(wait=True)
            self.track.plot()
            plt.show()
        elif self.render_mode == "rgb_array":
            # return a numpy array with rendered image
            display.display(plt.gcf())
            display.clear_output(wait=True)
            self.track.plot()
            # plt.show()
            fig = plt.gcf()

            with io.BytesIO() as buffer:
                fig.savefig(buffer, format='raw')
                buffer.seek(0)
                data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            return data.reshape(int(h), int(w), -1)
        
    def calc_gate_center(self, gate):
        """
        Calculate the center of a gate 

        Args:
            gate: Gate to calculate the center of
        
        Returns:
            Array with the center of the gate
            
        """
        return np.array([(gate[0][0] + gate[1][0]) / 2, (gate[0][1] + gate[1][1]) / 2])
    

    def get_box_boundaries(self):
        """
        Get the boundaries of the box that contains all gates

        Returns:
            Array with the boundaries of the box
        """
        gates = self.track.get_gates()
        # [[2, -1],  [2, 1]],
        # get maximum and minimum x and y values of all gates
        min_x = min([gate[0][0] for gate in gates]) - 3
        max_x = max([gate[1][0] for gate in gates]) + 3
        min_y = min([gate[0][1] for gate in gates]) - 3
        max_y = max([gate[1][1] for gate in gates]) + 3

        return np.array([min_x, min_y, max_x, max_y])
    
    def check_inside_box(self):
        """
        Check if the position of the agent is inside the box.

        Returns:
            True if the position is inside the box, False otherwise
        """
        
        x, y = self.track.state[0:2]

        # check if the position is within the boundaries of the track
        if ((x > self.box_boundaries[0]) and (x < self.box_boundaries[2])) and ((y > self.box_boundaries[1]) and (y < self.box_boundaries[3])):
            return True
        else: 
            return False

