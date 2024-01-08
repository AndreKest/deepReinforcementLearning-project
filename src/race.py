import matplotlib.pyplot as plt
import pygame
import numpy as np

DT = 0.2
ROBOT_RADIUS = 0.3      # radius of the vehicle (for plotting purpose only)
VEL_TRANS_LIMIT = 4     # velocity limit [m/s]   0...4
VEL_ROT_LIMIT = 2       # velocity limit [rad/s] -2...+2

class RaceEnv():
    def __init__(self):
        self.gate_idx = 0
        self.step_idx = 0
        self.reset()


    def get_action_dim(self):
        # return dimension of action vector (velocity / rotational velocity)
        return 2


    def get_observation_dim(self):
        # return dimension of state/observation vector (x/y/phi)
        return 3


    def get_action_limits(self):
        # return upper/lower bounds of action vector elements. Note that negative velocities are not allowed
        return (VEL_TRANS_LIMIT, VEL_ROT_LIMIT)


    def get_gates(self):
        # return list of all gates along the track
        return self.gates


    def get_gate_idx(self):
        # return index of the next gate that must be passed
        return self.gate_idx


    def get_step(self):
        # return the current numbre of steps
        return self.step_idx


    def get_max_steps(self):
        # return the maximum number of allowed steps per episode
        return self.max_steps


    def reset(self):
        self.gate_idx = 0
        self.step_idx = 0
        self.last_state = np.random.randn(self.get_observation_dim())*0.1
        self.state = np.random.randn(self.get_observation_dim())*0.1
        return np.copy(self.state)


    def step(self, action):
        self.step_idx += 1
        self._calc_next_state(action)
        obs = np.copy(self.state)
        reward = self._calc_reward()
        done = self._calc_done()
        return obs, reward, done


    def _calc_next_state(self, action):
        # calculate next state
        # early out if the maximum number of steps has been exceeded
        if self.step_idx > self.max_steps: return self.state

        # store last state
        self.last_state = np.copy(self.state)

        vel_trans, vel_rot = action

        # limit translational velocity
        vel_trans = max(vel_trans, 0)
        vel_trans = min(vel_trans, +VEL_TRANS_LIMIT)

        # limit rotational velocity
        vel_rot = max(vel_rot, -VEL_ROT_LIMIT)
        vel_rot = min(vel_rot, +VEL_ROT_LIMIT)

        # update vehicle position
        self.state[0] += DT * np.cos(self.state[2]) * vel_trans
        self.state[1] += DT * np.sin(self.state[2]) * vel_trans
        self.state[2] += DT * vel_rot


    def _calc_reward(self):
        # calculate reward
        # early out if the maximum number of steps has been exceeded
        if self.step_idx > self.max_steps: return 0
        # early out if the last gate has been passed
        if self.gate_idx >= len(self.gates): return 0

        if self._do_intersect(self.last_state[0:2], self.state[0:2], self.gates[self.gate_idx][0], self.gates[self.gate_idx][1]):
            self.gate_idx += 1
            return self.gate_reward + self.step_reward

        return self.step_reward


    def _calc_done(self):
        # calculate done flag
        # early out if the maximum number of steps has been exceeded
        if self.step_idx > self.max_steps: return True
        # early out if the last gate has been passed
        if self.gate_idx >= len(self.gates): return True

        # update gate index
        if self._do_intersect(self.last_state[0:2], self.state[0:2], self.gates[self.gate_idx][0], self.gates[self.gate_idx][1]):
            self.gate_idx += 1

        return False


    def _orientation(self, p, q, r):
        # 0 -> p, q and r are collinear
        # 1 -> clockwise
        # 2 -> counterclockwise
        val = (q[1]-p[1])*(r[0]-q[0]) - (q[0]-p[0])*(r[1]-q[1])
        if val == 0: return 0
        return 1 if val > 0 else 2


    def _do_intersect(self, p1, q1, p2, q2):
        # return true if line segments p1q1 and p2q2 intersect
        # adapted from https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
        o1 = self._orientation(p1, q1, p2)
        o2 = self._orientation(p1, q1, q2)
        o3 = self._orientation(p2, q2, p1)
        o4 = self._orientation(p2, q2, q1)

        # general case, does not consider the collinear case
        if o1 != o2 and o3 != o4: return True

        return False


    def plot(self):
        plt.figure(1)
        plt.clf()

        # plot robot
        c = plt.Circle((self.state[0], self.state[1]), ROBOT_RADIUS, facecolor='w', edgecolor='k')
        plt.gca().add_patch(c)
        # plot line indicating the robot direction
        plt.plot([self.state[0], self.state[0] + ROBOT_RADIUS * np.cos(self.state[2])],
                 [self.state[1], self.state[1] + ROBOT_RADIUS * np.sin(self.state[2])], 'k')

        # plot gates
        for gate in self.gates:
            plt.plot([gate[0][0], gate[1][0]], [gate[0][1], gate[1][1]], 'k')

        # highlight next gate
        if self.gate_idx < len(self.gates):
            highl = self.gates[self.gate_idx]
            plt.plot([highl[0][0], highl[1][0]], [highl[0][1], highl[1][1]], 'r', linewidth="2")

        plt.gca().axis('equal')
        plt.pause(0.001)  # pause a bit so that plots are updated


class StraightRace(RaceEnv):
    def __init__(self):
        self.step_reward = 0
        self.gate_reward = 1
        self.max_steps = 100
        self.gates = [[[5, -1],  [5, +1]],
                     [[10, -1], [10, +1]],
                     [[15, -1], [15, +1]],
                     [[20, -1], [20, +1]],
                     [[25, -1], [25, +1]],
                     [[30, -1], [30, +1]],
                     [[35, -1], [35, +1]],
                     [[40, -1], [40, +1]]]
        RaceEnv.__init__(self)


class CurvyRace(RaceEnv):
    def __init__(self):
        self.step_reward = 0
        self.gate_reward = 1
        self.max_steps = 100
        self.gates = [[[2, -1],  [2, 1]],
                     [[4, -1],  [4, 1]],
                     [[6, 0], [6, 2]],
                     [[8, 0.5], [8, 2.5]],
                     [[10, 0], [10, 2]],
                     [[12, -1], [12, 1]],
                     [[14, -2], [14, 0]],
                     [[16, -2.5], [16, -0.5]],
                     [[18, -2], [18, 0]],
                     [[20, -1], [20, 1]],
                     [[22, 0], [22, 2]],
                     [[24, 0.5], [24, 2.5]],
                     [[26, 0], [26, 2]],
                     [[28, -1], [28, 1]],
                     [[30, -2], [30, 0]],
                     [[32, -2.5], [32, -0.5]]];
        RaceEnv.__init__(self)


class CurvyRaceSparse(RaceEnv):
    def __init__(self):
        self.step_reward = 0
        self.gate_reward = 1
        self.max_steps = 100
        self.gates = [[[2, -1],  [2, 1]],
                     [[8, 0.5], [8, 2.5]],
                     [[16, -2.5], [16, -0.5]],
                     [[24, 0.5], [24, 2.5]],
                     [[32, -2.5], [32, -0.5]]];
        RaceEnv.__init__(self)


class Parcour(RaceEnv):
    def __init__(self):
        self.step_reward = 0
        self.gate_reward = 1
        self.max_steps = 1000
        self.gates = [[[2,1], [2,-1]],
                     [[4,1], [4,-1]],
                     [[6,1], [7,-1]],
                     [[7,2], [9,1]],
                     [[7,4], [10,4]],
                     [[7,6], [9,7]],
                     [[6,8], [8,9]],
                     [[5,10], [7,11]],
                     [[4,13], [7,13]],
                     [[5,16], [7,15]],
                     [[7,18], [8,16]],
                     [[10,18], [10,16]],
                     [[12,18], [12,16]],
                     [[14,18], [14,16]],
                     [[17,18], [16,16]],
                     [[19,16], [17,15]],
                     [[19,13], [17,13]],
                     [[19,11], [17,11]],
                     [[19,9], [17,9]],
                     [[19,7], [17,7]],
                     [[19,5], [17,5]],
                     [[19,3], [17,3]],
                     [[19,1], [17,1]],
                     [[19,-1], [17,-1]],
                     [[19,-3], [17,-3]],
                     [[19,-6], [17,-5]],
                     [[17,-9], [16,-6]],
                     [[14,-10], [14,-6]],
                     [[12,-9], [12,-6]],
                     [[10,-8], [10,-6]],
                     [[8,-8], [8,-6]],
                     [[6,-7], [6,-6]],
                     [[4,-7], [4,-6]],
                     [[2,-7], [2,-6]],
                     [[0,-7], [0,-6]]];
        RaceEnv.__init__(self)


if __name__ == "__main__":
    pygame.init()
    env = StraightRace()

    done = False
    t = 0

    for eps in range(100):
        state = env.reset()
        print("state: ", state)

        while not done:
            t+=1
            action = np.array([4,0])
            obs, reward, done = env.step(action)
            env.plot()

    print(t)
    plt.show()

