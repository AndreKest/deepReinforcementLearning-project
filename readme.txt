Name:         André Kestler
Datum:        08.01.2024
Studiengang:  Master KI
Fach:         Deep Reinforcement Learning
Professor:    Prof. Dr.-Ing. Thomas Nierhoff
 

Keywords: Deep Reinforcement Learning, A2C, DDPG, PPO, SAC, TD3, stable-baselines3, optuna, race environemnt

--------------------------------------------------------------------------------
Ordnerstruktur (TASK1)
.
├── db                        # Sqlite3 database files for/from Optuna optimization
    ├── enter                   # Optuna optimization files for center wrapper 
    └── xytheta                 # Optuna optimization files for x, y theta wrapper
├── models                    # Save stuff for best models (rewards, gates, hyperparameter and model weights)
    ├── enter                   # Save stuff for best models for center wrapper
    └── xytheta                 # Save stuff for best models for x, y theta wrapper
├── plots                     # The figures of the tasks (plot as png)
    ├── enter                   # Plots for the center wrapper
    └── xytheta                 # Plots x, y theta wrapper
├── src                       # Python code
    ├── eval_config.py          # File to run evaluation with best model parameters(creates plots)
    ├── gym_environemnts.py     # Race (Environment) file as a Gymnasium Environment
    ├── optimize_config.py      # File to optimize the algorithms with Optuna package
    ├── race.py                 # Race (Environment) file
    ├── utils.py                # Utils file
    └── wrapper.py              # Implemented wrapper (classes to change the observation space)
├── presentation.pptx         # Presentation file in pptx format
├── presentation.pdf          # Presentation file in pdf format
├── readme.txt                # Readme file
├── requirements.txt          # Requirements file for pip packages
└── task2.pdf                 # Task descripion


./src/db
- A2C.db   # Optuna optimization file for A2C algorithm 
- DDPG.db  # Optuna optimization file for DDPG algorithm 
- PPO.db   # Optuna optimization file for PPO algorithm 
- SAC.db   # Optuna optimization file for SAC algorithm 
- TD3.db   # Optuna optimization file for TD3 algorithm 


./src/models
- *_best_model_gates.txt      # How many gates has the best model passed during training episode
- *_best_model_params.csv     # Hyperparameters for the best model
- *_best_model_rewards.txt    # Cumulative reward per episode of the best model during training
- *_best_model.zip            # Save stuff for best model (weights, ...)


./src/plots
- *_best_model_cumulative_rewards_and_gates.png     # Scatter plot cumulative rewards with colored points for gate index
- *_best_model_cumulative_reward.png                # Cumulative reward plot for best models
-  *_best_model_gates.png                           # Gate Index plot for best models
-  *_best_model.gif                                 # GIF of the environemnt and the agent


--------------------------------------------------------------------------------
Task definition:
Solve the CurvyRace environment in file race.py with as few episodes as possible

Plot your results as follows:
- Plot your results as the cumulative reward per episode (y-axis) over episodes (x-axis)


--------------------------------------------------------------------------------
Version:

Python:             3.11.5
numpy:              1.23.5
matplotlib:         3.6.2
pandas:             2.0.3
torch:              2.0.1+cu118
stable-baselines3:  2.2.1
optuna:             3.5.0


--------------------------------------------------------------------------------
Install packages
- Install Python
- Go to the folder TASK1 
- In console: pip install -r requirements.txt


--------------------------------------------------------------------------------
Start

cd ./TASK1/                         # Go to folder TASK2/
python3 ./src/eval_config.py        # Evaluate the models
  Variables:  str_wrapper       # name of the wrapper   center, xytheta          (line: 307)
              algorithm         # name of the algorithm A2C, DDPG, PPO, SAC, TD3 (line: 308)
              MAX_STEPS         # number of steps during training                (line: 23)

python3 ./src/optimize_config.py    # Train and optimize the models
  Variables:  str_wrapper       # name of the wrapper   center, xytheta          (line: 348)
              algorithm         # name of the algorithm A2C, DDPG, PPO, SAC, TD3 (line: 349)
              MAX_STEPS         # number of steps during training                (line: 26)


--------------------------------------------------------------------------------
Algorithm
- A2C
- DDPG
- PPO
- SAC
- TD3

Used with stable-baselines3


--------------------------------------------------------------------------------
Wrapper (wrapper.py)
The idea of the transformation is that the neural network of the agent can approximate a simpler function, which can lead to an improved convergence of the training.


class WrapperCenter(gym.Env):
This is a custom environment wrapping class. Its main purpose is to transform the observations of the Deep Reinforcement Learning 
(DRL) agent to make the training process more efficient.  This class converts the original observations, which contain the x, y and 
theta coordinates of the agent into distance and angle information to the nearest gate centre point.
This wrapper is based on the idea of a fellow student (Tobias Weiß), but is implemented by myself.


class WrapperXYTheta(gym.Env):
In contrast to the first class, this class accepts the given observations of the agent unchanged. The observations in the observation space include the x-, y- 
and theta coordinates of the agent.

Both agents also implement reward shaping


--------------------------------------------------------------------------------
Reward Shaping
- r_tmp_gate = self.reward_gate(reward)
  Calculate the reward (how middle the agent drives through the gates)
  Idea: Agent should learn to drive through the middle of each gate
  Return: Reward for passing a gate 10 if passed middle area of the gate, 2 if passed above or below the middle area of the gate, 0 otherwise

- r_tmp_distance = self.reward_distance(reward, distance_to_center)
  Calculate the reward for getting closer to the next gate +1 and getting further away from the next gate -1
  Idea: Agent should learn to go to the next door (that the agent does not stay on the same place)
  Return: Reward for getting closer to the next gate 1 or punishment for getting further away from the next gate -1

- r_tmp_box = self.reward_box_boundary()
  Reward for being inside the box
  Idea: Agent should learn not to go out of a box (to go to far away from the track)
  Returns Reward for being inside the box 0 or punishment for being outside the box -10

reward = r_tmp_gate + r_tmp_distance + r_tmp_box
  The returned reward is the sum of all rewards



