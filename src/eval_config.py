"""
File to evaluate the best model of each algorithm.

Algorithm: A2C, PPO, SAC, DDPG, TD3

"""
####### IMPORTS #######
from gym_environments import GymRaceEnvContinous
from race import CurvyRace
from wrapper import WrapperCenter, WrapperXYTheta

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor

from utils import convert_params, save_frames_as_gif, plt_cumulative_reward
#######################

###### GLOBALS ######
MAX_STEPS = 300000 # 10000
TRACK = CurvyRace

#####################

###### UTILS ######
def set_wrapper(wrapper: str) -> None:
    """
    Set the wrapper as a global variable for the environemnt.

    Args:
        wrapper: Name of the wrapper (center, xytheta)

    Returns:
        None
    """
    global WRAPPER
    if wrapper == "center":
        WRAPPER = WrapperCenter
    elif wrapper == "xytheta":
        WRAPPER = WrapperXYTheta

###################

###### Evaluation of Algorithm ######
def evaluate_A2C(str_wrapper, algorithm="A2C", render=True, save_gif=True):
    """
    Evaluate the A2C algorithm.
    Save the evaluation as a gif and plot/save the cumulative reward.

    Args:
        str_wrapper: Name of the wrapper (center, xytheta)
        algorithm: Name of the algorithm
        render: Render the environment
        save_gif: Save the evaluation as a gif

    Returns:
        None
    """
    set_wrapper(str_wrapper)

    # create environment
    env = GymRaceEnvContinous(TRACK(), "rgb_array")
    monitor = Monitor(env)
    wrapped = WRAPPER(monitor)

    # load parameters
    params = pd.read_csv(f"./models/{str_wrapper}/{algorithm}_best_model_params.csv", index_col=0)
    params = params.to_dict()["0"]
    params = convert_params(params)

    # load model
    model = sb3.A2C.load(f"./models/{str_wrapper}/{algorithm}_best_model.zip", env=wrapped, **params)

    # evaluate model
    obs, _ = wrapped.reset()
    done = False
    truncated = False
    images = []

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = wrapped.step(action)

        if render == True or save_gif == True:
            img = wrapped.render()
            images.append(img)

    if save_gif == True:
        save_frames_as_gif(algorithm, images, path=f"./plots/{str_wrapper}/", filename=f"{algorithm}_best_model.gif")
    
    lst_rewards = np.loadtxt(f"./models/{str_wrapper}/{algorithm}_best_model_rewards.txt", delimiter=",")
    lst_gates = np.loadtxt(f"./models/{str_wrapper}/{algorithm}_best_model_gates.txt", delimiter=",")

    plt_cumulative_reward(lst_rewards, lst_gates, algorithm, path=f"./plots/{str_wrapper}/")

def evaluate_DDPG(str_wrapper, algorithm="DDPG", render=True, save_gif=True):
    """
    Evaluate the DDPG algorithm.
    Save the evaluation as a gif and plot/save the cumulative reward.

    Args:
        str_wrapper: Name of the wrapper (center, xytheta)
        algorithm: Name of the algorithm
        render: Render the environment
        save_gif: Save the evaluation as a gif

    Returns:
        None
    """
    set_wrapper(str_wrapper)

    # create environment
    env = GymRaceEnvContinous(TRACK(), "rgb_array")
    monitor = Monitor(env)
    wrapped = WRAPPER(monitor)

    # load parameters
    params = pd.read_csv(f"./models/{str_wrapper}/{algorithm}_best_model_params.csv", index_col=0)
    params = params.to_dict()["0"]
    params = convert_params(params)

    # load model
    model = sb3.DDPG.load(f"./models/{str_wrapper}/{algorithm}_best_model.zip", env=wrapped, **params)

    # evaluate model
    obs, _ = wrapped.reset()
    done = False
    truncated = False
    images = []

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = wrapped.step(action)

        if render == True or save_gif == True:
            img = wrapped.render()
            images.append(img)

    if save_gif == True:
        save_frames_as_gif(algorithm, images, path=f"./plots/{str_wrapper}/", filename=f"{algorithm}_best_model.gif")
    
    lst_rewards = np.loadtxt(f"./models/{str_wrapper}/{algorithm}_best_model_rewards.txt", delimiter=",")
    lst_gates = np.loadtxt(f"./models/{str_wrapper}/{algorithm}_best_model_gates.txt", delimiter=",")

    plt_cumulative_reward(lst_rewards, lst_gates, algorithm, path=f"./plots/{str_wrapper}/")

def evaluate_PPO(str_wrapper, algorithm="PPO", render=True, save_gif=True):
    """
    Evaluate the PPO algorithm.
    Save the evaluation as a gif and plot/save the cumulative reward.

    Args:
        str_wrapper: Name of the wrapper (center, xytheta)
        algorithm: Name of the algorithm
        render: Render the environment
        save_gif: Save the evaluation as a gif

    Returns:
        None
    """
    set_wrapper(str_wrapper)

    # create environment
    env = GymRaceEnvContinous(TRACK(), "rgb_array")
    monitor = Monitor(env)
    wrapped = WRAPPER(monitor)

    # load parameters
    params = pd.read_csv(f"./models/{str_wrapper}/{algorithm}_best_model_params.csv", index_col=0)
    params = params.to_dict()["0"]
    params = convert_params(params)

    # load model
    model = sb3.PPO.load(f"./models/{str_wrapper}/{algorithm}_best_model.zip", env=wrapped, **params)

    # evaluate model
    obs, _ = wrapped.reset()
    done = False
    truncated = False
    images = []

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = wrapped.step(action)

        if render == True or save_gif == True:
            img = wrapped.render()
            images.append(img)

    if save_gif == True:
        save_frames_as_gif(algorithm, images, path=f"./plots/{str_wrapper}/", filename=f"{algorithm}_best_model.gif")

    lst_rewards = np.loadtxt(f"./models/{str_wrapper}/{algorithm}_best_model_rewards.txt", delimiter=",")
    lst_gates = np.loadtxt(f"./models/{str_wrapper}/{algorithm}_best_model_gates.txt", delimiter=",")

    plt_cumulative_reward(lst_rewards, lst_gates, algorithm, path=f"./plots/{str_wrapper}/")


def evaluate_SAC(str_wrapper, algorithm="SAC", render=True, save_gif=True):
    """
    Evaluate the SAC algorithm.
    Save the evaluation as a gif and plot/save the cumulative reward.

    Args:
        str_wrapper: Name of the wrapper (center, xytheta)
        algorithm: Name of the algorithm
        render: Render the environment
        save_gif: Save the evaluation as a gif

    Returns:
        None
    """
    set_wrapper(str_wrapper)

    # create environment
    env = GymRaceEnvContinous(TRACK(), "rgb_array")
    monitor = Monitor(env)
    wrapped = WRAPPER(monitor)

    # load parameters
    params = pd.read_csv(f"./models/{str_wrapper}/{algorithm}_best_model_params.csv", index_col=0)
    params = params.to_dict()["0"]
    params = convert_params(params)

    # load model
    model = sb3.SAC.load(f"./models/{str_wrapper}/{algorithm}_best_model.zip", env=wrapped, **params)

    # evaluate model
    obs, _ = wrapped.reset()
    done = False
    truncated = False
    images = []

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = wrapped.step(action)

        if render == True or save_gif == True:
            img = wrapped.render()
            images.append(img)

    if save_gif == True:
        save_frames_as_gif(algorithm, images, path=f"./plots/{str_wrapper}/", filename=f"{algorithm}_best_model.gif")
    
    lst_rewards = np.loadtxt(f"./models/{str_wrapper}/{algorithm}_best_model_rewards.txt", delimiter=",")
    lst_gates = np.loadtxt(f"./models/{str_wrapper}/{algorithm}_best_model_gates.txt", delimiter=",")

    plt_cumulative_reward(lst_rewards, lst_gates, algorithm, path=f"./plots/{str_wrapper}/")

def evaluate_TD3(str_wrapper, algorithm="TD3", render=True, save_gif=True):
    """
    Evaluate the TD3 algorithm.
    Save the evaluation as a gif and plot/save the cumulative reward.

    Args:
        str_wrapper: Name of the wrapper (center, xytheta)
        algorithm: Name of the algorithm
        render: Render the environment
        save_gif: Save the evaluation as a gif

    Returns:
        None
    """
    set_wrapper(str_wrapper)

    # create environment
    env = GymRaceEnvContinous(TRACK(), "rgb_array")
    monitor = Monitor(env)
    wrapped = WRAPPER(monitor)

    # load parameters
    params = pd.read_csv(f"./models/{str_wrapper}/{algorithm}_best_model_params.csv", index_col=0)
    params = params.to_dict()["0"]
    params = convert_params(params)

    # load model
    model = sb3.TD3.load(f"./models/{str_wrapper}/{algorithm}_best_model.zip", env=wrapped, **params)

    # evaluate model
    obs, _ = wrapped.reset()
    done = False
    truncated = False
    images = []

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = wrapped.step(action)

        if render == True or save_gif == True:
            img = wrapped.render()
            images.append(img)

    if save_gif == True:
        save_frames_as_gif(algorithm, images, path=f"./plots/{str_wrapper}/", filename=f"{algorithm}_best_model.gif")
    
    lst_rewards = np.loadtxt(f"./models/{str_wrapper}/{algorithm}_best_model_rewards.txt", delimiter=",")
    lst_gates = np.loadtxt(f"./models/{str_wrapper}/{algorithm}_best_model_gates.txt", delimiter=",")

    plt_cumulative_reward(lst_rewards, lst_gates, algorithm, path=f"./plots/{str_wrapper}/")

#####################################
    
if "__main__" == __name__:
    str_wrapper = "xytheta"
    algorithm = "SAC"

    if algorithm == "A2C":
        evaluate_A2C(str_wrapper)
    elif algorithm == "DDPG":
        evaluate_DDPG(str_wrapper)
    elif algorithm == "PPO":
        evaluate_PPO(str_wrapper)
    elif algorithm == "SAC":
        evaluate_SAC(str_wrapper)
    elif algorithm == "TD3":
        evaluate_TD3(str_wrapper)
    else:
        raise ValueError("Algorithm not known.")
