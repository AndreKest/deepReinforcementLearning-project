"""
File to optimize hyperparameters for the algorithms with Optuna.

Algorithm: A2C, PPO, SAC, DDPG, TD3

"""
####### IMPORTS #######
import sqlite3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import optuna
from stable_baselines3 import A2C, PPO, SAC, DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from gym_environments import GymRaceEnvContinous

from wrapper import WrapperCenter, WrapperXYTheta
from race import *
#######################

###### GLOBALS ######
MAX_STEPS_CENTER = 10000
MAX_STEPS_XYTHETA = 300000
MAX_STEPS = 300000
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

def save_best_model(params: dict, algorithm: str, str_wrapper: str) -> None:
    """ Train the model and save it with the best hyperparameters. """

    set_wrapper(str_wrapper)
    env = GymRaceEnvContinous(TRACK())
    monitor = Monitor(env, filename=None, allow_early_resets=True)
    wrapper = WRAPPER(monitor)

    if algorithm == "A2C":
        model = A2C(policy="MlpPolicy", env=wrapper, **params)
    elif algorithm == "PPO":
        model = PPO(policy="MlpPolicy", env=wrapper, **params)
    elif algorithm == "SAC":
        model = SAC(policy="MlpPolicy", env=wrapper, **params)
    elif algorithm == "DDPG":
        model = DDPG(policy="MlpPolicy", env=wrapper, **params)
    elif algorithm == "TD3":
        model = TD3(policy="MlpPolicy", env=wrapper, **params)
    else:
        raise ValueError("Invalid algorithm name.")

    model.learn(total_timesteps=MAX_STEPS, progress_bar=True)

    obs, _ = wrapper.reset()
    done = False
    truncated = False

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = wrapper.step(action)


    # Save best model, rewards, gates and hyperparameters
    model.save(f"models/{str_wrapper}/{algorithm}_best_model.zip")
    # Save rewards and gates
    np.savetxt(f"models/{str_wrapper}/{algorithm}_best_model_gates.txt", np.array(monitor.get_episode_rewards()), delimiter=",")
    np.savetxt(f"models/{str_wrapper}/{algorithm}_best_model_rewards.txt", np.array(wrapper.lst_rewards), delimiter=",")
    # Save params
    pd.DataFrame(params.values(), index=params.keys()).to_csv(f"models/{str_wrapper}/{algorithm}_best_model_params.csv")

    print(f"Best model saved in models/{str_wrapper}/{algorithm}_best_model.zip")

##################

###### Hyperparameters ######
def _get_hyperparameters_A2C(trial: optuna.Trial) -> dict:
    """
    Get hyperparameters for A2C algorithm.
    """
    params = {
    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2),
    "n_steps": trial.suggest_int("n_steps", 2, 100),
    "gamma": trial.suggest_float("gamma", 0.7, 1.0),
    "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
    "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
    "vf_coef": trial.suggest_float("vf_coef", 0.0, 1.0),
    "use_rms_prop": trial.suggest_categorical("use_rms_prop", [True, False]),
    "rms_prop_eps": trial.suggest_float("rms_prop_eps", 1e-5, 1e-2),
    "use_sde": trial.suggest_categorical("use_sde", [True, False]),
    }

    return params

def _get_hyperparameters_DDPG(trial: optuna.Trial) -> dict:
    """
    Get hyperparameters for DDPG algorithm.
    """
    params = {
    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2),
    "buffer_size": trial.suggest_categorical("buffer_size", [10000, 100000, 1000000]),
    "learning_starts": trial.suggest_categorical("learning_starts", [1000, 10000, 100000]),
    "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
    "tau": trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.05, 0.1]),
    "gamma": trial.suggest_float("gamma", 0.7, 1.0),
    "train_freq": trial.suggest_int("train_freq", 1, 10),
    }

    return params

def _get_hyperparameters_PPO(trial: optuna.Trial) -> dict:
    """
    Get hyperparameters for PPO algorithm.
    """
    params = {
    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2),
    "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
    "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    "gamma": trial.suggest_float("gamma", 0.7, 1.0),
    "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
    "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
    "vf_coef": trial.suggest_float("vf_coef", 0.0, 1.0),
    "use_sde": trial.suggest_categorical("use_sde", [True, False]),
    }

    return params

def _get_hyperparameters_SAC(trial: optuna.Trial) -> dict:
    """
    Get hyperparameters for SAC algorithm.
    """
    params = {
    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2),
    "buffer_size": trial.suggest_categorical("buffer_size", [10000, 100000, 1000000]),
    "learning_starts": trial.suggest_categorical("learning_starts", [1000, 10000, 100000]),
    "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
    "tau": trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.05, 0.1]),
    "gamma": trial.suggest_float("gamma", 0.7, 1.0),
    "train_freq": trial.suggest_int("train_freq", 1, 10),
    "use_sde": trial.suggest_categorical("use_sde", [True, False]),
    }

    return params

def _get_hyperparameters_TD3(trial: optuna.Trial) -> dict:
    """
    Get hyperparameters for TD3 algorithm.
    """
    params = {
    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2),
    "buffer_size": trial.suggest_categorical("buffer_size", [10000, 100000, 1000000]),
    "learning_starts": trial.suggest_categorical("learning_starts", [1000, 10000, 100000]),
    "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
    "tau": trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.05, 0.1]),
    "gamma": trial.suggest_float("gamma", 0.7, 1.0),
    "policy_delay": trial.suggest_int("policy_delay", 1, 10),
    }

    return params

#############################

###### Objective function ######
def objective_A2C(trial: optuna.Trial) -> float:
    """
    Objective function for A2C
    return: cumulative reward 
    """
    params = _get_hyperparameters_A2C(trial)

    env = GymRaceEnvContinous(TRACK())
    monitor = Monitor(env, filename=None, allow_early_resets=True)
    wrapper = WRAPPER(monitor)

    model = A2C(policy="MlpPolicy", env=wrapper, **params)

    model.learn(total_timesteps=MAX_STEPS)

    cumulative_reward = sum(wrapper.lst_rewards)


    return float(cumulative_reward)

def objective_DDPG(trial: optuna.Trial) -> float:
    """
    Objective function for DDPG
    return: cumulative reward 
    """
    params = _get_hyperparameters_DDPG(trial)

    env = GymRaceEnvContinous(TRACK())
    monitor = Monitor(env, filename=None, allow_early_resets=True)
    wrapper = WRAPPER(monitor)

    model = DDPG(policy="MlpPolicy", env=wrapper, **params)

    model.learn(total_timesteps=MAX_STEPS)

    cumulative_reward = sum(wrapper.lst_rewards)


    return float(cumulative_reward)

def objective_PPO(trial: optuna.Trial) -> float:
    """
    Objective function for PPO
    return: cumulative reward 
    """
    params = _get_hyperparameters_PPO(trial)

    env = GymRaceEnvContinous(TRACK())
    monitor = Monitor(env, filename=None, allow_early_resets=True)
    wrapper = WRAPPER(monitor)

    model = PPO(policy="MlpPolicy", env=wrapper, **params)

    model.learn(total_timesteps=MAX_STEPS)

    cumulative_reward = sum(wrapper.lst_rewards)


    return float(cumulative_reward)

def objective_SAC(trial: optuna.Trial) -> float:
    """
    Objective function for SAC
    return: cumulative reward 
    """
    params = _get_hyperparameters_SAC(trial)

    env = GymRaceEnvContinous(TRACK())
    monitor = Monitor(env, filename=None, allow_early_resets=True)
    wrapper = WRAPPER(monitor)

    model = SAC(policy="MlpPolicy", env=wrapper, **params)

    model.learn(total_timesteps=MAX_STEPS)

    cumulative_reward = sum(wrapper.lst_rewards)
    gates_passed = sum(monitor.get_episode_rewards())


    return float(cumulative_reward)

def objective_TD3(trial: optuna.Trial) -> float:
    """
    Objective function for TD3
    return: cumulative reward 
    """
    params = _get_hyperparameters_DDPG(trial)

    env = GymRaceEnvContinous(TRACK())
    monitor = Monitor(env, filename=None, allow_early_resets=True)
    wrapper = WRAPPER(monitor)

    model = TD3(policy="MlpPolicy", env=wrapper, **params)

    model.learn(total_timesteps=MAX_STEPS)

    cumulative_reward = sum(wrapper.lst_rewards)


    return float(cumulative_reward)


################################
    
###### OPTIMIZE ######
def optimize_A2C(str_wrapper, n_trials=100) -> None:
    """
    Optimize A2C algorithm.
    """
    set_wrapper(str_wrapper)
    storage_url = f"sqlite:///db/{str_wrapper}/A2C.db"
    study = optuna.create_study(direction="maximize", storage=storage_url, study_name="A2C", load_if_exists=True)

    study.optimize(objective_A2C, n_trials=n_trials, show_progress_bar=True)

    save_best_model(study.best_params, "A2C", str_wrapper)

def optimize_DDPG(str_wrapper, n_trials=100) -> None:
    """
    Optimize DDPG algorithm.
    """
    set_wrapper(str_wrapper)
    storage_url = f"sqlite:///db/{str_wrapper}/DDPG.db"
    study = optuna.create_study(direction="maximize", storage=storage_url, study_name="DDPG", load_if_exists=True)

    study.optimize(objective_DDPG, n_trials=n_trials, show_progress_bar=True)

    save_best_model(study.best_params, "DDPG", str_wrapper)

def optimize_PPO(str_wrapper, n_trials=100) -> None:
    """
    Optimize PPO algorithm.
    """
    set_wrapper(str_wrapper)
    storage_url = f"sqlite:///db/{str_wrapper}/PPO.db"
    study = optuna.create_study(direction="maximize", storage=storage_url, study_name="PPO", load_if_exists=True)

    study.optimize(objective_PPO, n_trials=n_trials, show_progress_bar=True)

    save_best_model(study.best_params, "PPO", str_wrapper)

def optimize_SAC(str_wrapper, n_trials=100) -> None:
    """
    Optimize SAC algorithm.
    """
    set_wrapper(str_wrapper)
    storage_url = f"sqlite:///db/{str_wrapper}/SAC.db"
    storage_url = r"sqlite:///db/xytheta/SAC.db"
    study = optuna.create_study(direction="maximize", storage=storage_url, study_name="SAC", load_if_exists=True)

    study.optimize(objective_SAC, n_trials=n_trials, show_progress_bar=True)

    save_best_model(study.best_params, "SAC", str_wrapper)

def optimize_TD3(str_wrapper, n_trials=100) -> None:
    """
    Optimize TD3 algorithm.
    """
    set_wrapper(str_wrapper)
    storage_url = f"sqlite:///db/{str_wrapper}/TD3.db"
    study = optuna.create_study(direction="maximize", storage=storage_url, study_name="TD3", load_if_exists=True)

    study.optimize(objective_DDPG, n_trials=n_trials, show_progress_bar=True)

    save_best_model(study.best_params, "TD3", str_wrapper)
    
######################
    
if __name__ == "__main__":
    algorithm = "SAC"
    wrapper = "xytheta" # center

    if algorithm == "A2C":
        optimize_A2C(wrapper, n_trials=100)
    elif algorithm == "PPO":
        optimize_PPO(wrapper, n_trials=100)
    elif algorithm == "DDPG":
        optimize_DDPG(wrapper, n_trials=100)
    elif algorithm == "SAC":
        optimize_SAC(wrapper, n_trials=100)
    elif algorithm == "TD3":
        optimize_TD3(wrapper, n_trials=100)
    else:
        raise ValueError("Invalid algorithm name.")
