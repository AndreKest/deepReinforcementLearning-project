import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def save_frames_as_gif(algorithm, frames, path='./', filename='gym_animation.gif', fps=10):
    """ Save a list of frames as a gif.
        Source: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

        Args:
            frames: A list of frames to be saved
            path: The path were to save the gif
            filename: The filename of the gif
            fps: Frames per second
    """
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    plt.title("Best Model "+algorithm)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=fps)


def convert_params(param_dict):
    """
    Convert the parameters to the correct type. 

    Args:
        param_dict: Dictionary with parameters

    Returns:
        Dictionary with converted parameters
    """
    converted_params = {}
    for key, value in param_dict.items():
        if key == 'use_sde':
            converted_params[key] = value.lower() == 'true'
        else:
            try:
                converted_params[key] = int(value)
            except ValueError:
                try:
                    converted_params[key] = float(value)
                except ValueError:
                    converted_params[key] = value
    return converted_params

def plt_cumulative_reward(rewards, gates, algorithm, path='./', filename='best_model'):
    """
    Plot the cumulative reward of a trained model.

    Args:
        rewards: List of rewards
        path: Path to save the plot
        filename: Filename of the plot

    Returns:
        None
    """
    plt.clf()
    plt.title("Cumulative Reward "+algorithm)
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.savefig(path + f"/{algorithm}_{filename}_cumulative_reward.png")

    plt.clf()
    
    plt.title("Gates "+algorithm)
    plt.plot(gates)
    plt.xlabel("Episode")
    plt.ylabel("Gates")
    plt.savefig(path + f"/{algorithm}_{filename}_gates.png")

    plt.clf()
    color = np.array(gates, dtype=np.int32)
    plt.title("Cumulative Reward and Gates "+algorithm)
    plt.scatter(range(len(rewards)), rewards, c=color, s=7)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    cbar = plt.colorbar()
    cbar.set_label("Gate")
    plt.savefig(path + f"/{algorithm}_{filename}_cumulative_reward_and_gates.png")
