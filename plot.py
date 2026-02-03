import matplotlib.pyplot as plt
from glob import glob
import os
import os.path as op
import numpy as np

def extract_mean_reward(file):
    """
    Extracts mean reward values from output logs
    
    :param file: file object returned from open()

    Returns an np.array of floats
    """
    lines = file.readlines()
    vals = [float(r.split('reward: ')[1].split(',')[0]) for r in lines if 'reward' in r]
    return np.array(vals)

def plot_rewards(results, show=False):
    """
    Plots reward curves from training results
    
    :param results: (dict) results from training indexed by 'model_name'
    :param show: (bool) show the plot
    """
    for model in results.keys():
        plt.plot(results[model], label=model)

    plt.legend()
    plt.xlabel('Generations')
    plt.ylabel('Average Reward')
    plt.ylim([0, 1.1])
    plt.savefig(op.join('results', 'average_rewards.png'))
    if show: plt.show(block=False)

if __name__ == '__main__':
    result_files = glob(op.join('logs', '*_output.txt'))

    rewards = {}
    for file in result_files:
        model_name = os.path.basename(file).split('_output')[0]
        fo = open(file)
        reward = extract_mean_reward(fo)
        print(model_name)
        print(reward)
        rewards[model_name] = reward
    
    plot_rewards(rewards, show=True)

        
