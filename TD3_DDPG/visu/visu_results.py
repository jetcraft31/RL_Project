import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

# This file contains variance functions to plot results
# Only a few of them are used and up-to-date
# The others are left here as examples of how to design new plotting functions


def plot_data(result, label):
    """
    Generic plot function to return a curve from a file with an index and a number per line
    importantly, several datasets can be stored into the same file
    and the curve will contain a variance information based on the repetition
    Retrieving the variance information is based on pandas
    :param filename: the file containing the data
    :param label: the label to be shown in the plot (a string)
    :return: a curve with some variance and a label, embedded in plt. 
    """
    data = np.array(result)
    x_mean = np.mean(data)
    x_std =  np.std(data)
    plt.plot(data, label=label)
    #plt.fill_between(list(range(len(data))), x1, x2, alpha=0.25)
    return x_mean, x_std


def exploit_reward_full(result) -> None:
    plot_data(result, "reward ")

    plt.xlabel("Cycles")
    plt.ylabel("Reward")
    plt.legend(loc="lower right")
    plt.savefig('data/results/rewards_' + '.pdf')
    plt.show()


def exploit_critic_loss_full(result) -> None:
    plot_data(result, "critic loss ")

    plt.xlabel("Cycles")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig('data/results/critic_loss_'  + 'pg.pdf')
    plt.show()


def exploit_policy_loss_full(result) -> None:
    plot_data(result, "policy loss")

    plt.xlabel("Cycles")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")
    plt.savefig('data/results/policy_loss_' + 'pg.pdf')
    plt.show()
