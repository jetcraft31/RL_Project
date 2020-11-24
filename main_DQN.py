import gym
import pandas as pd
import argparse

from Agent.DQN_Agent import DQN
from utils.visu.visu_results2 import exploit_reward_full,exploit_duration_full

def create_data_folders() -> None:
    """
    Create folders where to save output files if they are not already there
    :return: nothing
    """
    if not os.path.exists("data/save"):
        os.mkdir("./data")
        os.mkdir("./data/save")
    if not os.path.exists("data/critics"):
        os.mkdir("./data/critics")
    if not os.path.exists('data/policies/'):
        os.mkdir('data/policies/')
    if not os.path.exists('data/results/'):
        os.mkdir('data/results/')

def newReward(obsesrvation, obsesrvation_):
    return abs(obsesrvation_[0] - (-0.5))

def update(args):

    records = []
    for episode in range(args.TRAIN_EPISODE_NUM):
        # initial
        observation = env.reset()

        iter_cnt, total_reward = 0, 0
        while True:
            iter_cnt += 1

            # fresh env
            #env.render()

            # RL choose action based on observation
            action = RL.select_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done, _ = env.step(action)
            #reward = newReward(observation, observation_)
            # RL learn from this transition
            RL.store_transition(observation, action, reward, observation_)
            if RL.memory_counter > args.MEMORY_CAPACITY:
                RL.learn()

            # accumulate reward
            total_reward += reward
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                total_reward = round(total_reward, 2)
                records.append((iter_cnt, total_reward))
                print("Episode {} finished after {} timesteps, total reward is {}".format(episode + 1, iter_cnt, total_reward))
                break

    # end of game
    print('--------------------------------')
    print('game over')
    env.close()

    # save model
    RL.save_model()
    print("save model")

    df = pd.DataFrame(records, columns=["iters", "reward"])
    df.to_csv("data/save/DQN_{}_{}_{}.csv".format(RL.lr, args.E_GREEDY, args.BATCH_SIZE), index=False)

    exploit_reward_full(df["reward"])
    exploit_duration_full(df["iters"])

if __name__ == "__main__":

    create_data_folders()

    # argument
    parse = argparse.ArgumentParser()
    parse.add_argument('-lr', '--learning_rate',
                        type=float, default=0.01,
                        help='Learning rate')
    parse.add_argument('-rd', '--reward_decay',
                        type=float, default=0.9,
                        help='Reward decay')
    parse.add_argument('--BATCH_SIZE', type = int, default=32)
    parse.add_argument('--E_GREEDY', type = float, default=0.999)
    parse.add_argument('--TARGET_REPLACE_ITER', type = int, default=100)
    parse.add_argument('--MEMORY_CAPACITY', type = int, default=2000)
    parse.add_argument('--TRAIN_EPISODE_NUM', type = int, default=500)
    args = parse.parse_args()

    # env setup
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    env = gym.wrappers.TimeLimit(env, 2000)

    # algorithm setup
    print("Use DQN...")
    print('--------------------------------')
    env_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # to confirm the shape
    RL = DQN(action_n=env.action_space.n, state_n=env.observation_space.shape[0], env_shape=env_shape,
            learning_rate=args.learning_rate, reward_decay=args.reward_decay,args = args)

    update(args)
