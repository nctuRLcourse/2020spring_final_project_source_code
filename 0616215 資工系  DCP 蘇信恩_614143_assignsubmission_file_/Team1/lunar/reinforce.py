import argparse
import gym
import numpy as np
from itertools import count
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pickle
from scipy import optimize
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--mm', action='store_true', help='use mm to determine beta or not')
parser.add_argument("--filename", type=str, default="default")
parser.add_argument("--omega", type=float, default=3)
parser.add_argument("--beta", type=float, default=1)
parser.add_argument("--episode", type=int, default=10000)
# parser.add_argument("--load", type=str, default=)
args = parser.parse_args()


env = gym.make('LunarLander-v2')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[
            0]
        self.hidden = 16
        self.affine1 = nn.Linear(self.observation_dim, self.hidden)
        # self.dropout = nn.Dropout(p=)
        self.affine2 = nn.Linear(self.hidden, self.action_dim)

        self.saved_log_probs = []
        self.rewards = []
        self.beta = None
        self.omega = None

        self.mm = False
        self.ii = 0

    def forward(self, x):
        x = self.affine1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return self.softmax(action_scores)
        # return F.softmax(action_scores, dim=1)

    def softmax(self, x):
        if self.mm:
            self.beta = self.find_beta(x)
        x = x - torch.max(x)
        x = torch.exp(x * self.beta)
        x = x / torch.sum(x)
        return x


    def calc_mm(self, Q):
        with torch.no_grad():
            Q = Q.clone().detach()
            c = torch.max(Q)
            Q -= c
            tmp = Q * self.omega
            tmp = torch.mean(torch.exp(tmp))
            tmp = torch.log(tmp + 1e-9)
            tmp = tmp / self.omega
            return tmp + c
    
    def find_beta(self, Q):
        with torch.no_grad():
            try:
                # self.ii += 1
                beta = optimize.brentq(self.f, -10, 10, args=(Q))
                '''
                if self.ii == 100:
                    self.ii = 0
                    # print("beta: {}".format(beta))
                '''
            except:
                traceback.print_exc()
                # print(self.f(-10, Q))
                # print(self.f(10, Q))
                exit()
            return beta

    def f(self, beta, Q):
        with torch.no_grad():
            Q = Q.clone().detach()
            Q = Q - torch.mean(Q)
            _max = torch.max(Q + 1e-8, axis = 1)[0]
            Q = Q / torch.abs(_max)

            mm_Q = self.calc_mm(Q)
            rv = 0
            for a in range(Q.shape[1]):
                tmp = (Q[0][a] - mm_Q)
                rv += np.exp(beta * tmp) * tmp

            if abs(rv) < 1e-8:
                rv = 0
            return rv


policy = Policy()
if args.mm:
    policy.mm = args.mm
    print("using mm")

policy.omega = args.omega
policy.beta = args.beta
optimizer = optim.Adam(policy.parameters(), lr=0.005)

eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def pickle_save(data, name):
    with open("./{}.pk".format(name), 'wb') as f:
        pickle.dump(data, f)
    print("File {}.pk saved!".format(name))
    
def main():
    running_reward = 10
    mean_returns = []
    n_solve = 0
    try:
        for i_episode in range(args.episode):
            state, ep_reward = env.reset(), 0
            for t in range(1, 10000):  # Don't infinite loop while learning
                action = select_action(state)
                state, reward, done, _ = env.step(action)
                if args.render:
                    env.render()
                policy.rewards.append(reward)
                ep_reward += reward
                if done:
                    break
            

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            mean_returns.append(running_reward)
            finish_episode()
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                        i_episode, ep_reward, running_reward))
            # print(env.spec.reward_threshold)
            # exit()
            if running_reward > env.spec.reward_threshold:
            # if running_reward > :
                n_solve += 1
                print("n_solve: {}".format(n_solve))
                if n_solve == 100:

                    print("Solved! Running reward is now {} and "
                        "the last episode runs to {} time steps!".format(running_reward, t))
                    
                    break

            else:
                n_solve = 0
    except:
        traceback.print_exc()
    finally:
        if args.filename != "default":
            filename = atgs.filename
        else:
            if args.mm:
                filename = 'mean_returns_mm_{}_{}'.format(args.omega, args.episode)
            else:
                filename = 'mean_returns_softmax_{}_{}'.format(args.beta, args.episode)
        pickle_save(mean_returns, filename)

if __name__ == '__main__':
    main()
