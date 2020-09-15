import traceback
import numpy as np
import gym
import pickle
from scipy import optimize
import math

def eps_greedy(Q, s, params):
    '''
    Epsilon greedy policy
    '''
    eps = params[0]
    if np.random.uniform(0, 1) < eps:
        # Choose a random action
        return np.random.randint(Q.shape[1])
    else:
        # Choose the action of a greedy policy
        return greedy(Q, s)

def f(beta, Q, s, omega):
    Q = np.copy(Q)
    
    Q = Q - np.mean(Q, axis=1)[:, np.newaxis]
    _max = np.max(Q , axis = 1) + 1e-8
    Q  = Q / abs(_max[:, np.newaxis])
    
    mm_Q_s = mm(Q, s, omega)

    rv = 0

    for a in range(Q.shape[1]):
        tmp = (Q[s][a] - mm_Q_s)
        rv += np.exp(beta * tmp) * tmp

    if abs(rv) < 1e-8:
        # print(rv)
        rv = 0
    return rv


def mm(Q, s, omega):
    probs = np.copy(Q[s])
    c = max(probs)
    probs -= c
    
    tmp = np.array(probs) * omega
    tmp = np.mean(np.exp(tmp))
    tmp = np.log(tmp + 1e-9)
    tmp = tmp / omega
    return tmp + c


def mm_sample(Q, s, params):
    omega = params[0]
    # print(f(20,Q,s,omega,env))
    # exit()
    try:
        beta = optimize.brentq(f, -10, 10, args=(Q, s, omega))
    except:
        traceback.print_exc()
        print(f(-10, Q, s, omega))
        print(f(10, Q, s, omega))
        exit()

    # print(f(beta, Q, s, omega))
    # print(beta)
    return softmax_sample(Q, s, (beta,))

    
def softmax_sample(Q, s, params):
    beta = params[0]
    # print(Q[s])
    # print(beta)
    _p = np.exp(Q[s] * beta)
    _p += 1e-10
    _sum = np.sum(_p)
    _p /= _sum

    return np.random.choice(Q.shape[1], 1, p = _p)[0]

def greedy(Q, s):
    '''
    Greedy policy
    return the index corresponding to the maximum action-state value
    '''
    return np.argmax(Q[s])


def run_episodes(env, Q, num_episodes=100, to_print=False, action_selector=eps_greedy, params=(0,0)):
    '''
    Run some episodes to test the policy
    '''
    tot_rew = []
    state = env.reset()

    for _ in range(num_episodes):
        done = False
        game_rew = 0
        
        while not done:
            # select a greedy action
            action = action_selector(Q, state, params)
            action = int(action)
            next_state, rew, done, _ = env.step(action)

            state = next_state
            game_rew += rew
            if done:
                state = env.reset()
                tot_rew.append(game_rew)

    if to_print:
        print('Mean score: %.3f of %i games!' %
              (np.mean(np.array(tot_rew)), num_episodes))

    return np.mean(np.array(tot_rew))


def Q_learning(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    nA = env.action_space.n
    nS = env.observation_space.n

    # Initialize the Q matrix
    # Q: matrix nS*nA where each row represent a state and each colums represent a different action
    Q = np.zeros((nS, nA))
    games_reward = []
    test_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0

        # decay the epsilon value until it reaches the threshold of 0.01
        if eps > 0.01:
            eps -= eps_decay

        # loop the main body until the environment stops
        while not done:
            # select an action following the eps-greedy policy
            action = eps_greedy(Q, state, eps)

            # Take one step in the environment
            next_state, rew, done, _ = env.step(action)

            # Q-learning update the state-action value (get the max Q value for the next state)
            Q[state][action] = Q[state][action] + lr * \
                (rew + gamma*np.max(Q[next_state]) - Q[state][action])

            state = next_state
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)

        # Test the policy every 300 episodes and print the results
        if (ep % 300) == 0:
            test_rew = run_episodes(env, Q, 1000)
            print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(
                ep, eps, test_rew))
            test_rewards.append(test_rew)

    return Q



def SARSA(env, lr=0.01, num_episodes=10000, gamma=0.95, action_selector=eps_greedy, params=(), test=True):
    nA = env.action_space.n
    nS = env.observation_space.n

    # Initialize the Q matrix
    # Q: matrix nS*nA where each row represent a state and each colums represent a different action
    Q = np.zeros((nS, nA)) + (1.0 / nA)
    games_reward = []
    test_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0

        # decay the epsilon value until it reaches the threshold of 0.01

        if action_selector == eps_greedy:
            eps = params[0]
            eps_decay = params[1]
            if eps > 0.01:
                eps -= eps_decay
            params = (eps, eps_decay)

        action = action_selector(Q, state, params)
        action = int(action)
        

        # loop the main body until the environment stops
        while not done:
            # Take one step in the environment
            next_state, rew, done, _ = env.step(action)

            # choose the next action (needed for the SARSA update)
            next_action = action_selector(Q, next_state, params)
            next_action = int(next_action)
            # SARSA update
            Q[state][action] = Q[state][action] + lr * \
                (rew + gamma*Q[next_state][next_action] - Q[state][action])

            state = next_state
            action = next_action
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)

        # Test the policy every 300 episodes and print the results
        if test and (ep % 300) == 0:
            test_rew = run_episodes(env, Q, 1000)
            if action_selector == eps_greedy:
                eps, eps_decay = params
                print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(
                    ep, eps, test_rew))
            else:
                print("Episode:{:5d}  Rew:{:2.4f}".format(ep, test_rew))
                
            test_rewards.append(test_rew)

    return Q, test_rewards


def save_Q_table(name, Q_table):
    with open("./{}.pk".format(name), 'wb') as f:
        pickle.dump(Q_table, f)
    print("File {}.pk saved!".format(name))

def load_Q_table(name):
    with open("./{}.pk".format(name), 'rb') as f:
        
        table = pickle.load(f)
    print("File {}.pk loaded!".format(name))
    return table

def test_range(a, b, step, lr, num_episodes, gamma, action_selector, filename):
    env = gym.make('Taxi-v3')
    env._max_episode_steps = 20
    nA = env.action_space.n
    nS = env.observation_space.n
    params = None
    mean_rewards = []
    N_VALIDATION = 1
    for index,i in enumerate(np.arange(a, b, step)):
        print("test_range run with {}".format(index))
        if action_selector == eps_greedy:
            params = (i, 0.001)
        else:
            params = (i,) 
        total_reward = 0
        for j in range(N_VALIDATION):
            for _i_ in range(5):
                try:   
                    Q, _ = SARSA(env, lr=lr, num_episodes=num_episodes,
                            gamma=gamma, action_selector=action_selector, params=params, test=False)
                    break
                except:
                    traceback.print_exc()
                    if _i_ == 5-1:
                        Q = np.zeros((nS, nA))
                        
                    continue
                

            if action_selector == eps_greedy:
                _p = (0, 0)
            else:
                _p = (i, 0)

            total_reward += run_episodes(env, Q, num_episodes=25, to_print=True, action_selector=action_selector, params=_p)

        mean_rewards.append((i, total_reward/N_VALIDATION))


    with open(filename, 'wb') as f:
        pickle.dump(mean_rewards, f)
    
    return mean_rewards

if __name__ == '__main__':
    '''
    mr = test_range(0, 1, 0.02, lr=0.1, num_episodes=10000, gamma=0.99, action_selector=eps_greedy, filename="eps")
    for eps, r in mr:
        print("Mean reward is: {} with eps {}".format(r, round(eps, 2)))
    '''
    mr = test_range(1, 10, 0.5 ,lr=0.1, num_episodes=20000, gamma=0.99, action_selector=mm_sample, filename="mm")

    exit()
    
    env = gym.make('Taxi-v3')
    env._max_episode_steps = 20

    TRAIN = False
    TEST = False
    if TRAIN:
        params_greedy = (0.4, 0.001)
        Q_eps, test_rewards_eps = SARSA(env, lr=.1, num_episodes=20000,
                       gamma=0.99, action_selector=eps_greedy, params=params_greedy, test=False)
        save_Q_table("Q_eps", Q_eps)
    if TEST:
        Q_eps = load_Q_table("Q_eps")
        run_episodes(env, Q_eps, to_print=True, action_selector=eps_greedy, params=(0,0))



    TRAIN = False
    TEST = False
    BETA =  1
    if TRAIN:
        Q_softmax, test_rewards_softmax = SARSA(env, lr=.1, num_episodes=20000,
                        gamma=0.99, action_selector=softmax_sample, params=(BETA,), test=False)
        save_Q_table("Q_softmax", Q_softmax)
    
    if TEST:
        Q_softmax = load_Q_table("Q_softmax")
        run_episodes(env, Q_softmax, to_print=True, action_selector=softmax_sample, params=(BETA,0))

    OMEGA = 100
    if True:
        Q_mm, test_rewards_mm = SARSA(env, lr=.1, num_episodes=500,
                        gamma=0.99, action_selector=mm_sample, params=(OMEGA,), test=True)
        save_Q_table("Q_mm", Q_mm)

    if True:
        Q_mm = load_Q_table("Q_mm")
        run_episodes(env, Q_mm, to_print=True, action_selector=mm_sample, params=(OMEGA,0))