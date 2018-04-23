#By William Fraher
#April 19, 2018

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_size = observation_size * action_size

iterations = 1000 #amount of updates to be performed
population = 20 #amount of parameters to keep track of
keep_count = 4 #the best keep_count thetas are used for the future
max_timesteps = 500 #maximum amount of steps in an environment. In CartPole-v1, this is 500.


theta = np.array([hidden_size]) #4 inputs, two outputs (observation and action)
mean = np.random.randn(hidden_size) #the mean for the gaussian distribution
stdev = np.random.randn(hidden_size) #standard deviation for gaussian distribution

def preprocess(state):
    #Preprocesses the state into a horizontal vector.
    return np.reshape(state, [1,observation_size])

total_rewards = []
for i in range(iterations):
    diag = np.diag(stdev) #diagonal matrix of stdev
    theta = np.random.multivariate_normal(mean,diag,population) #samples population parameter sets given the current mean and standard deviation
    results = [] #results of each rollout
    best = [] #takes the keep_count best thetas
    for e in theta:
        s = preprocess(env.reset()) #lets our feedforward network manipulate the state
        d = False
        rewards = 0
        W = np.reshape(e,[observation_size,action_size]) #reshapes the parameter vector to compute actions
        for t in range(max_timesteps):
	    #env.render()
            a = np.argmax(np.matmul(s,W))
            ns, r, d, _ = env.step(a)
            s = preprocess(ns)
            rewards += r
            if d:
                results.append([e,rewards])
                break
        if t == max_timesteps and not d:
            results.append([e,rewards])
    print 'Iteration ' + str(i) +' finished with reward ' + str(results[np.argmax(np.asarray(results)[:,1])][1])
    total_rewards.append(results[np.argmax(np.asarray(results)[:,1])][1]) #saves the reward to graph later
    for b in range(keep_count): #takes the best keep_count thetas
        best.append(results[np.argmax(np.asarray(results)[:,1])][0]) #add the best one to a new list
        results.pop(np.argmax(np.asarray(results)[:,1])) #take out the best one from the old list 
    #Creates the new mean and standard deviation based off of our best samples
    mean = np.mean(best,axis=0)
    stdev = np.std(best,axis=0)
    
plt.plot(total_rewards)
plt.show()
