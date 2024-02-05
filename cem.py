import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import operator
import math


CEM_ITER = 100
n_action_sequences = 10
TOP_K= 5
cov = [[0.1, 0], [0, 0.1]]
TEST_ITER = 10
HORIZON = 3

# Create RNN Model
class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2l = nn.Linear(input_size + hidden_size, hidden_size)
        self.l20 = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        
        hidden = self.i2h(combined)
        l = self.i2l(combined)
        output = self.l20(l)
        return output, hidden
    
    def init_hidden(self):
        init = torch.zeros(1, self.hidden_size)
        init = init.cuda()

        return init 


def CEM(initial_observation,horizon):

    means = []

    for _ in range(horizon):
        mean = [0, 0]
        means.append(mean)

    for iteration in range(CEM_ITER):

        # each timestep gets its own gaussian
        action_sequences = np.zeros((n_action_sequences,horizon,2))

        # sample from distributions
        for h in range(horizon):
            #x, y = np.random.multivariate_normal(means[h], cov, n_action_sequences).T
            actions_at_time_h = np.random.multivariate_normal(means[h], cov, n_action_sequences).T.reshape(n_action_sequences,2)

            action_sequences[:,h,:] = actions_at_time_h

        # calculate best action sequences
        dists = []
        for m in range(n_action_sequences):
            observation = initial_observation.copy()
            hidden = model.init_hidden()
            for h in range(horizon):

                inp = observation + action_sequences[m][h].tolist() 
                inp = torch.tensor(inp, dtype=torch.float32).cuda()
                inp = inp.reshape(-1,inp.shape[0])
                y_pred, hidden = model(inp, hidden)
                y_pred = y_pred[0].detach().cpu().numpy()

                if math.isnan(y_pred[6]):
                    y_pred[6] = 0
                elif math.isinf(y_pred[6]):
                    y_pred[6] = 1
                else:
                    y_pred[6] = round(y_pred[6])

                if math.isnan(y_pred[7]):
                    y_pred[7] = 0
                elif math.isinf(y_pred[7]):
                    y_pred[7] = 1
                else:
                    y_pred[7] = round(y_pred[7])


                observation = y_pred.tolist()

            dist = np.linalg.norm(goal-y_pred)
            dists.append(dist)

        sorted_dists = sorted(enumerate(dists), key=operator.itemgetter(1))[:TOP_K]
        best_indices = list(zip(*sorted_dists))[0]
        best_dists = list(zip(*sorted_dists))[1]
        #print("Average Distance: ", sum(best_dists) / TOP_K)
        action_sequences = action_sequences[best_indices,:,: ]


        # calculate new distributions
        for h in range(horizon):
            mean = np.mean(action_sequences[:,h,:], axis=0)
            means[h] = mean

    return means

env = gym.make(
    "LunarLander-v2",
    continuous=True,
    #render_mode="human"
)

observation, info = env.reset()
observation = observation.tolist()


highest_reward = 0
goal = [0,0,0,0.5,0,0,1,1]
goal = np.array(goal)
for i in range(10000):
    action = env.action_space.sample()  # agent policy that uses the observation and info

    observation, reward, terminated, truncated, info = env.step(action)


    if reward > highest_reward:
        highest_reward = reward
        #goal = observation

    if terminated or truncated:
        observation, info = env.reset()


input_size = 10
hidden_size = 128
output_size = 8
model = RNN(input_size,hidden_size,output_size)
model = model.cuda()
model.load_state_dict(torch.load('model_weights.pth'))

env = gym.make(
    "LunarLander-v2",
    continuous=True,
    render_mode="human",
    #wind_power= 0,
    #turbulence_power=0,
)

observation, info = env.reset()
observation = observation.tolist()


total_dist = 0
for _ in range(TEST_ITER):
    print("hello")

    observation, info = env.reset()
    observation = observation.tolist()

    while True:
        means = CEM(observation,HORIZON)
        exit = False

        for mean in means:
            action = np.random.multivariate_normal(mean, cov, 1).T
            action = action.flatten()

            observation, reward, terminated, truncated, info = env.step(action)
            observation = observation.tolist()


            if terminated or truncated:
                dist = np.linalg.norm(goal-observation)
                print(dist)
                total_dist += dist
                exit = True
                break

        if exit == True:
            break





print("CEM")
print(total_dist / TEST_ITER)

total_dist = 0
for _ in range(TEST_ITER):

    observation, info = env.reset()
    observation = observation.tolist()

    while True:
        action = env.action_space.sample()  # agent policy that uses the observation and info

        observation, reward, terminated, truncated, info = env.step(action)
        observation = observation.tolist()


        if terminated or truncated:
            dist = np.linalg.norm(goal-observation)
            print(dist)
            total_dist += dist
            break

        previous_observation = observation
print("Random")
print(total_dist / TEST_ITER)

env.close()
