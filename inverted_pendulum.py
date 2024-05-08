import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import operator
import math

if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev)

print(dev)


class LunarLanderDynamics():

    ## Training Data
    n_samples=100000
    batch_size = 1
    n_epochs = 500

    ## Model Data
    #input_size = 10
    input_size = 5 # state space size + action size
    hidden_size = 128
    #output_size = 8
    output_size = 4
    learning_rate = 0.01

    ## CEM Data
    n_iter= 500
    n_action_sequences = 100
    n_top= 5
    #cov = [[0.1, 0], [0, 0.1]]



    def __init__(self):
        #self.goal = [0,0,0,0,0,0,1,1]
        self.goal = [0,0,0,0]
        self.goal = np.array(self.goal)

    def generate_model(self):

        model = RNN(self.input_size,self.hidden_size,self.output_size)
        model = model.cuda()

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        print(model)

        return model,loss_fn,optimizer

    def gather_training_data(self, loss_fn, use_cem = False, render = False):


        if render:
            #env = gym.make(
            #    "LunarLander-v2",
            #    continuous=True,
            #    render_mode="human"
            #)
            env = gym.make(
                "InvertedPendulum-v4",
                render_mode="human"
            )
        else:
            #env = gym.make(
            #    "LunarLander-v2",
            #    continuous=True,
            #)
            env = gym.make(
                "InvertedPendulum-v4",
            )

        observation, info = env.reset()
        observation= observation.tolist()

        X = np.zeros((self.n_samples,self.batch_size,self.input_size))
        Y = np.zeros((self.n_samples,self.output_size))

        terminated = False
        truncated = False

        for i in range(self.n_samples):
            for j in range(self.batch_size):

                if use_cem:
                    means, std_dev = self.CEM(observation,loss_fn)
                    action = np.random.normal(loc=means[0], scale=std_dev[0], size=(1,1))
                    action = action.flatten()
                else:
                    action = env.action_space.sample()  # agent policy that uses the observation and info

                observation, reward, terminated, truncated, info = env.step(action)

                observation = observation.tolist()
                action = action.tolist()

                X[i][j] = observation + action

            Y[i] = observation

            if terminated or truncated:
                if use_cem:
                    l = np.linalg.norm(self.goal[1]-observation[1])
                    print(l)

                terminated = False
                truncated = False

                observation, info = env.reset()
                observation = observation.tolist()

        env.close()

        X = torch.tensor(X, dtype=torch.float32).cuda()
        Y = torch.tensor(Y, dtype=torch.float32).cuda()


        x_train,y_train,x_val,y_val = self.split_data(X,Y)


        return x_train,y_train,x_val,y_val 
    def split_data(self,X,Y):
        train_size = int(0.8 * len(X))
        test_size = len(X) - train_size
        x_train, x_val = torch.utils.data.random_split(X, [train_size, test_size])
        y_train, y_val = torch.utils.data.random_split(Y, [train_size, test_size])

        return x_train,y_train,x_val,y_val 

    def __train_batch(self,model,loss_fn,optimizer,xbatch, y):
        hidden = model.init_hidden()

        for i in range(xbatch.size()[0]):
            x =  xbatch[i]
            x = x.reshape(-1,x.shape[0])
            output, hidden = model(x, hidden)
            
        y = y.reshape(-1,y.shape[0])
        loss = loss_fn(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return output, loss.item()


    def __predict_batch(self,model,loss_fn,optimizer,xbatch,y):
        with torch.no_grad():
            
            hidden = model.init_hidden()
        
            for i in range(xbatch.size()[0]):
                x =  xbatch[i]
                x = x.reshape(-1,x.shape[0])
                output, hidden = model(x, hidden)
            
            y = y.reshape(-1,y.shape[0])
            l = loss_fn(output, y)
            return l

    def predict(self,model,loss_fn,optimizer,X,Y):

        loss = 0
        for i in range(0, len(X)):
            Xbatch = X[i]
            y = Y[i]
            l = self.__predict_batch(model,loss_fn,optimizer,Xbatch,y)
            loss += l

        loss = loss / len(X)

        return loss


    def train(self,model,loss_fn,optimizer, x_train,y_train,x_val,y_val):
        for epoch in range(self.n_epochs):
            loss = 0
            for i in range(0, len(x_train)):
                Xbatch = x_train[i]
                y = y_train[i]
                output, l = self.__train_batch(model,loss_fn,optimizer,Xbatch,y)
                loss += l

            train_loss = loss / len(x_train)

            val_loss = self.predict(model,loss_fn,optimizer,x_val,y_val)



            print(f'Finished epoch {epoch}, Training loss: {train_loss}, Validation Loss: {val_loss}')

    def CEM(self,initial_observation,loss_fn):

        ## each time step gets its own gaussian
        ## initialize means to zero
        means = [[0] for _ in range (self.batch_size)]
        std_dev = [[0.5] for _ in range (self.batch_size)]



        ## Optimize guassians over many iterations
        #for iteration in range(self.n_iter):

        best_average = math.inf
        i = 0
        while True or i > self.n_iter:
            #print(i)

            action_sequences = np.random.normal(loc=means, scale=std_dev, size=(self.n_action_sequences,self.batch_size,1))


            # calculate best action sequences
            dists = []
            for action_idx in range(self.n_action_sequences):

                observation = initial_observation.copy()
                hidden = model.init_hidden()

                for h in range(self.batch_size):

                    # sample from distriubtion given current time step
                    inp = observation + action_sequences[action_idx][h].tolist() 
                    # generate input to feed into model
                    inp = torch.tensor(inp, dtype=torch.float32).cuda()
                    inp = inp.reshape(-1,inp.shape[0])

                    # get model prediction
                    y_pred, hidden = model(inp, hidden)
                    y_pred = y_pred[0].detach().cpu().numpy()


                    observation = y_pred.tolist()

                dist = np.linalg.norm((self.goal[1]-y_pred[1]))
                dists.append(dist)

            #print("Dists")
            #print(dists)
            sorted_dists = sorted(enumerate(dists), key=operator.itemgetter(1))[:self.n_top]
            best_indices = list(zip(*sorted_dists))[0]
            best_dists = list(zip(*sorted_dists))[1]

            #print ("Best Indices")
            #print(best_indices)
            #print("Average Distance: ", sum(best_dists) / TOP_K)

            # iterate until our predictions don't get better
            if (sum(best_dists) / self.n_top) >= best_average:
                break
            else:
                best_average = sum(best_dists) / self.n_top 


            action_sequences = action_sequences[best_indices,:,: ]

            # calculate new distributions
            means = np.mean(action_sequences, axis=0)
            std_dev = np.std(action_sequences, axis=0)

            i += 1

        return means, std_dev




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


ll = LunarLanderDynamics()

model,loss_fn,optimizer = ll.generate_model()

x_train,y_train,x_val,y_val = ll.gather_training_data(loss_fn)

ll.train(model,loss_fn,optimizer,x_train,y_train,x_val,y_val)
print("Saving Weights")
torch.save(model.state_dict(), "model_weights.pth")

ll.n_epochs = 10
ll.n_samples = 100
for _ in range(5):
    x,y = ll.gather_training_data(loss_fn,True, False)
    #X,Y = ll.gather_training_data(loss_fn,True, False)
    #X,Y = ll.gather_training_data(loss_fn, True, True)

    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()

    X = np.append(X,x,axis=0)
    Y = np.append(Y,y,axis=0)

    print(X.shape)
    print(Y.shape)

    X = torch.tensor(X, dtype=torch.float32).cuda()
    Y = torch.tensor(Y, dtype=torch.float32).cuda()

    ll.train(model,loss_fn,optimizer,X,Y)

    # Prediction loss on random data
    #x,y = ll.gather_training_data(loss_fn,False,False)
    #print("Prediction loss on random data: ",ll.predict(model,loss_fn,optimizer,x,y))


ll.n_samples = 10000
ll.gather_training_data(loss_fn,True, True)

