# import numpy as np
import random
# import time

import torch
from torch import nn
# import tensorflow as tf

from collections import deque

from lunar import LunarLanderEnv

# Lecturas interesantes: 
# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf (Playing atari with DQN)
# https://www.nature.com/articles/nature14236 (Human level control through RL)
# https://www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained

#AQUI VA LA RED NEURONAL
class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        """
        Initialize the DQN model with the given parameters.
        Parameters:
        state_size (int): The size of the state space (number of features in the state).
        action_size (int): The size of the action space (number of possible actions).
        hidden_size (int): The size of the hidden layer in the neural network.
        """
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        """
        Forward pass of the DQN model.
        Parameters:
        x (torch.Tensor): Input tensor representing the state.
        Returns:
        torch.Tensor: Output tensor representing the Q-values for each action.
        """
        return self.net(x)
    
    #puede requerir mas funciones segun la libreria escogida.
    
class ReplayBuffer():
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size) # deque es una doble cola que permite añadir y quitar elementos de ambos extremos
        
    def push(self, state, action, reward, next_state, done):
        # insert into buffer
        # representa un paso del agente en el entorno (transición de un estado a otro)
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        # get a batch of experiences from the buffer
        # cada vez que se llama la funcion sample se copia de nuevo el buffer como
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)  # Unzip the batch into individual components
        return (torch.tensor(states, dtype=torch.float32), 
                torch.tensor(actions, dtype=torch.int64), 
                torch.tensor(rewards, dtype=torch.float32), 
                torch.tensor(next_states, dtype=torch.float32), 
                torch.tensor(dones, dtype=torch.bool))
      
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent():
    def __init__(self, lunar: LunarLanderEnv, gamma=0.99, 
                epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                learning_rate=0.001, batch_size=64, 
                memory_size=10000, episodes=1500, 
                target_network_update_freq=10,
                replays_per_episode=1000):
        """
        Initialize the DQN agent with the given parameters.
        
        Parameters:
        lunar (LunarLanderEnv): The Lunar Lander environment instance.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Decay rate for exploration rate.
        epsilon_min (float): Minimum exploration rate.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Size of the batch for experience replay.
        memory_size (int): Number of experiences stored on the replay memory.
        episodes (int): Number of episodes to train the agent.
        target_network_update_freq (int): Frequency of updating the target network.
        replays_per_episode (int): Number of experiences to replay per episode.
        """
        
        # Initialize hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.episodes = episodes
        
        self.target_updt_freq = target_network_update_freq
        self.replays_per_episode = replays_per_episode
        
        # Initialize replay memory
        # a deque is a double sided queue that allows us to append and pop elements from both ends
        self.memory = ReplayBuffer(memory_size)
        
        # Initialize the environment
        self.lunar = lunar
        
        observation_space = lunar.env.observation_space
        action_space = lunar.env.action_space
        
        # La red neuronal debe tener un numero de parametros
        # de entrada igual al espacio de observaciones
        # y un numero de salida igual al espacio de acciones.
        # Asi como un numero de capas intermedias adecuadas.
        self.q_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=64 #elegir un tamaño de capa oculta
        )
        
        self.target_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=64 #elegir un tamaño de capa oculta
        )
        
        # Set weights of target network to be the same as those of the q network
        self.target_network.update_target_network()
      
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        

        print(f"QNetwork:\n {self.q_network}")
          
    def act(self):
        """
        This function takes an action based on the current state of the environment.
        it can be randomly sampled from the action space (based on epsilon) or
        it can be the action with the highest Q-value from the model.
        """
        pass
    
        next_state, reward, done = self.lunar.take_action(action, verbose=False)
        
        return next_state, reward, done, action
    
    def update_model(self):
        """
        Perform experience replay to train the model.
        Samples a batch of experiences from memory, computes target Q-values,
        and updates the model using the computed loss.
        """
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        return loss
        
    def update_target_network(self):
        # copiar los pesos de la red q a la red objetivo (cada cuantos episodios se actualizan los pesos de la red q a la red objetivo)
        pass
        
    def save_model(self, path):
        """
        Save the model weights to a file.
        Parameters:
        path (str): The path to save the model weights.
        Returns:
        None
        """
        # guardar el modelo en el path indicado
        pass
    
    def load_model(self, path):
        """
        Load the model weights from a file.
        Parameters:
        path (str): The path to load the model weights from.
        Returns:
        None
        """
        # cargar el modelo desde el path indicado
        pass
        
    def train(self):
        """
        Train the DQN agent on the given environment for a specified number of episodes.
        The agent will interact with the environment, store experiences in memory, and learn from them.
        The target network will be updated periodically based on the update freq parameter.
        The agent will also decay the exploration rate (epsilon) over time.
        The training process MUST be logged to the console.    
        Returns:
        None
        """
        
        pass