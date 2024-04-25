import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self,batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size
    
    def generate_batches(self):
        #hacemos batch chunks a raiz de nuestros memories
        n_states = len(self.states)
        batch_start = np.arange(0,n_states,self.batch_size) #parte en batches del tamaño que toca
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices) #eliminar la correlacion y es necesario para el SGD
        batches = [indices[i:i+self.batch_size]for i in batch_start]

        return np.array(self.states), np.array(self.actions),\
                np.array(self.probs), np.array(self.vals),\
                np.array(self.rewards), np.array(self.dones),\
                batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        self.sates.append(state)
        #self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions =[]
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, lr,
                  fcl_1_dims =256, fcl_2_dims = 256, checkpoint_dir = '../checkpoints/'): #'tmp/ppo'
        #no entiendo el papel del checkpoint_dir aun
        super(ActorNetwork,self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims,fcl_1_dims),
            nn.ReLU(),
            nn.Linear(fcl_1_dims,fcl_2_dims),
            nn.ReLU(),
            nn.Linear(fcl_2_dims,n_actions),
            nn.Softmax(dim=-1) #dim=-1 aplica softmax a cada fila de manera independiente
            #offtopic, softmax2d se usa para imágenes porque lo aplica sobre [H, W] cuando tiene la estructura [N, C, H, W]
        )
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        #state or batch of state
        dist = self.actor(state)
        #usamos el output para definir una distribución categórica
        dist = Categorical(dist)
        #Necesitamos crear la distribución como tal, la clase nos da la opció de llamarla
        #y generar samples, por ejemplo
    
        return dist
    
    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CritickNetwork(nn.Module)
    def __init__(self,input_dims, lr,
                  fcl_1_dims = 256, fcl_2_dims=256, checkpoint_dir = '../checkpoints/'):
        super(CritickNetwork,self).__init__()
        #Inicializar super permite que:
        #Todos los parámetros del modelo son correctamente registrados.
        #El modelo puede ser colocado en un dispositivo (CPU o GPU) adecuadamente.
        #El modelo puede ser serializado (guardado) y cargado correctamente.
        #Las funciones autograd de PyTorch pueden rastrear los parámetros para la diferenciación automática.
        self.checkpoint_file = os.path.join(checkpoint_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims,fcl_1_dims),
                nn.ReLU(),
                nn.Linear(fcl_1_dims,fcl_2_dims),
                nn.ReLU(),
                nn.Linear(fcl_2_dims,1) #solo da el valor del estado
        )
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        #En este caso se usará el mismo lr para los dos, pero podría ser que sea mejor tunearlos
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self,state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__ (self,n_actions, gamma=0.99, lr = 0.0003, # valores base policy_clip=0.1, batch_size=64, N=2048, n_epochs=10)
                  gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, rl) #ver cuando pasa estos parámetros
        self.critic = CritickNetwork(input_dims, rl)
        self.memory = PPOMemory(batch_size)

    def remember(self,state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("--saving models--")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("--loading models--")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()