import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
from copy import deepcopy
from math import *
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"



#MODEL
class Judge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1) #2 input planes
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(3*3*256, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
    
#DATASET
def load_data():
    # load MNIST dataset
    transform = transforms.ToTensor()
    training_data = datasets.MNIST(
        root='data', 
        train=True, 
        download=True, 
        transform=transform
    )
    # load test dataset
    test_data = datasets.MNIST(
        root='data', 
        train=False, 
        download=True, 
        transform=transform
    )

    train_loader = DataLoader(training_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

    return train_loader, test_loader

#MASK
def generate_mask(image, pixel_coordinates=None):
    # create mask
    mask = torch.zeros_like(image)

    # reveal the pixel at each given coordinate if it's non-zero
    if pixel_coordinates:
        for pixel_coordinate in pixel_coordinates:
            x, y = pixel_coordinate
            if torch.any(image[:, x, y] != 0):
                mask[:, x, y] = 1

    # apply mask to image
    masked_image = image * mask

    return mask, masked_image

#GAME LOGIC
class Game:
    def __init__(self, image, label_honest, label_liar):
        self.image = image
        self.revealed_pixels = []
        self.mask, self.masked_image = generate_mask(image, self.revealed_pixels)
        self.labels = [label_honest, label_liar]
        self.label_honest = label_honest
        self.label_liar = label_liar
        self.revealed = 0
        self.turn = random.randint(0, 1)

    def step(self, action):
        x, y = action
        if torch.all(self.mask[:, x, y] == 0) and torch.any(self.image[:, x, y] != 0):  # only reveal if the pixel is not already revealed and is non-zero
            self.revealed_pixels.append((x, y))
            self.mask, self.masked_image = generate_mask(self.image, self.revealed_pixels)
            self.revealed += 1
        self.turn = 1 - self.turn  # switch turn

    def is_over(self):
        return self.revealed >= 6
    
    def get_winner(self, model):
        with torch.no_grad():
            device = next(model.parameters()).device
            inputs = torch.cat((self.mask.unsqueeze(0), self.masked_image.unsqueeze(0)), dim=1).to(device)
            outputs = model(inputs)
        # get the logits for the claimed classes
        logit_honest = outputs[0, self.label_honest]
        logit_liar = outputs[0, self.label_liar]

        # determine the winner
        return "honest" if logit_honest > logit_liar else "liar"
    
    def clone(self):
        cloned_game = copy.deepcopy(self)
        return cloned_game
    
    def get_valid_actions(self):
        valid_actions = []
        for x in range(self.image.shape[1]):
            for y in range(self.image.shape[2]):
                if self.image[0, x, y] != 0 and self.mask[0, x, y] == 0:
                    valid_actions.append((x, y))
        return valid_actions

    def get_observation(self):
        return self.mask, self.masked_image#, self.label_honest, self.label_liar #not needed

    def prepare_inputs(self, observation):
        mask, masked_image = observation
        inputs = torch.cat((mask.unsqueeze(0), masked_image.unsqueeze(0)), dim=1)
        return inputs.to(device)
    
    def get_state(self):
        return {
            "image": self.image,
            "revealed_pixels": self.revealed_pixels,
            "mask": self.mask,
            "masked_image": self.masked_image,
            "labels": self.labels,
            "revealed": self.revealed,
            "turn": self.turn
        }

# Load the MNIST dataset
train_loader, test_loader = load_data()

# Load the trained model
model = Judge()
model.load_state_dict(torch.load('judge.pth'))
model.to(device)
model.eval()



class Node:  
    def __init__(self, game, done, parent, observation, action_index, model):
        self.child = None
        self.T = 0
        self.N = 0
        self.game = game
        self.observation = observation
        self.done = done
        self.parent = parent
        self.action_index = action_index
        self.nn_v = 0
        self.p = None
        self.model = model 
        
        
    def getPUCTscore(self):
        c = 1  # set the exploration constant to 1
        Q = self.T / self.N if self.N > 0 else 0  # average reward of the node
        U = c * self.p * np.sqrt(self.parent.N) / (1 + self.N)  # exploration term
        return Q + U  # PUCT score
    
    
    def detach_parent(self):
        # free memory detaching nodes
        del self.parent
        self.parent = None
       
        
    def create_child(self):
        # print("Creating child nodes...")
        if self.game.is_over():
            return
        actions = self.game.get_valid_actions()  # call get_valid_actions on the Game object
        # print("Game state:", self.game.get_state())  # use get_state method to get the game state
        # print("Valid actions:", actions)
        child = {}
        for action in actions:
            game = copy.deepcopy(self.game)
            game.step(action)
            observation = game.get_observation()
            done = game.is_over()
            node = Node(game, done, self, observation, action, self.model)
            node.p = 1 / len(actions)  # set P to 1/(#valid actions)
            child[tuple(action)] = node
        self.child = child
            
    def explore(self):
        # print("Exploring...")
        current = self
        while current.child:
            child = current.child
            # print("Current node:", current)
            # print("Child nodes:", child)
            max_U = max(c.getPUCTscore() for c in child.values())
            actions = [a for a, c in child.items() if c.getPUCTscore() == max_U]
            action = random.choice(actions)
            current = child[action]
        current.N += 1 
        if current.N < 1:
            current.nn_v = current.rollout()
            current.T = current.T + current.nn_v
        else:
            current.create_child()
            if current.child:
                current = random.choice(list(current.child.values()))     
        return current
                
            
    def rollout(self):
        if self.done:
            return 0
        else:
            self.model.eval()  
            with torch.no_grad():
                inputs = self.game.prepare_inputs(self.observation)
                logits = self.model(inputs) 
                logit_honest = logits[0, self.game.label_honest]
                logit_liar = logits[0, self.game.label_liar]
                result = 1 if logit_honest > logit_liar else -1
                print("Rollout result:", result)  # print the result of the rollout
                return result
           
            
    def next(self):
        # print("Child nodes:", self.child)
        # print("Game status:", self.done)
        if self.done:
            raise ValueError("game has ended")
        if not self.child:
            raise ValueError('no children found and game hasn\'t ended')
        child = self.child
        max_N = max(node.N for node in child.values())
        actions = [a for a, node in child.items() if node.N == max_N]
        action = random.choice(actions)
        next_node = child[action]
        return next_node, action, next_node.observation

#GAME   
def play_game(model, test_loader):
   # select random MNIST image and its label
    image, label_honest = random.choice(test_loader.dataset)
    image = image.to(device)

    # initialize
    game = Game(image, label_honest, label_honest)  # liar's label is initially the same as honest's

    # initialize MCTS with Judge and game state
    mcts = Node(game, game.is_over(), None, game.get_observation(), None, model)

    # determine the first player randomly
    game.turn = random.choice([0, 1])

    # liar claims a label
    possible_labels = list(range(10))
    possible_labels.remove(label_honest)
    game.label_liar = random.choice(possible_labels)
    
    print(f"Honest claims: {label_honest}")
    print(f"Liar claims: {game.label_liar}")

    #game loop
    while not game.is_over():
        
        if game.turn == 0:  # liar's turn
            # liar performs 10k rollouts using MCTS and selects the move with the highest PUCT score
            for _ in range(10000):
                mcts.explore()
            next_node, action, _ = mcts.next()
            game.step(action)
            mcts = next_node
        else:  # honest's turn
            # honest player performs 10k rollouts using MCTS and selects the move with the highest PUCT score
            for _ in range(10000):
                mcts.explore()
            next_node, action, _ = mcts.next()
            game.step(action)
            mcts = next_node
            
    # determine the winner
    winner = game.get_winner(model)


    print(f"Judge's choice: {winner}")
    print(f"Game ended. Winner: {winner}")
    return winner

# play the game
play_game(model, train_loader)
