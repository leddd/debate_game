import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
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
        predicted_label = torch.argmax(outputs, dim=1).item()
        return "honest" if predicted_label == self.label_honest else "liar"
    
    def clone(self):
        cloned_game = copy.deepcopy(self)
        return cloned_game
   
#GAME   
def play_game(model, train_loader):
    
    # Select a random image from the dataset
    images, labels = next(iter(train_loader))
    random_index = random.randint(0, len(images) - 1)
    image = images[random_index]
    correct_label = labels[random_index]

    # Generate a random incorrect label
    incorrect_label = (correct_label + random.randint(1, 9)) % 10
    
    # Assign the correct label to the honest player and the incorrect label to the liar
    label_honest = correct_label
    label_liar = incorrect_label

    # Initialize the game
    game = Game(image, label_honest, label_liar)

    # Convert the grayscale image to RGB and create a copy for visualization
    visual_image = np.repeat(image.numpy(), 3, axis=0)

    # Function to color a pixel    
    def color_pixel(image, x, y, color):
        image[:, x, y] = 0.5 * image[:, x, y] + 0.5 * np.array(color)

    # Function to select a random non-zero pixel
    def select_random_pixel(game):
        while True:
            x = random.randint(0, game.image.shape[1] - 1)  # height
            y = random.randint(0, game.image.shape[2] - 1)  # width
            if game.image[0, x, y] != 0:
                return x, y

    # Create a figure for the animation
    fig = plt.figure()

    # Function to update the game state and visualize it
    def update(i):
        if not game.is_over():
            action = select_random_pixel(game)
            game.step(action)

            # Update the visual_image
            x, y = action
            if game.turn == 1:  # liar's turn just ended
                color_pixel(visual_image, x, y, [1, 0, 0])  # red pixel
            else:  # honest's turn just ended
                color_pixel(visual_image, x, y, [0, 0, 1])  # blue pixel

            plt.imshow(visual_image.transpose((1, 2, 0)), cmap='gray')

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=range(image.shape[1] * image.shape[2]), interval=1000)
    
    plt.show()

    # Determine the winner
    winner = game.get_winner(model)
    print(f"The winner is the {winner} player.")
    print(f'Honest player claimed label: {label_honest}')
    print(f'Liar player claimed label: {label_liar}')
    print(f'Judge chose label: {winner}')

#####

# Load the MNIST dataset
train_loader, test_loader = load_data()

# Load the trained model
model = Judge()
model.load_state_dict(torch.load('judge.pth'))
model.to(device)
model.eval()

# Play the game
play_game(model, train_loader)
