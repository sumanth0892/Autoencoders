import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary

class MNIST_autoencoder(nn.Module):
	def __init__(self, input_size,output_size):
		super().__init__()
		self.lin1 = nn.Linear(input_size, 128)
		self.lin2 = nn.Linear(128,32)
		self.lin3 = nn.Linear(32,128)
		self.lin4 = nn.Linear(128,output_size)

	def forward(self, x):
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x = F.relu(self.lin3(x))
		x = F.sigmoid(self.lin4(x))
		return x

input_size = 784
model = MNIST_autoencoder(input_size,input_size)
print(summary(model,input_size = (input_size,)))



