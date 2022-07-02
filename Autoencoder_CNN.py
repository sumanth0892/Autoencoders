import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CNN_Autoencoder(nn.Module):
	def __init__(self,input_channels):
		super().__init__()
		self.conv1 = nn.Conv2d(input_channels,32,2,2)
		self.conv2 = nn.Conv2d(32,64,2,2)
		self.conv3 = nn.Conv2d(64,128,3,2)
		self.flatten = nn.Flatten()
		self.lin1 = nn.Linear(1152,10)
		self.lin2 = nn.Linear(10,1152)
		self.unflatten = nn.Unflatten(1,(128,3,3))
		self.deConv1 = nn.ConvTranspose2d(128,64,3,2)
		self.deConv2 = nn.ConvTranspose2d(64,32,2,2)
		self.deConv3 = nn.ConvTranspose2d(32,input_channels,2,2)

	def forward(self,x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = self.flatten(x)
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x = self.unflatten(x)
		x = F.relu(self.deConv1(x))
		x = F.relu(self.deConv2(x))
		x = F.relu(self.deConv3(x))
		return x

if __name__ == '__main__':
	model = CNN_Autoencoder(3)
	print(summary(model,input_size = (3,28,28)))
