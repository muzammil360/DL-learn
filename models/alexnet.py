import torch
import torch.nn.functional as F

class alexnet(torch.nn.Module):
	def __init__(self):
		super(alexnet,self).__init__()

		self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4, padding = 5)
		self.conv2 = torch.nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride = 1, padding = 2)
		self.conv3 = torch.nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1, padding = 1)
		self.conv4 = torch.nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 1, padding = 1)
		self.conv5 = torch.nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)

		self.maxpool = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)

		self.batchnorm1 = torch.nn.BatchNorm2d(num_features = 96 , affine = False)
		self.batchnorm2 = torch.nn.BatchNorm2d(num_features = 256 , affine = False)


		self.fc6 = torch.nn.Linear(in_features = 9216, out_features = 4096, bias=True)
		self.fc7 = torch.nn.Linear(in_features = 4096 , out_features = 4096, bias=True)
		self.fc8 = torch.nn.Linear(in_features = 4096 , out_features = 1000, bias=True)


	def forward(self,x):
		
		# layer 1
		x = self.conv1(x)
		x = F.relu(x)
		x = self.batchnorm1(x)		
		x = self.maxpool(x)
		
		# layer 2
		x = self.conv2(x)
		x = F.relu(x)
		x = self.batchnorm2(x)
		x = self.maxpool(x)		

		# layer 3
		x = self.conv3(x)
		x = F.relu(x)
		
		# layer 4
		x = self.conv4(x)
		x = F.relu(x)
		
		# layer 5
		x = self.conv5(x)
		x = F.relu(x)
		x = self.maxpool(x)
		
		x = x.view(-1,256*6*6)

		# layer 6
		x = self.fc6(x)
		
		# layer 7
		x = self.fc7(x)
		
		# layer 8
		x = self.fc8(x)
		
		return x


