import gym
from gym import error, spaces, utils
import torchvision.transforms as T
import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

from torchvision.utils import save_image

from image_enhancement.envs.actions import select, select_fine

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

size_image_training=224
numbers_of_actions=28

def calculateDistance(i1, i2):
	return torch.mean((i1 - i2) ** 2).item()

def euclideanDistance(i1,i2):
	return torch.dist(i1, i2, 2).item()


class ImageEnhancementEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):

		#da capire come parametrizzare
		self.action_space = spaces.Discrete(numbers_of_actions)
		self.observation_space = spaces.Box(0, 255, [3, size_image_training, size_image_training])


		self.state = None
		self.previus_state= None
		self.target= None
		self.steps=0
		self.initial_distance=None
		self.done=0
		self.startImage = None

		self.startImageRaw=None
		self.finalImage = None
		self.targetRaw = None


		self.initial_distance_RAW=None
		self.final_distance_RAW=None

		self.target_image_RAW_batched=None
		self.final_image_RAW_batched=None

	# print(self.type_distance,type_distance)

	def doStepOriginal(self, actions):
		temp = self.startImageRaw.detach().clone()
		for a in actions:
			temp = select_fine(temp, a)
		self.finalImage = temp.detach().clone()

		self.final_distance_RAW=calculateDistance(self.finalImage,self.targetRaw)
		self.final_image_RAW_batched = self.finalImage.detach().clone().unsqueeze(0)


	def step(self, action):
		assert self.action_space.contains(action)
		self.previus_state=self.state.detach().clone()
		self.steps+=1

		self.state = select_fine(self.state, action)


		distance_previus_state = calculateDistance(self.target, self.previus_state)

		distance_state = calculateDistance(self.target,self.state)
		reward=distance_previus_state-distance_state

		done = 0
		if (reward > 0):
			reward = 1 / (distance_previus_state / (distance_previus_state - distance_state))
		elif (reward < 0):
			reward = -1
		done = 0


		if distance_state > (self.initial_distance):
			done = 1

		self.finalImage=self.state.clone()


		return self.state.clone(), reward, done, distance_state


	def reset(self,raw,target):
		self.done=0

		self.steps=0

		#rawImage=T.functional.resize(raw,size=[size_image_training])
		#expImage=T.functional.resize(target,size=[size_image_training])
		convert_tensor = T.Compose([
			T.Resize(224),
			T.ToTensor(),
		])

		rawImage=convert_tensor(raw)
		expImage=convert_tensor(target)

		self.state=rawImage.detach().clone()
		self.target = expImage.detach().clone()

		self.initial_distance = calculateDistance(self.target, self.state)

		#print(self.state.mean(),self.state.std(),self.state.max())

		convert_tensor = T.ToTensor()

		self.startImage = rawImage.detach().clone()


		self.startImageRaw=convert_tensor(raw)

		self.targetRaw=convert_tensor(target)

		self.initial_distance_RAW=calculateDistance(self.startImageRaw,self.targetRaw)

		self.target_image_RAW_batched=self.targetRaw.detach().clone().unsqueeze(0)

		return self.state

	def render(self):
		plt.imshow(self.state.permute(1,2,0))
		plt.show()

	def multiRender(self):
		imIn = self.startImageRaw.permute(1,2,0)
		imOut = self.finalImage.permute(1,2,0)
		imTarget = self.targetRaw.permute(1,2,0)
		imDiff = (self.startImageRaw - self.finalImage).permute(1,2,0)

		fig = plt.figure(figsize=(4., 4.))
		grid = ImageGrid(fig, 111,  # similar to subplot(111)
						 nrows_ncols=(2, 2),  # creates 2x2 grid of axes
						 axes_pad=0.1,  # pad between axes in inch.
						 )

		for ax, im in zip(grid, [imIn, imOut, imTarget, imDiff]):
			# Iterating over the grid returns the Axes.
			ax.imshow(im)

		plt.show()

	# fig.show()
	def save(self,name):
		#print(self.state)
		#rdner = np.transpose(self.state.numpy(), (1, 2, 0))
		#matplotlib.image.imsave(name+'.png', (cv2.cvtColor(rdner, cv2.COLOR_BGR2RGB)))
		#cv2.imwrite("final.png",cv2.cvtColor(self.state, cv2.COLOR_BGR2RGB))
		save_image(self.finalImage, "FinalImage/" + name)
