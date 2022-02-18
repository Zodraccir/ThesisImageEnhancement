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
numbers_of_actions=29

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

		self.total_reward=0
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
		#self.state=performAction(action,self.state)
		distances=[]
		#print(self.initial_distance)
		for a in range(0,self.action_space.n-1):
			#print("doing action",a)
			tmp_state=select_fine(self.state,int(a))
			distances.append(calculateDistance(self.target,tmp_state))
		distance_previus_state = calculateDistance(self.target, self.previus_state)
		#print(distances)
		distances.sort()
		max = distances[0]  # max value in sense of minimum distance from targer
		min = distances[-1]  # min value in sense of maximum distance from targer
		#print(max,min)
		#print(distances)
		self.state=select_fine(self.state,action)
		distance_state = calculateDistance(self.target,self.state)
		reward=0
		#print(min, max)
		#print("dist-stat",distance_state)
		done = 0
		if distance_state>distance_previus_state:
			#print("lesser then previus")
			reward=-1
			done=1
		elif distance_state<distance_previus_state:
			#print("more then previus")
			#reward=1-((distance_state-max)/(distance_previus_state-max)) -1 #if reward is 0, best action
			reward=1/(self.initial_distance / (distance_previus_state-distance_state))
		elif distance_state==distance_previus_state:
			#print("equal")
			reward=0
		upgrade = (1 - (distance_state / self.initial_distance))
		#print(self.initial_distance, distance_state, distance_previus_state,(distance_previus_state-distance_state),"reward", reward, upgrade)
		if(upgrade<=0.8):
			upgrade=-1
		if action==28:
			done=1
			reward=(upgrade-0.8)/(1-0.8)
		#print(action, reward)
		#print(upgrade,reward,done,distance_previus_state,distance_state,max)
		'''
		if(reward>=0.8):
			reward=1
		elif(reward<0.8 and reward > -0.1):
			reward=0
		elif(reward <= 0.1):
			reward=-1
		#print("rewad!!!",reward)
		'''
		#reward = distance_previus_state-distance_state
		#threshold=0.00001
		#distance_from_previus=calculateDistance(self.previus_state,self.state)
    	#if(reward>0):

			#splits=distance_previus_state/10

			#for i in range(1,10):
				#if(reward<i*splits):
					#reward=0.1*i
					#break

			#print(reward)

		#elif(reward<0):
			#reward=-1



		'''

		if abs(distance_state.item())<threshold:
			done=1
			print("Passsaggi effettuati correttamente")

		if distance_state.item()>(self.initial_distance+0.2*self.initial_distance):
			done=11
			#print("Limite sforato")
		if self.steps>15:
			done=11
			#print("Max operazioni effettuate")
		'''
		#print(reward_state.item())
		#print(reward)

		#process reward, if

		self.finalImage=self.state.clone()

		'''if self.steps > 20:
			done = 11'''
		# print("Max operazioni effettuate")
		self.total_reward=self.total_reward+reward

		#if(reward<-0.5):
			#print("passaggio sbagliato")
			#done=1

		return self.state.clone(), reward, done, distance_state


	def reset(self,raw,target):
		self.done=0

		self.steps=0

		#rawImage=T.functional.resize(raw,size=[size_image_training])
		#expImage=T.functional.resize(target,size=[size_image_training])

		rawImage=raw
		expImage=target

		self.state=rawImage.detach().clone()
		self.target = expImage.detach().clone()

		self.initial_distance = calculateDistance(self.target, self.state)

		#print(self.state.mean(),self.state.std(),self.state.max())


		self.startImage = rawImage.detach().clone()
		self.startImageRaw=raw.detach().clone()

		self.targetRaw=target.detach().clone()

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
