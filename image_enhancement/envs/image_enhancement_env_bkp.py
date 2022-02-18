import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torchvision.transforms as T
import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid




class Act():
    def gamma_corr(image, gamma, channel=None):
        mod = image.clone()
        if channel is not None:
            mod[:, channel, :, :] = mod[:, channel, :, :] ** gamma
        else:
            mod = mod ** gamma
        return mod

    def brightness(image, bright, channel=None):
        mod = image.clone()
        if channel is not None:
            mod[:, channel, :, :] = torch.clamp(mod[:, channel, :, :] + bright, 0, 1)
        else:
            mod = torch.clamp(mod + bright, 0, 1)

        return mod

    def contrast(image, alpha, channel=None):
        mod = image.clone()
        if channel is not None:
            mod[:, channel, :, :] = torch.clamp(
                torch.mean(mod[:, channel, :, :]) + alpha * (mod[:, channel, :, :] - torch.mean(mod[:, channel, :, :])),
                0, 1)
        else:
            mod = torch.clamp(torch.mean(mod) + alpha * (mod - torch.mean(mod)), 0, 1)
        return mod
	
def calculateDistance(i1, i2,type_distance=0):

	if(type_distance==0):
		return torch.mean((i1 - i2) ** 2)
	elif(type_distance==1):
		return torch.std((i1 - i2) ** 2)
	elif(type_distance==2):
		return torch.sum((i1-i2) **2)
	elif(type_distance==3):
		return torch.sqrt(torch.sum((i1-i2) **2 ))
	elif(type_distance==4):
		return torch.dist(i1,i2,1)
	elif(type_distance==5):
		return torch.dist(i1,i2,2)
	else:
		return 0
	
	#return torch.dist(i1,i2,p)

def performAction(action,img):
	temp_state = img.unsqueeze_(0)
	'''
	#print(action)
	if(action==0):
		return Act.brightness(temp_state, 0.08).squeeze()
	elif(action==1):
		return Act.brightness(temp_state, -0.08).squeeze()
	elif(action==2):
		return Act.contrast(temp_state, 0.8).squeeze()
	elif(action==3):
		return Act.contrast(temp_state, 1.2).squeeze()
	elif(action==4):
		return Act.gamma_corr(temp_state,1/1.33).squeeze()
	elif(action==5):
		return Act.gamma_corr(temp_state,1.33).squeeze()
	else:
		print(action)
		
	'''
	if (action < 6):
		act = action
		if (act < 3):
			# print("Action taken brightness positive in channel, ",int(act)," action",action)
			return Act.brightness(temp_state, 0.02, int(act)).squeeze()
		else:
			act = act - 3
			# print("Action taken brightness negative in channel, ",int(act)," action",action)
			return Act.brightness(temp_state, -0.02, int(act)).squeeze()
	if(action==6):
		return Act.brightness(temp_state, 0.08).squeeze()
	elif(action==7):
		return Act.brightness(temp_state, -0.08).squeeze()
	elif(action==8):
		return Act.contrast(temp_state, 0.8).squeeze()
	elif(action==9):
		return Act.contrast(temp_state, 1.2).squeeze()
	elif(action==10):
		return Act.gamma_corr(temp_state,1/1.33).squeeze()
	elif(action==11):
		return Act.gamma_corr(temp_state,1.33).squeeze()
	else:
		print(action)
	
	'''

	elif (action < 12):
		act = action - 6
		# print("*1", act)
		if (act < 3):
			# print("Action taken gamma positive in channel, ",int(act)," action",action)
			return Act.gamma_corr(temp_state, 1 / 1.33, int(act)).squeeze()
		else:
			act = act - 3
			# print("Action taken gamma negative in channel, ",int(act)," action",action)
			return Act.gamma_corr(temp_state, 1.33, int(act)).squeeze()
	elif (action < 18):
		act = action - 12
		# print("*1", act)
		if (act < 3):
			# print("Action taken contrast positive in channel, ",int(act)," action",action)
			return Act.contrast(temp_state, 0.8, int(act)).squeeze()
		else:
			act = act - 3
			# print("Action taken contrast negative in channel, ",int(act)," action",action)
			return Act.contrast(temp_state, 1.2, int(act)).squeeze()
	else:
		print(action)
	'''
	

class ImageEnhancementEnv(gym.Env):
	metadata = {'render.modes': ['human']}



	def __init__(self):

		#da capire come parametrizzare
		self.action_space = spaces.Discrete(12)
		self.observation_space = spaces.Box(0, 255, [3, 64, 64])
		self.type_distance=None


		self.state = None
		self.previus_state= None
		self.target= None
		self.steps=0
		self.initial_distance=None
		self.done=0
		self.startImage = None


	def setTypeDistance(self,type_distance):
		self.type_distance=type_distance
		#print(self.type_distance,type_distance)


	def step(self, action):
		assert self.action_space.contains(action)
		self.previus_state=self.state.detach().clone()
		self.steps+=1
		self.state=performAction(action,self.state)
		distance_state = calculateDistance(self.target,self.state,self.type_distance)
		distance_previus_state= calculateDistance(self.target,self.previus_state,self.type_distance)
		reward = distance_previus_state-distance_state
		threshold=0.0000001

		done=0

		if abs(distance_state.item())<threshold:
			done=1
			print("Passsaggi effettuati correttamente")

		if distance_state.item()>self.initial_distance+0.2*self.initial_distance:
			done=1
			print("Limite sforato")
		if self.steps>50:
			done=1
			#print("Max operazioni effettuate")

		#print(reward_state.item())
		#print(reward)

		return self.state.clone(), reward, done, distance_state


	def reset(self,raw,target):
		self.done=0

		transform = T.Compose([T.ToTensor()])

		self.steps=0
		#file=random.choice(os.listdir("rawTest"))

		#img_path_raw = "rawTest/"+file
		
		#print("img_path",img_path_raw)

		#img = cv2.imread(img_path_raw)
		img = cv2.resize(raw, (64, 64), interpolation = cv2.INTER_AREA)

		rawImage = transform(img)

		#img_path_exp = "ExpTest/"+file


		#img = cv2.imread(img_path_exp)
		img = cv2.resize(target, (64, 64), interpolation = cv2.INTER_AREA)
		expImage = transform(img)

		
		self.target=expImage.detach().clone()

		self.state=rawImage.detach().clone()

		self.startImage=rawImage.detach().clone()

		self.initial_distance=calculateDistance(self.target,self.state,self.type_distance)
		
		
		return self.state

	def render(self):
		rdner=np.transpose(self.state.numpy(),(1,2,0))
		plt.imshow(cv2.cvtColor(rdner, cv2.COLOR_BGR2RGB))
		plt.show()

	def multiRender(self):
		imIn = np.transpose(self.startImage.numpy(), (1, 2, 0))
		imOut = np.transpose(self.state.numpy(), (1, 2, 0))
		imTarget = np.transpose(self.target.numpy(), (1, 2, 0))
		imDiff = np.transpose((self.startImage - self.state).numpy(), (1, 2, 0))

		fig = plt.figure(figsize=(4., 4.))
		grid = ImageGrid(fig, 111,  # similar to subplot(111)
						 nrows_ncols=(2, 2),  # creates 2x2 grid of axes
						 axes_pad=0.1,  # pad between axes in inch.
						 )

		for ax, im in zip(grid, [imIn, imOut, imTarget, imDiff]):
			# Iterating over the grid returns the Axes.
			ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

		plt.show()

	# fig.show()
	def save(self,name):
		#print(self.state)
		rdner = np.transpose(self.state.numpy(), (1, 2, 0))
		matplotlib.image.imsave(name+'.png', (cv2.cvtColor(rdner, cv2.COLOR_BGR2RGB)))
		#cv2.imwrite("final.png",cv2.cvtColor(self.state, cv2.COLOR_BGR2RGB))
