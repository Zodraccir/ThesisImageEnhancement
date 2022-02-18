import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

'''
class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.features=nn.Sequential(
            nn.Conv2d(input_dims[0], 64, 3, 2, 1),  # in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2),  # kernel_size
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_input_dims, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_actions),
        )

        #self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims=self.features(state)
        #print(dims.view(dims.size()[0], -1).shape)
        return int(np.prod(dims.size()))

    def forward(self,state):
        x=self.features(state)
        #print(x.shape)
        h=x.view(x.shape[0],-1)
        #print(h.shape)
        actions=self.classifier(h)

        #print("ACTION",actions.shape)
        return actions


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save({
        	'model_state_dict':self.state_dict(),
        	'optimizer_state_dict':self.optimizer.state_dict(),
        	'loss':self.loss
        	}
        	, self.checkpoint_file)

    def load_checkpoint(self,learn):
        print('... loading checkpoint ...')
        if(os.path.isfile(self.checkpoint_file)):
            checkpoint=T.load(self.checkpoint_file, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss=checkpoint['loss']
            if(learn):
                print("training mode")
                self.train()
            else:
                print("evaluation mode")
                self.eval()
        else:
        	print("file not exists")
'''


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=n_actions),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
        }
            , self.checkpoint_file)

    def load_checkpoint(self, learn):
        print('... loading checkpoint ...')
        if (os.path.isfile(self.checkpoint_file)):
            checkpoint = T.load(self.checkpoint_file, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss = checkpoint['loss']
            if (learn):
                print("training mode")
                self.train()
            else:
                print("evaluation mode")
                self.eval()
        else:
            print("file not exists")


if __name__ == '__main__':
    # model = torchvision.models.AlexNet()
    model = DeepQNetwork(0.01, 24,
                                    input_dims=[],
                                    name='_q_eval',
                                    chkpt_dir='_q_eval')
    print(model)

    input = T.randn(8, 3, 224, 224)
    out = model(input)
    print(out.shape)
