import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

class Net(Module):   
    def __init__(self, x1, num_classese = 53):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Conv2d(1, (420,580)),
            Conv2d(1, 4, kernel_size=2, stride=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=1),
            # Defining another 2D convolution layer
            Conv2d(4, 8, kernel_size=2, stride=1),
            BatchNorm2d(8),
            # Buy New Ram Can Do
            # Conv2d(8, 16, kernel_size=3, stride=3),
            # BatchNorm2d(16),
            # Dropout(0.25),
            # Conv2d(16, 32, kernel_size=3, stride=3),
            # BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=1),
        )

        test = self.cnn_layers(x1)
        _size = test.size()
        _c = _size[1]
        _h = _size[2]
        _w = _size[3]

        self.linear_layers = Sequential(
            Linear(_c * _h * _w, num_classese),
            Dropout(0.2),
            Softmax(dim=1),
        )
        

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        # print('forward cnn x size:', x.size())
        x = x.view(x.size(0), -1)
        # print('after cnn viewed: ', x.size())
        x = self.linear_layers(x)
        # print('forward result x: ', x)
        return x



class ConNet(Module):   
    def __init__(self):
        super(ConNet, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=3, stride=2),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 16, kernel_size=3, stride=2),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(16 * 25 * 35, 3),
            Softmax(dim=1),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        print('forward x size: ', x.size())
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
