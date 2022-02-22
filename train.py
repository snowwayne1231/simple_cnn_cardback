# importing the libraries
import pandas as pd
import numpy as np
import os, sys, json

# from tqdm import tqdm
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.util import crop

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from nets import Net

def get_train_dataset(dir = 'cards'):
    print('Start Loading Datasets From {}'.format(dir))
    print('  0%', end='\r')
    REISZE_RATIO = 8
    MIN_RESIZE_WIDTH = 200
    cards_dir_path = os.path.join(os.path.dirname(__file__), dir)
    path_jpgs = os.listdir(cards_dir_path)
    imgs = []
    labels = []
    length_jpgs = len(path_jpgs)
    _i = 0
    for jpath in path_jpgs:
        _name = jpath.split('.')[0]
        _type = _name[0]
        _num_string = _name[1:]
        _number = int(_num_string) if _num_string.isnumeric() else _i
        if _number > 13:
            _number = ((_number-1) % 13) + 1
        
        if _type == 'h':
            _number += 13
        elif _type == 'd':
            _number += 26
        elif _type == 'c':
            _number += 39

        _path_file = os.path.join(cards_dir_path, jpath)
        _img = imread(_path_file, as_gray=True)
        _img_shape_0 = int(_img.shape[0]  / REISZE_RATIO)
        _img_shape_1 = int(_img.shape[1]  / REISZE_RATIO)
        if _img_shape_1 < MIN_RESIZE_WIDTH:
            _shape_ratio = int(_img.shape[1] / MIN_RESIZE_WIDTH)
            _img_shape_0 = int(_img.shape[0]  / _shape_ratio)
            _img_shape_1 = int(_img.shape[1]  / _shape_ratio)
        
        if 'bee_ag' in _path_file:
            _img = crop(_img, ((200, 20),), copy=True)
            # save_img(_img, 'test_bee_ag.jpg')
            _number_str = _name[-2:].strip()
            if _number_str.isnumeric():
                _number = int(_number_str)
            else:
                if _number_str == 'J':
                    _number = 11
                elif _number_str == 'Q':
                    _number = 12
                elif _number_str == 'K':
                    _number = 13
                elif _number_str == 'A':
                    _number = 1
                else:
                    print('[Error]Failed Card Parse.')
                    print('_name: ', _name)
                    print('_number_str: ', _number_str)
                    print('_number: ', _number)
                    exit(2)

        _img_resized = resize(_img, (_img_shape_0, _img_shape_1), anti_aliasing=True)
        # print('_name: {} , type: {} , number: {}'.format(_name, _type, _number))
        # print('img : ', _img)
        # _img = _img /255
        # print('_img shape: ', _img.shape)
        # print('_img_resized shape: ', _img_resized.shape)
        # exit(2)

        imgs.append(_img_resized.astype('float32'))
        labels.append(_number)
        _i += 1
        _percent = _i / length_jpgs
        print(' {:.2%} '.format(_percent), end='\r')

    print('100.00%')
    return np.array(imgs), np.array(labels)



def get_tensor_dataset(x,y):
    train_x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    train_x = torch.from_numpy(train_x)

    train_y = torch.from_numpy(y)
    train_y = train_y.type(torch.LongTensor)
    return train_x, train_y



def handle_to_check_jack(model, x, y):
    multiple = 4
    fitler_remain = [_y!=11 for _y in y]
    new_y = y[fitler_remain]
    new_x = x[fitler_remain]

    new_x_m = new_x.repeat(multiple, axis=0)
    new_y_m = new_y.repeat(multiple, axis=0)

    next_x = np.concatenate((new_x_m, x), axis=0)
    next_y = np.concatenate((new_y_m, y), axis=0)

    print('next_x shape: ', next_x.shape)
    print('next_y shape: ', next_y.shape)

    train_x, train_y = get_tensor_dataset(next_x, next_y)
    
    

    _i = 0
    num_corrected = 0
    num_jack = 0
    num_jack_right = 0
    output_train = cnn_net(train_x)
    for o in output_train:
        predicted = torch.argmax(o)
        _ans = next_y[_i]
        if _ans == predicted:
            num_corrected += 1
        
        if _ans == 11:
            num_jack += 1
            if _ans == predicted:
                num_jack_right += 1
        _i += 1

    print('Huge x100 Accuracy: {:.2f}'.format(num_corrected / _i))
    print('Jack Num: {} Jack Corrected: {} , Jack Accuracy: {:.2f}'.format(num_jack, num_jack_right, num_jack/num_jack_right if num_jack_right > 0 else 0))
    


def save_img(data, name):
    output_dir_path = os.path.join(os.path.dirname(__file__), 'outputs')
    if not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)
    
    file_name_path = os.path.join(output_dir_path, name)
    name_dir = os.path.dirname(file_name_path)
    if not os.path.isdir(name_dir):
        os.makedirs(name_dir)

    imsave(file_name_path, img_as_ubyte(data))



def train(epoch, model, x, y, criterion):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(x), Variable(y)
    # getting the validation set
    x_val, y_val = Variable(x), Variable(y)
    # converting the data into GPU format

    # clearing the Gradients of the model parameters
    
    
    # prediction for training and validation set
    output_train = model(x_train)
    # print('output_train: ', output_train[0])
    # print('y_train: ', y_train[0])
    output_val = model(x_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    return loss_train, loss_val


def save_result(data, path = 'result'):
    with open(os.path.join(os.path.dirname(__file__), 'results', '{}.json'.format(path)), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    # x, y = get_train_dataset()
    AVG_TIMES = 8
    
    cmd_parameters = sys.argv[1:]
    _dir_path = cmd_parameters[0]
    x, y = get_train_dataset(_dir_path)

    print('x shape: ', x.shape)
    print('y shape: ', y.shape)
    # y = np.vectorize(lambda _: (_-1)%13)
    y = np.array([(_-1)%13 for _ in y])

    train_x, train_y = get_tensor_dataset(x, y)

    # defining the model
    cnn_net = Net(train_x, 13)
    # cnn_net = ConNet()
    # defining the loss function
    optimizer = Adam(cnn_net.parameters(), lr=0.0001)
    criterion = CrossEntropyLoss()

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    
        
    print(cnn_net)

    times = 0
    n_epochs = 24
    accuracies = []
    train_losses_sets = []
    rightMaps = []

    while (times < AVG_TIMES):
        times += 1
        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            optimizer.step()
            optimizer.zero_grad()
            tloss, vloss = train(epoch, cnn_net, train_x, train_y, criterion)

            train_losses.append(tloss.item())
            val_losses.append(vloss)
            # computing the updated weights of all the model parameters
            tloss.backward()
            
            # tr_loss = tloss.item()
            print('Epoch : ',epoch, '\t', 'loss :', vloss)

        _i = 0
        num_corrected = 0
        output_train = cnn_net(train_x)
        martix = np.zeros((14,14))
        rightMap = {}
        
        for _idx in range(14):
            martix[0][_idx] = _idx
            martix[_idx][0] = _idx
        
        for oo in output_train:
            predicted = torch.argmax(oo)
            _ans = y[_i]
            if _ans == predicted:
                num_corrected += 1
                num_ans = int(_ans+1)
                martix[num_ans][predicted+1] += 1
                if rightMap.get(num_ans):
                    rightMap[num_ans] += 1
                else:
                    rightMap[num_ans] = 1
            else:
                # print('Wrong Predicted: {} | Answer: {}'.format(predicted, _ans))
                save_img(x[_i], '{}/_ans_{}_pred_{}.jpg'.format(_dir_path, _ans, predicted))
                martix[_ans+1][predicted+1] += 1
            _i += 1

        accuracy = num_corrected / _i
       
        print('Basic Accuracy: {:.2f}'.format(accuracy))
        print(martix)
        accuracies.append(accuracy)
        train_losses_sets.append(train_losses)
        rightMaps.append(rightMap)

    # print('train_losses_sets: ', train_losses_sets)

    save_result({
        'BasicAccuracies': accuracies,
        'DirPath': _dir_path,
        'rightMaps': rightMaps,
        'TrainLossesSets': train_losses_sets,
    }, _dir_path.split('\\')[-1])
    # handle_to_check_jack(cnn_net, x, y)
        

    
    
    
