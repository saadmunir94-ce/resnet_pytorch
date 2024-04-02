pyimport torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import torchvision as tv
import os
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import operator

import warnings

warnings.simplefilter("ignore")

# load all of the data from the csv file
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        if name == 'data.csv':
            csv_path = os.path.join(root, name)
dataFrame = pd.read_csv(csv_path, sep=';')

# and perform a train-test-split
train, validation = train_test_split(dataFrame, test_size=0.10, random_state=42)
print(train.shape, validation.shape)
# set up data loading for the training and validation
df_train = ChallengeDataset(train, 'train')
df_val = ChallengeDataset(validation, 'val')
opti = [False]
Epochs = 220
# Epochs=1
BATCH = [32]
LR = [0.001]
is_dropout = True
for is_adam in opti:
    for Batch_size in BATCH:
        for learning_rate in LR:

            # formulate the training and validation data loaders
            train_DL = t.utils.data.DataLoader(df_train, batch_size=Batch_size, shuffle=True)
            val_DL = t.utils.data.DataLoader(df_val, batch_size=Batch_size)
            # create an instance of our ResNet model
            resnet = model.ResNet()
            # set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
            criteria = t.nn.BCELoss()
            path = f'/For_{Batch_size}_{learning_rate}'
            # Create the target directories for storing checkpoints if they do not exist
            if not os.path.isdir('./Adam'):
                os.mkdir('./Adam')
            if not os.path.isdir('./Final_Models'):
                os.mkdir('./Final_Models')
            if is_adam:
                path = './Adam' + path
                # setup adam optimizer
                optimizer = t.optim.Adam(resnet.parameters(), lr=learning_rate, weight_decay=0.135)
            else:
                path = './Final_Models' + path
                # setup SGD optimizer
                optimizer = t.optim.SGD(resnet.parameters(), lr=learning_rate, momentum=0.9) 
                
            # create an object of type Trainer and set its early stopping criterion
            trained_model = Trainer(model=resnet, crit=criteria, optim=optimizer, train_dl=train_DL, val_test_dl=val_DL,
                                    cuda=True, early_stopping_patience=25, path=path)

            # call fit on trainer
            train_loss_list, valid_list = trained_model.fit(epochs=Epochs)
            valid_loss_list = [i[1] for i in valid_list]
            res = train_loss_list, valid_loss_list
            # plot the training and validation loss curves
            plt.figure()
            plt.plot(np.arange(len(res[0])), res[0], label='train loss')
            plt.plot(np.arange(len(res[1])), res[1], label='val loss')
            plt.yscale('log')
            plt.legend()
            plt.grid()
            plt.savefig(path + f'/losses_{Batch_size}_{learning_rate}.png')
            plt.close()
            # Retrieve the best result with the highest F1 score on the validation set, restore its checkpoint
            # and save it as onnx model
            valid_list.sort(key=operator.itemgetter(2), reverse=True)
            epoch_count, valid_loss, f1_score = valid_list[0]
            f = open(path + "/best_results.txt", "w+")
            f.write(f'F1 score on the best model is {f1_score} with valid_loss = {valid_loss} and'
                    f' epoch of {epoch_count} \n Batch size = {Batch_size} and learning rate = {learning_rate}')
            f.close()

            if any(fname.endswith(".ckp") for fname in os.listdir(path)):
                trained_model.restore_checkpoint(epoch_count)
                trained_model.save_onnx(path + '/best_checkpoint_{:03d}.onnx'.format(epoch_count))
