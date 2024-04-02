import torch as t
from sklearn.metrics import f1_score, classification_report
import numpy as np
# from tqdm.autonotebook import tqdm
import warnings
import os
import shutil

warnings.simplefilter("ignore")


class Trainer:
    """
    Trainer class for training the resnet model.
    
    Attributes:
        _model (torch.nn.Module): The model to be trained.
        _crit (torch.nn.Module): The loss function.
        _optim (torch.optim.Optimizer): The optimizer for updating model parameters.
        _train_dl (torch.utils.data.DataLoader): The training data loader.
        _val_test_dl (torch.utils.data.DataLoader): The validation (or test) data loader.
        _cuda (bool): Whether to use GPU for training.
        _early_stopping_patience (int): Patience for early stopping.
        path (str): Path to save checkpoints.
        f1Score (float): F1 score on the validation set computed after each training epoch.
    """
    def __init__(self,
                 model, 
                 crit,  
                 optim,  
                 train_dl, 
                 val_test_dl,  
                 cuda=True,  
                 early_stopping_patience=-1,
                 path='./checkpoints'): 
        """
        Initializes the Trainer.

        Parameters:
            model (torch.nn.Module): The model to be trained.
            crit (torch.nn.Module): The loss function.
            optim (torch.optim.Optimizer): The optimizer for updating model parameters. 
            train_dl (torch.utils.data.DataLoader): The training data loader.
            val_test_dl (torch.utils.data.DataLoader): The validation (or test) data loader. 
            cuda (bool, optional): Whether to use GPU for training. Default is True.
            early_stopping_patience (int, optional): Patience for early stopping. Default is -1.
            path (str, optional): Path to save checkpoints. Default is './checkpoints'.
        """
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience
        self.path = path
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
            print("Existing Directory deleted")
        os.mkdir(self.path)
        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

        self.f1Score = 0
                     
    def save_checkpoint(self, epoch):
        """
        Saves the model checkpoint.

        Parameters:
            epoch (int): Epoch number.
        """
        t.save({'state_dict': self._model.state_dict()}, self.path + '/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        """
        Restore the model from a checkpoint.

        Parameters:
            epoch_n (int): Epoch number.

        """
        ckp = t.load(self.path + '/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        """
        Saves the model in ONNX format.

        Parameters:
            fn (str): Filename for the ONNX model.
        """
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        """
        Performs a single training step.

        Parameters:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.

        Returns:
            float: Loss value.
        """
        # perform following steps:
        # -reset the gradients
        self._optim.zero_grad()
        # -propagate through the network ==> forward pass
        pred = self._model(x)
        # -calculate the loss
        yy = t.squeeze(y).float()
        loss = self._crit(pred, yy)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss.item()

    def train_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            float: Average loss for the epoch.
        """
        # set training mode
        # model.train() tells your model that you are training the model. So effectively layers like dropout, batchnorm etc.
        # which behave different on the train and test procedures know what is going on and hence can behave accordingly.
        self._model = self._model.train()
        loss = 0
        # iterate through the training set
        for img, label in self._train_dl: 
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                img = img.to('cuda')
                label = label.to('cuda')
            else:
                img = img.to('cpu')
                label = label.to('cpu')
            # perform a training step 
            loss = loss + self.train_step(x=img, y=label)

        # calculate the average loss for the epoch and return it
        avg_loss = loss / len(self._train_dl)
        return avg_loss

    def val_test_step(self, x, y):
        """
        Perform a single validation/testing step.

        Parameters:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.

        Returns:
            tuple: Tuple containing loss value and predictions.
        """
        predict = self._model(x)
        # propagate through the network and calculate the loss and predictions
        yy = t.squeeze(y).float()
        loss = self._crit(predict, yy)
        # return the loss and the predictions
        return loss.item(), predict

    def val_test(self):
        """
        Compute average loss for the validation dataset.

        Returns:
            float: Average loss for the validation dataset.
        """
        # turn off the training mode
        self._model = self._model.eval()
        # disable gradient computation
        ''' 
        "torch.tensor.detach()" detaches the output from the computational graph. So no gradient will be backproped along this variable.
        "torch.no_grad()" says that no operation should build the graph.
        '''
        tot_loss = 0
        y_true = None
        y_pred = None
        with t.no_grad():
            # iterate through the validation set
            for i, (img, labels) in enumerate(self._val_test_dl):
                # transfer the batch to the gpu if given
                if self._cuda:
                    images = img.to('cuda')
                    labels = labels.to('cuda')
                else:
                    images = img.to('cpu')
                    labels = labels.to('cpu')
                # perform a validation step
                loss, prediction = self.val_test_step(images, labels)
                tot_loss = tot_loss + loss
                # save the predictions and the labels for each batch
                # because of labels and predictions => torch.squeeze
                # make predictions in 1 or 0
                # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
                if i == 0:
                    y_true = labels
                    y_pred = prediction
                else:
                    y_true = t.cat((y_true, labels), dim=0)
                    y_pred = t.cat((y_pred, prediction), dim=0)

            # https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
            squeeze_pred = t.squeeze(y_pred.cpu().round())

            f1 = f1_score(t.squeeze(y_true.cpu()), squeeze_pred, average='macro')  
            # return the loss and print the calculated metrics
            self.f1Score = f1
            if self.f1Score >= 0.70:
                f = open(self.path + "/results.txt", "a+")
                f.write(f'F1 score on valid_set at epoch{self.cnt_epoch} is {self.f1Score} and valid_loss ='
                        f'{tot_loss / len(self._val_test_dl)}\n')
            return tot_loss / len(self._val_test_dl)  # split = 1600/200 batch and 400/200

    def fit(self, epochs=-1):
        """
        Train the model.
        
        Parameters:
            epochs (int, optional): Number of epochs to train. Default is -1 (train until early stopping).

        Returns:
            tuple: Tuple containing lists of training and validation losses.
        """
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        train_loss = []
        val_loss = []
        patience_count = 0
        self.cnt_epoch = 0
        f1_mean = []

        while True:
            # stop by epoch number
            if self.cnt_epoch == epochs:
                break
            self.cnt_epoch += 1
            # train for a epoch and then calculate the loss and metrics on the validation set
            avg_train_loss = self.train_epoch()
            avg_val_loss = self.val_test()
            f1_mean.append(self.f1Score)
            print('Epoch: {} F1_Score: {} Validation Loss: {}'.format(self.cnt_epoch, self.f1Score, avg_val_loss))
            # append the losses to the respective lists
            train_loss.append(avg_train_loss)
            val_loss.append((self.cnt_epoch, avg_val_loss, self.f1Score))
            if avg_val_loss > 1.04 * val_loss[len(val_loss) - 2][1]:
                patience_count += 1
            if self.f1Score >= 0.70:
                self.save_checkpoint(self.cnt_epoch)
            if patience_count >= self._early_stopping_patience or self.cnt_epoch >= epochs:
                # print('Enough.. I have no Patience, I am STOPPING')
                return train_loss, val_loss
