# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:10:23 2022

@author: antia
"""

# DEFINE PATHS TO DATA 
path = './results/'
path_ws1 = '../data/WS1_preprocessed_multiclass.pkl'
path_ws2 = '../data/WS2_preprocessed_multiclass.pkl'
path_ws3 = '../data/WS3_preprocessed_multiclass.pkl'
path_ws4 = '../data/WS4_preprocessed_multiclass.pkl'

# IMPORT LIBRARIES 
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import time
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, auc,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,  OneHotEncoder, label_binarize
from sklearn.model_selection import ParameterGrid
from scipy import signal
import argparse
from itertools import cycle

#%% FUNCTIONS 
def subsam(sig, length):
    subsampled = signal.resample(x=sig[:length], num=2000)
    return subsampled

def normalize(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    norm_data = scaler.transform(X)
    return norm_data
    
def plot_roc_curve(fper, tper, auc, title=''):
    plt.figure()
    plt.plot(fper, tper, color='red', label='ROC curve and AUC = %0.3f' %auc)
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve'+ title)
    plt.legend()
    plt.show()
    
    
def save_roc_multiclass(y_test, y_score, name, n_classes = 4, path = path):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
        # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    ax.plot(fpr["micro"],tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    
    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    
    colors = cycle(["aqua", "darkorange", "cornflowerblue","green"])
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )
    
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Some extension of Receiver operating characteristic to multiclass")
    ax.legend(loc="lower right")
    fig.savefig(path + name + '.png')   
    plt.close(fig)


def save_roc_curve(fper, tper, auc, path , title='', name=''):
    fig, ax = plt.subplots( nrows=1, ncols=1 )  
    ax.plot(fper, tper, color='red', label='ROC curve and AUC = %0.3f' %auc)
    ax.plot([0, 1], [0, 1], color='green', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic Curve'+ title)
    ax.legend()
    fig.savefig(path + name + '.png')   
    plt.close(fig)
      
def loss_curve(loss_tr, loss_val, name, path):
    fig, ax = plt.subplots( nrows=1, ncols=1 )  
    ax.plot(loss_tr, label='Training set')
    ax.plot(loss_valid, label='Validation set')
    ax.set_title('Training and Validation Loss')
    fig.savefig(path +name+ '.png')   
    plt.close(fig)

def get_auc(model, x, xref, feat, Y, path_plot, title='', name=''):

    probs = torch.exp(model.forward(torch.Tensor(x).to(model.device),torch.Tensor(xref).to(model.device),torch.Tensor(feat.values)).to(model.device))
    probs = probs.detach().numpy()
    y = Y
    auc = roc_auc_score(y, probs[:,1])
    fper, tper, thresholds = roc_curve(y, probs[:,1])
    save_roc_curve(fper, tper, auc, path_plot, title, name)
    y_pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y, y_pred)
    return auc,cm  
  
def preprocess_pickle(path, features = False):
    data = pd.read_pickle(path)
    # We also extract the values of the reference we are going to include in the model
    xref = data[data['Label']=='eje sano']
    xref = xref.sample(frac=0.304)
    xref = xref['Subsampled']
      
    # remove this values of xref from the data
    data = data.drop(xref.index)
    
    # not use time values
    #xref = list(map(lambda x: np.array([r[1] for r in x]), xref.values))
    xref = np.stack(xref).astype(None)
    
    X = np.stack(data['Subsampled'].values).astype(None)
    feat = data[['Lado','Direction','Corte','Load','Velocidad']]
    Y = data['Label']

    if features:
        return X, Y, feat, xref
    else:
        return X, Y
    
def preprocess_pickle_configuration(path, lado, direction, corte, carga, velocidad):
    data = pd.read_pickle(path)
    data['Load'] = data['Load'].round()

    # NOW FILTER FOR SPECIFIC CONFIGURATION
    data = data[(data['Lado']==lado) & (data['Direction']==direction) & (data['Load']==carga) 
                & (data['Corte']==corte) & (data['Velocidad']==velocidad)]
    
    
    # We also extract the values of the reference we are going to include in the model
    # XREF AFTER FILTERING FOR THE SPECIFIC CONFIGURATION??????? CREO Q SI, TEN MAS SENTIDO NO? 
    xref = data[data['Label']=='eje sano']
    xref = xref.sample(frac=0.304)
    xref = xref['Subsampled']
      
    # remove this values of xref from the data
    data = data.drop(xref.index)
    
    # not use time values
    #xref = list(map(lambda x: np.array([r[1] for r in x]), xref.values))
    xref = np.stack(xref).astype(None)
    
    X = np.stack(data['Subsampled'].values).astype(None)
    feat = data[['Lado','Direction','Corte','Load','Velocidad']]
    Y = data['Label']
    
    return X, Y, feat, xref
    
def dataloader(X, Y, features, fraction=0.2, val=False):
    y_bin = label_binarize(Y, classes=['eje sano', 'd1', 'd2','d3'])
    x_tr, x_test, y_tr, y_test, feat_tr, feat_test = train_test_split(X, y_bin, features, test_size=fraction, random_state=42)
    x_tr, x_val, y_tr, y_val, feat_tr, feat_val = train_test_split(x_tr, y_tr, feat_tr, train_size=0.7, random_state=42)
    
    
    
    # Trainloader creation
    tensor_x = torch.Tensor(x_tr)
    tensor_y = torch.Tensor(y_tr)
    tensor_f = torch.Tensor(feat_tr.values)
    
    trainset = TensorDataset(tensor_x, tensor_f, tensor_y) 
    trainloader = DataLoader(trainset, batch_size = 20, shuffle = True)
    
    
    # Validloader creation
    tensor_x_val = torch.Tensor(x_val)
    tensor_y_val = torch.Tensor(y_val)
    tensor_f_val = torch.Tensor(feat_val.values)
    
    validset = TensorDataset(tensor_x_val, tensor_f_val, tensor_y_val) 
    validloader = DataLoader(validset, batch_size = 20, shuffle = True)

    
    # Testloader creation
    tensor_x_test = torch.Tensor(x_test)
    tensor_y_test = torch.Tensor(y_test)
    tensor_f_test = torch.Tensor(feat_test.values)
    
    testset = TensorDataset(tensor_x_test, tensor_f_test,tensor_y_test) 
    testloader = DataLoader(testset, batch_size = 20, shuffle = True)
    
    
    if val==True:
        return trainloader, validloader, testloader, x_tr, y_tr, feat_tr, x_val, y_val, feat_val, x_test, y_test, feat_test
    else:
        return trainloader, validloader, testloader

# DEFINE  ARGUMENTS TO CROSS VALIDATE
parser = argparse.ArgumentParser(description='descripción del script')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='algo algo algo')
parser.add_argument('--epochs', type=int, default=5,
                    help='easdkf')
parser.add_argument('--drop_prob', type=float, default=0.3,
                    help='sdfasdfas') # esto es para la lstm
parser.add_argument('--dropout', type=float, default=0.4,
                    help='sdfasdfas') #esto es para el mlp
parser.add_argument('--hidden', type=int, default=10,
                    help='sdfasdfas') #esto es para el mlp
parser.add_argument('--train', type=bool, default=False,
                    help='sdfasdfas') #esto es para saber si hay que entrenar o testear
args = parser.parse_args()

print(args)


#%%  DESIGN CNN + LSTM: 6 LAYERS, BINARY CLASSIFICATION, XREF, MLP STATIC FEATURES 
class CNN_LSTM_MLP(nn.Module):
    
    def __init__(self, in_channels, out_channels, out_channels_cnn2, out_channels_cnn3,
                 out_channels_cnn4, out_channels_cnn5, out_channels_cnn6,
                 kernel_size, kernel_maxpool, nlabels,
                 input_size, hidden_dim, n_layers, sigma,
                 sequence_length, use_batch_norm, in_features, out_features,
                 batch_size, batch_size_ref, epochs, drop_prob, dropout, lr):
        
        super().__init__() 
        
        # First: 1-dimension CNN with relu activation 
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding ='same')
        
        self.conv2 = nn.Conv1d(out_channels, out_channels_cnn2, kernel_size, padding ='same')
        
        self.conv3 = nn.Conv1d(out_channels_cnn2, out_channels_cnn3, kernel_size, padding ='same')

        self.conv4 = nn.Conv1d(out_channels_cnn3, out_channels_cnn4, kernel_size, padding ='same')
        
        self.conv5 = nn.Conv1d(out_channels_cnn4, out_channels_cnn5, kernel_size, padding ='same')
        
        self.conv6 = nn.Conv1d(out_channels_cnn5, out_channels_cnn6, kernel_size, padding ='same')

        self.relu = nn.ReLU()
        
        self.kernel = kernel_size
        
        self.kernel_maxpool = kernel_maxpool
        
        
        # Second: LSTM 
        
        self.hidden_dim = hidden_dim
        
        self.input_size = input_size
        
        self.sigma = sigma

        # define an RNN with specified parameters
        # batch_first=True means that the first dimension of the input will be the batch_size
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        
        self.use_batch_norm = use_batch_norm
       
        if self.use_batch_norm:

            self.batch_norm1 = nn.BatchNorm1d(n_layers*hidden_dim)
        
        
        # Third: MLP binary classifier with sigmoid activation 
        self.fc1 = nn.Linear(n_layers*hidden_dim*2 + out_features, nlabels)
        
        # FOURTH: MLP FOR THE FEATURES
        self.in_features = in_features
        self.out_features = out_features
        
        self.fc2 = nn.Linear(in_features,out_features)
        
        #other params
        self.sigmoid = nn.Sigmoid() # BINARY OUTPUT
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.pool = nn.MaxPool1d(kernel_size = kernel_maxpool)
        
        self.num_layers = n_layers
        
        self.lr = lr #Learning Rate
        
        self.epochs = epochs
        
        self.sequence_length = sequence_length
        
        self.optim = optim.Adam(self.parameters(), self.lr)
        
        #self.criterion = nn.BCEWithLogitsLoss() #   BINARY OUTPUT
        
        self.criterion = nn.NLLLoss() 
        
        self.batch_size = batch_size
        
        # A list to store the loss evolution along training
        
        self.loss_during_training = [] 
        
        self.valid_loss_during_training = []

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.to(self.device)
        
    def forward(self, x, xref, f, sigma=1, h0=None, valid=False):
        
        
        # If we use stacked LSTMs, we have to control the evaluation mode due to the dropout between LSTMs
        if(valid):
            self.eval()
        else:
            self.train()
        
        batch_size = x.size(0) # Number of signals N
        seq_length = x.size(1) # T
        
        # --------PREPROCESS THE SIGNALS--------------
        # 1.PASS THE INPUT SIGNAL THROUGH CNN
        
        x = x.reshape((batch_size,1,seq_length))
        #print('x input', x.shape)
        # 1st convolution
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        #print('OUTPUT OF CONV1 + pool LAYER SHAPE ', x.shape)
        
        # 2nd convolution
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        #print('OUTPUT OF CONV2 + pool LAYER SHAPE ', x.shape)
        
        # 3rd convolution
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        #print('OUTPUT OF CONV3 LAYER SIGNAL SHAPE ', x.shape)
        
        # 4th convolution
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        #print('OUTPUT OF CONV4 LAYER SIGNAL SHAPE ', x.shape)
        
        # 5th convolution
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)
        #print('OUTPUT OF CONV5 LAYER SIGNAL SHAPE ', x.shape)
        
        # 6th convolution
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)
        #print('output of con6 ', x.shape)
        
        # 2. PASS THE OUTPUT OF THE CNN THROUGH LSTM 
        
        x = x.reshape((batch_size,x.shape[2],x.shape[1]))
        #print(x.shape)
        
        r_out, (hN, cN) = self.lstm(x, h0) #destgos quero o ultimo solo
        #print('OUTPUT OF LSTM LAYER SHAPE (LAST STATE SHAPE) ', hN.shape)
        hN = hN.permute((1,0,2))
        #print('after transp',hN.shape)
        hN = hN.reshape(hN.shape[0],-1)
        #r out transpose 1,0,3
        #hago shape de 20,-1
        
        #  OBTAIN THE OUTPUT OF THE DENSE LAYER (CLASSIFICATION WITH LAST STATE)
        
        #hacer reshape para quitar 1 quedarnos con 2 dim
        # LA PONEMOS EN SHAPE [BATCH_SIZE, HIDDEN DIM]
        #hN = hN.reshape(hN.shape[1],hN.shape[2]) #ESTO FUNCIONA PARA 1 LAYER
        #hN = hN.reshape(-1, self.hidden_dim) #ESTO TB FUNCIONA PARA 1 LAYER
        #print('hn after reshaping', hN.shape)
        
        # INCLUDE BATCH NORMALIZATION: INMPROVES GENERALIZATION
        if self.use_batch_norm and hN.shape[0]>1:
            hN = self.batch_norm1(hN)
        #print('output of batch norm', hN.shape)  
        
        # DROPOUT LAYER: ADD REGULARIZATION 
        hN = self.dropout(hN)
        #print('input SIGNAL to the FC layer', hN.shape)
        
        # REPEAT THE SAME FOR THE XREF 
        # xref = torch.Tensor(xref)
        batch_size_ref = xref.size(0) # Number of signals N
        seq_length = xref.size(1)
        
        #print('input xref shape',xref.shape)
        xref = xref.reshape((batch_size_ref,1,seq_length))
        #print('xref after reshaping', xref.shape)
        xref = self.conv1(xref)
        xref = self.relu(xref)
        xref = self.pool(xref)
        #print('xref after conv1', xref.shape)
        xref = self.conv2(xref)
        xref = self.relu(xref)
        xref = self.pool(xref)
        #print('xref after conv2', xref.shape)
        xref = self.conv3(xref)
        xref = self.relu(xref)
        xref = self.pool(xref)
        #print('xref after conv3', xref.shape)
        xref = self.conv4(xref)
        xref = self.relu(xref)
        xref = self.pool(xref)
        #print('xref after conv4', xref.shape)
        xref = self.conv5(xref)
        xref = self.relu(xref)
        xref = self.pool(xref)
        #print('xref after conv5', xref.shape)
        xref = self.conv6(xref)
        xref = self.relu(xref)
        xref = self.pool(xref)
        #print('xref after conv6', xref.shape)
        
        # NOW FOR THE LSTM (xref)
        xref = xref.reshape(batch_size_ref,xref.shape[2],xref.shape[1])
        r_out_ref, (hN_ref, cN_ref) = self.lstm(xref, h0) 
        hN_ref = hN_ref.permute((1,0,2))
        hN_ref = hN_ref.reshape(hN_ref.shape[0],-1)
        if self.use_batch_norm and hN_ref.shape[0]>1:
            hN_ref = self.batch_norm1(hN_ref)
        hN_ref = torch.mean(hN_ref,0, True)
        hN_ref = hN_ref.reshape(1,-1)
        hN_ref = hN_ref.repeat(batch_size,1)
        
        # 3. PASS THE FEATURES THROUGH A MLP LAYER
        #print('INPUT FEATURES SHAPE', f.shape)
        f = self.fc2(f)
        #print('OUTPUT FC2 FEATURES SHAPE', f.shape)
        f = self.relu(f)
        #print('OUTPUT FEATURES OF FC2 AND RELU SHAPE', f.shape)
        f = self.dropout(f)
        
        # 4. JOIN SIGNAL AND FEATURES TO OUTPUT CLASSIFICATION RESULT
        #print('shape of the signals',hN.shape)
        #print('shape of the features',f.shape)
        z = torch.cat((hN,f, hN_ref), 1)
        #print('CONCATENATED SIGNAL + FEATURES SHAPE (input to fc layer)',z.shape)
        
        output = self.fc1(z) # QUEREMOS SOLO EL ULTIMO ESTADO PARA PREDECIR LABEL 
        #print('OUTPUT OF FINAL FULLY CONV LAYER SHAPE ', output.shape)
        
        output = self.logsoftmax(output) # BINARY OUTPUT
        #print('OUTPUT OF LOGSOFTMAX SHAPE ', output.shape)
        
        return output
           
    def trainloop(self,trainloader, validloader, xrefloader):
        
        # set model back to train mode
        self.train()
        
        # Optimization Loop
        
        for e in range(int(self.epochs)):

            start_time = time.time()
            
            # Random data permutation at each epoch
            
            running_loss = 0.
            
            for ((signals, features, labels), xref) in zip(trainloader, xrefloader):  
                #print(labels)
        
                self.optim.zero_grad()  #TO RESET GRADIENTS!
                xref = xref[0]

                labels = labels.type(torch.LongTensor)
                signals, features, labels, xref = signals.to(self.device), features.to(self.device), labels.to(self.device), xref.to(self.device) 
                
                #signals = signals[:,:self.sequence_length] # COGER SOLO 100 OBSERVATIONS, NO 16MIL
                
                out = self.forward(x=signals, xref=xref, f=features)
                
                labels = labels.nonzero()[:,1] # to get the index of each class
                loss = self.criterion(out,labels)
                
                running_loss += loss.item()
                    
                #torch.autograd.set_detect_anomaly(True)
                
                loss.backward()
                
                self.optim.step()
                
                
            self.loss_during_training.append(running_loss/len(trainloader))
            
            # Validation Loss
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad(): 
                
                # set model to evaluation mode
                self.eval()
                
                running_loss = 0.
                
                for ((signals, features, labels), xref) in zip(validloader, xrefloader):
                    xref = xref[0]
                    labels = labels.type(torch.LongTensor)
                    signals, features, labels, xref = signals.to(self.device), features.to(self.device), labels.to(self.device) , xref.to(self.device) 

                    #signals = signals[:,:self.sequence_length]
                    
                    out = self.forward(x=signals, xref=xref, f=features)
                    
                    labels = labels.nonzero()[:,1] # to get the index of each class
                    loss = self.criterion(out,labels)

                    running_loss += loss.item()   
                    
                self.valid_loss_during_training.append(running_loss/len(validloader)) 
                if len(self.valid_loss_during_training)>1:
                    if self.valid_loss_during_training[-1]<np.min(self.valid_loss_during_training[:-1]):
                        torch.save(self.state_dict(), path + 'checkpoint_EarlyStopping_bestvalepoch_SERVER_cnn_lstm_3lay_xref_subsampled_lr'+str(args.lr)+'_dropprob'+str(args.drop_prob)+'_dropout'+str(args.dropout)+'_epochs'+str(args.epochs)+'_hidden'+str(args.hidden)+'.pth')
    
               # set model back to train mode
            self.train()
                    
            if(e % 1 == 0): # Every 10 epochs (now 1 bc we are using only 5)

                print("Epoch %d. Training loss: %f, Validation loss: %f, Time per epoch: %f seconds" 
                      %(e,self.loss_during_training[-1],self.valid_loss_during_training[-1],
                       (time.time() - start_time)))

    def eval_performance(self,dataloader,xrefloader):
        
        loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            
            # set model to evaluation mode
            self.eval()

            for ((signals, features, labels), xref) in zip(dataloader, xrefloader):
                xref = xref[0]
                
                #signals = signals[:,:self.sequence_length]
                #xref = xref[:,:self.sequence_length]
                signals, features, labels, xref = signals.to(self.device), features.to(self.device), labels.to(self.device) , xref.to(self.device) 

                probs = self.forward(x=signals, xref=xref, f=features)

                labels = labels.nonzero()[:,1]
                
                top_p, top_class = probs.topk(1, dim=1)
                equals = (top_class == labels.view(signals.shape[0],1))
                accuracy += torch.mean(equals.type(torch.FloatTensor))
    
            return accuracy/len(dataloader)


    def get_auc(self, dataloader, xrefloader, y, name=''):
        
        probs = np.zeros(y.shape)
        y_true = np.zeros(y.shape)
        
        aux = 0
        auc = []
        with torch.no_grad():
            self.eval()
            for ((signals, features, labels), xref) in zip(dataloader, xrefloader):
                xref = xref[0]
                signals, features, xref = signals.to(self.device), features.to(self.device), xref.to(self.device) 
                logits = self.forward(x=signals, xref=xref, f=features).detach().cpu().numpy()
                
                probs[aux:aux+len(labels)] = np.exp(logits)
                y_true[aux:aux+len(labels)] = labels
                aux+=len(labels)

            morethan1class = [len(np.unique(y_true[:,i])) for i in range(y_true.shape[1])]
            classes = np.where(np.array(morethan1class)>1)[0]
            auc = roc_auc_score(y_true[:, classes], probs[:, classes], multi_class='ovo', average='macro')
            save_roc_multiclass(y_test = y_true, y_score = probs, name=name)
        return auc

# LOAD DATA TO TRAIN NN 
print("Loading data and preprocessing...")
X_ws1, Y_ws1, feat_ws1, xref_ws1 = preprocess_pickle(path_ws1, features=True)
trainloader, validloader, testloader, x_tr, y_tr, feat_tr, x_val, y_val, feat_val, x_test, y_test, feat_test = dataloader(X_ws1, Y_ws1, feat_ws1, val=True)

xr = TensorDataset(torch.Tensor(xref_ws1))
xrefloader = DataLoader(xr, batch_size=4, shuffle=True)
my_NN = CNN_LSTM_MLP(in_channels=1, 
                        out_channels=10, 
                        out_channels_cnn2 = 40,
                        out_channels_cnn3 = 60,
                        out_channels_cnn4 = 80,
                        out_channels_cnn5 = 100,
                        out_channels_cnn6 = 120,
                        kernel_size=2,
                        kernel_maxpool = 2,
                        nlabels=4,
                        input_size=120, 
                        hidden_dim = args.hidden,
                        n_layers=3,
                        sigma = 1,
                        sequence_length = 2000,
                        use_batch_norm=True,
                        in_features = feat_tr.shape[1], 
                        out_features = 10,
                        batch_size=20,
                        batch_size_ref=4,
                        epochs=args.epochs, 
                        drop_prob=args.drop_prob,
                        dropout =args.dropout,
                        lr=args.lr)
if args.train:
    # TRAIN COMPLETE MODEL IN WS1 AND SAVE IT
    print("Training the model...")     
    my_NN.trainloop(trainloader, validloader,xrefloader)
    torch.save(my_NN.state_dict(), path + 'checkpoint_LASTEPOCH_SERVER_cnn_lstm_3lay_xref_subsampled_lr'+str(args.lr)+'_dropprob'+str(args.drop_prob)+'_dropout'+str(args.dropout)+'_epochs'+str(args.epochs)+'_hidden'+str(args.hidden)+'.pth')
    
    print("Calculating results...")
    # EVALUATION METRICS FOR WS1 COMPLETE DATASET 
    my_NN.load_state_dict(torch.load(path + 'checkpoint_EarlyStopping_bestvalepoch_SERVER_cnn_lstm_3lay_xref_subsampled_lr'+str(args.lr)+'_dropprob'+str(args.drop_prob)+'_dropout'+str(args.dropout)+'_epochs'+str(args.epochs)+'_hidden'+str(args.hidden)+'.pth'))
    # accuracy
    acc_train = my_NN.eval_performance(trainloader, xrefloader)
    acc_val = my_NN.eval_performance(validloader, xrefloader)
    acc_tst = my_NN.eval_performance(testloader, xrefloader)

    
    # loss
    loss_tr = my_NN.loss_during_training
    loss_valid = my_NN.valid_loss_during_training
    loss_curve(loss_tr, loss_valid, name= 'SERVER_loss_cnn_6lay_xref_subsam_MULTICLASS', path=path)

    # AUC 
    auc_train = my_NN.get_auc(trainloader, xrefloader, y_tr, name='ROC_CNN_LSTM_TRAIN')
    auc_val = my_NN.get_auc(validloader, xrefloader, y_val, name = 'ROC_CNN_LSTM_VALID')
    auc_test = my_NN.get_auc(testloader, xrefloader, y_test, name = 'ROC_CNN_LSTM_TEST')

    # save accuracy and auc of results in txt
    f = open(path + 'results_complete_multiclass_lstm_cnn_6lay_xref_subsam_ws1_lr'+str(args.lr)+'_dropprob'+str(args.drop_prob)+'_dropout'+str(args.dropout)+'_epochs'+str(args.epochs)+'_hidden'+str(args.hidden)+'.txt', 'w')
    f.write('ACCURACY FOR TRAINING: ' + str(round(acc_train.item(),3)))
    f.write('ACCURACY FOR VALIDATION: ' + str(round(acc_val.item(),3)))
    f.write('ACCURACY FOR TEST: ' + str(round(acc_tst.item(),3)))
    f.write('AUC FOR TRAINING: ' + str(round(auc_train,3)))
    f.write('AUC FOR VALIDATION: ' + str(round(auc_val,3)))
    f.write('AUC FOR TEST: ' + str(round(auc_test,3)))
    f.close()


    # DEFINE ALL POSSIBLE CONFIGURATIONS 
    configurations = {'Load': [0.0,1.0],'Velocidad': [0.0,1.0],'Lado' : [0,1],'Direction':[0,1],'Corte': [0,1]}

    grid = list(ParameterGrid(configurations))

    # TABLE TO SAVE RESULTS
    results = pd.DataFrame(columns=['Lado','Corte','Direction','Load','Velocidad',
                                    'ACC WS1', 'ACC WS2', 'ACC WS3', 'ACC WS4',
                                    'AUC WS1', 'AUC WS2', 'AUC WS3', 'AUC WS4'])

    model = my_NN
    #  ITERATE AND SAVE RESULTS
    for i,conf in enumerate(grid):
        print("Configuration: "+str(i)+"/"+str(len(grid)))
        # OBTAIN ALL DATA FOR EACH SUBSET
        X_ws1, Y_ws1, feat_ws1, xref_ws1 = preprocess_pickle_configuration(path_ws1, lado=conf['Lado'], direction = conf['Direction'], corte=conf['Corte'], carga=conf['Load'],velocidad=conf['Velocidad'])
        X_ws2, Y_ws2, feat_ws2, xref_ws2 = preprocess_pickle_configuration(path_ws2, lado=conf['Lado'], direction = conf['Direction'], corte=conf['Corte'], carga=conf['Load'],velocidad=conf['Velocidad'])
        X_ws3, Y_ws3, feat_ws3, xref_ws3 = preprocess_pickle_configuration(path_ws3, lado=conf['Lado'], direction = conf['Direction'], corte=conf['Corte'], carga=conf['Load'],velocidad=conf['Velocidad'])
        X_ws4, Y_ws4, feat_ws4, xref_ws4 = preprocess_pickle_configuration(path_ws4, lado=conf['Lado'], direction = conf['Direction'], corte=conf['Corte'], carga=conf['Load'],velocidad=conf['Velocidad'])
        
        # CREAMOS DATALOADER PARA CADA SUBSET, OBTENEMOS ACC Y AUC
        
        # WS1
        y_bin_ws1 = label_binarize(Y_ws1, classes=['eje sano', 'd1', 'd2','d3'])
        dataset_ws1 = TensorDataset(torch.Tensor(X_ws1), torch.Tensor(feat_ws1.values),torch.Tensor(y_bin_ws1)) 
        dataloader_ws1 = DataLoader(dataset_ws1, batch_size = 20, shuffle = True)  
        xr_ws1 = TensorDataset(torch.Tensor(xref_ws1))
        xrefloader_ws1 = DataLoader(xr_ws1, batch_size=4, shuffle=True)

        acc_ws1 = model.eval_performance(dataloader_ws1, xrefloader_ws1)
        auc_ws1 = model.get_auc(dataloader_ws1, xrefloader_ws1, y_bin_ws1, name= 'ROC_CNN_LSTM_WS1_'+str(list(conf.items())))


        # WS2
        y_bin_ws2 = label_binarize(Y_ws2, classes=['eje sano', 'd1', 'd2','d3'])
        dataset_ws2 = TensorDataset(torch.Tensor(X_ws2), torch.Tensor(feat_ws2.values),torch.Tensor(y_bin_ws2)) 
        dataloader_ws2 = DataLoader(dataset_ws2, batch_size = 20, shuffle = True)
        xr_ws2 = TensorDataset(torch.Tensor(xref_ws2))
        xrefloader_ws2 = DataLoader(xr_ws2, batch_size=4, shuffle=True)
        
        acc_ws2 = model.eval_performance(dataloader_ws2, xrefloader_ws2)
        auc_ws2 = model.get_auc(dataloader_ws2, xrefloader_ws2, y_bin_ws2, name= 'ROC_CNN_LSTM_WS2_'+str(list(conf.items())) )

    
        
        # WS3
        y_bin_ws3 = label_binarize(Y_ws3, classes=['eje sano', 'd1', 'd2','d3'])
        dataset_ws3 = TensorDataset(torch.Tensor(X_ws3), torch.Tensor(feat_ws3.values),torch.Tensor(y_bin_ws3)) 
        dataloader_ws3 = DataLoader(dataset_ws3, batch_size = 20, shuffle = True)
        xr_ws3 = TensorDataset(torch.Tensor(xref_ws3))
        xrefloader_ws3 = DataLoader(xr_ws3, batch_size=4, shuffle=True)
        
        acc_ws3 = model.eval_performance(dataloader_ws3, xrefloader_ws3)
        auc_ws3 = model.get_auc(dataloader_ws3, xrefloader_ws3, y_bin_ws3, name= 'ROC_CNN_LSTM_WS3_'+str(list(conf.items())))

        
        # WS4
        y_bin_ws4 = label_binarize(Y_ws4, classes=['eje sano', 'd1', 'd2','d3'])
        dataset_ws4 = TensorDataset(torch.Tensor(X_ws4), torch.Tensor(feat_ws4.values),torch.Tensor(y_bin_ws4)) 
        dataloader_ws4 = DataLoader(dataset_ws4, batch_size = 20, shuffle = True)
        xr_ws4 = TensorDataset(torch.Tensor(xref_ws4))
        xrefloader_ws4 = DataLoader(xr_ws4, batch_size=4, shuffle=True)
        
        acc_ws4 = model.eval_performance(dataloader_ws4, xrefloader_ws4)
        auc_ws4 = model.get_auc(dataloader_ws4, xrefloader_ws4, y_bin_ws4, name= 'ROC_CNN_LSTM_WS4_'+str(list(conf.items())))

        # UPDATE TABLE AFTER EACH ITERATION FOR A SPECIFIC CONFIGURATION
        
        results = results.append({'Lado': conf['Lado'],
                                'Corte': conf['Corte'],
                                'Direction': conf['Direction'],
                                'Load': conf['Load'],
                                'Velocidad': conf['Velocidad'],
                                'ACC WS1': round(acc_ws1.item(),3),
                                'ACC WS2': round(acc_ws2.item(),3),
                                'ACC WS3': round(acc_ws3.item(),3),
                                'ACC WS4': round(acc_ws4.item(),3),
                                'AUC WS1': round(auc_ws1,3),
                                'AUC WS2': round(auc_ws2,3),
                                'AUC WS3': round(auc_ws3,3),
                                'AUC WS4': round(auc_ws4,3)}, ignore_index=True)

    results.to_csv(path + 'results_SERVER_CNN_LSTM_COMPLETE_MODEL_MULTICLASS_xref_subsampled_lr'+str(args.lr)+'_dropprob'+str(args.drop_prob)+'_dropout'+str(args.dropout)+'_epochs'+str(args.epochs)+'_hidden'+str(args.hidden)+'.csv')



        
