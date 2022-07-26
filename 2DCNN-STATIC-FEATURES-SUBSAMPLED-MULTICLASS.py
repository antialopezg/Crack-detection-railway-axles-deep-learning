# -*- coding: utf-8 -*-
"""
Created on Mon May  9 17:06:18 2022

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
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, auc,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,  OneHotEncoder, label_binarize
from sklearn.model_selection import ParameterGrid
from scipy import signal
from skimage.transform import resize
import argparse
from itertools import cycle

# DEFINE FUNCTIONS USED
def subsam(sig, length):
    subsampled = signal.resample(x=sig[:length], num=2000)
    return subsampled

def normalize(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    norm_data = scaler.transform(X)
    return norm_data

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
def compute_spec(x):
    #f, t, Sxx = signal.spectrogram(x, fs=12800)
    f, t, Sxx_im = signal.spectrogram(x, fs=12800, mode='complex')
    f, t, Sxx_mag = signal.spectrogram(x, fs=12800, mode='magnitude')
    #S = np.stack((Sxx,Sxx_im, Sxx_mag), axis=-1)
    S = np.stack((np.real(Sxx_im),np.imag(Sxx_im), Sxx_mag), axis=-1)
    return S

def spectrogram(xsig):
    d1 = pd.DataFrame()
    d1['x_ws1'] = list(xsig)
    X_spec_WS1 = pd.DataFrame()
    X_spec_WS1['Sx'] = d1['x_ws1'].map(lambda x: compute_spec(x))
    X_spec_WS1['Sx'] = X_spec_WS1['Sx'].map(lambda x: resize(x, (64, 64), mode = 'constant'))
    x = np.stack(X_spec_WS1['Sx'])#.astype(None)

    return x
def plot_roc_curve(fper, tper, auc, title=''):
    plt.figure()
    plt.plot(fper, tper, color='red', label='ROC curve and AUC = %0.3f' %auc)
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve'+ title)
    plt.legend()
    plt.show()
    
def plot_spec(x):
    f, t, Sxx = signal.spectrogram(x, fs=12800)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram representation')
    plt.show()
    
def preprocess_pickle(path, features = False):
    data = pd.read_pickle(path)
    
    
    X = np.stack(data['Subsampled'].values).astype(None)
    feat = data[['Lado','Direction','Corte','Load','Velocidad']]
    Y = data['Label']
    
    
    if features:
        return X, Y, feat
    else:
        return X, Y
    
def preprocess_pickle_configuration(path, lado, direction, corte, carga, velocidad):
    data = pd.read_pickle(path)
    data['Load'] = data['Load'].round()


    # NOW FILTER FOR SPECIFIC CONFIGURATION
    data = data[(data['Lado']==lado) & (data['Direction']==direction) & (data['Load']==carga) 
                & (data['Corte']==corte) & (data['Velocidad']==velocidad)]
    
    
    
    X = np.stack(data['Subsampled'].values).astype(None)
    feat = data[['Lado','Direction','Corte','Load','Velocidad']]
    Y = data['Label']
    
    return X, Y, feat
    
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
parser.add_argument('--epochs', type=float, default=100.0,
                    help='easdkf')
parser.add_argument('--drop_prob', type=float, default=0.3,
                    help='sdfasdfas')
parser.add_argument('--train', type=bool, default=False,
                    help='sdfasdfas') #esto es para saber si hay que entrenar o testear
args = parser.parse_args()
print(args)
#print(args.lr)    

# DESIGN 2DCNN: 6 LAYERS, BINARY CLASSIFICATION, XREF, MLP STATIC FEATURES 
class DCNN_MLP(nn.Module):
    
    
    def __init__(self, in_channels, out_channels, out_channels_cnn2, out_channels_cnn3,
                 out_channels_cnn4, out_channels_cnn5, out_channels_cnn6,
                 kernel_size, kernel_maxpool, nlabels, 
                 sequence_length, in_features, out_features, 
                 batch_size, batch_size_ref, epochs, drop_prob, lr):
        
        
        super().__init__() 
    
        
        # First: 1-dimension CNN with relu activation 
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding ='same')
        
        self.conv2 = nn.Conv2d(out_channels, out_channels_cnn2, kernel_size, padding ='same')
        
        self.conv3 = nn.Conv2d(out_channels_cnn2, out_channels_cnn3, kernel_size, padding ='same')

        self.conv4 = nn.Conv2d(out_channels_cnn3, out_channels_cnn4, kernel_size, padding ='same')
        
        self.conv5 = nn.Conv2d(out_channels_cnn4, out_channels_cnn5, kernel_size, padding ='same')
        
        self.conv6 = nn.Conv2d(out_channels_cnn5, out_channels_cnn6, kernel_size, padding ='same')

        self.relu = nn.ReLU()
        
        self.kernel = kernel_size
        
        self.kernel_maxpool = kernel_maxpool
        
        self.out_channels = out_channels
        
        self.out_channels_cnn2 = out_channels_cnn2

             # Third: MLP binary classifier with sigmoid activation 
             
        # son dez mil porque é 20(out_channels*2)*500(seqlength reduced in pooling)
        # sumolle out_channels*2 xq é a dimension do q sale do fc das features
        
        
        self.final_dim_cnn = out_channels_cnn6*kernel_size + out_features
        
        # é out_channels_cnn2 * seq length after max pooling layers 
        
        
        #self.fc1 = nn.Linear(int(out_channels_cnn6 * sequence_length/(kernel_maxpool*6)) + out_features, nlabels)
        self.fc1 = nn.Linear(out_channels_cnn6 + out_features, nlabels)
        # aqui o de multiplicar por 2 é xq paso ambos por cnns, entonces concateno 3 inputs, 2 de cnn pa os que teño params cnn por 2 mas out feat
        #print('esto e o parametro q lle meto a fc1', self.final_dim_cnn)
        
        
        # FOURTH: MLP FOR THE FEATURES
        self.in_features = in_features
        self.out_features = out_features
        
        self.fc2 = nn.Linear(in_features, out_features)
        
        
        # ANOTHER PARAMETERS AND VARIABLES IMPORTANT FOR NN
        self.sigmoid = nn.Sigmoid() # BINARY OUTPUT
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.dropout = nn.Dropout(p=drop_prob)
        
        self.pool = nn.MaxPool2d(2,2)
        
        
        self.lr = lr #Learning Rate
        
        self.epochs = epochs
        
        self.sequence_length = sequence_length
        
        self.batch_size_ref = batch_size_ref
        
        self.optim = optim.Adam(self.parameters(), self.lr)
        
        #self.criterion = nn.BCEWithLogitsLoss() #   BINARY OUTPUT
        
        self.criterion = nn.NLLLoss() 
        
        self.batch_size = batch_size
        
        # A list to store the loss evolution along training
        
        self.loss_during_training = [] 
        
        self.valid_loss_during_training = []

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        #print(self.device) 
        self.to(self.device)
        
    def forward(self, x, f):
        
        
        batch_size = x.size(0) # Number of signals N
        seq_length = x.size(1) # T
        
        # FIRST PASS THE INPUT SPECTROGRAM THROUGH 6 2DCNN LAYERS
        #print('input spectrogram shape', x.shape)
        x = x.reshape([x.shape[0], 3, x.shape[1], x.shape[2]])
        #print('INPUT SPECTROGRAM SHAPE ', x.shape)
        #print(type(x))
        

        x = self.conv1(x)
        #print('OUTPUT OF CONV1 LAYER SPECTROGRAM SHAPE ', x.shape)
        x = self.relu(x)
        #print('OUTPUT OF CONV LAYER AND RELU SHAPE ', x.shape)
        
        x = self.pool(x)
        
        #print('OUTPUT OF CONV1 + pool LAYER SHAPE ', x.shape)
        x = self.conv2(x)
        #print('OUTPUT OF CONV2 LAYER SPECTROGRAM SHAPE ', x.shape)
        x = self.relu(x)
        
        x = self.pool(x)
        #print('OUTPUT OF CONV2 + pool LAYER SHAPE ', x.shape)
        
        # 3rd convolution
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        #print('OUTPUT OF CONV3 LAYER SPECTROGRAM SHAPE ', x.shape)
        
        # 4th convolution
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        #print('OUTPUT OF CONV4 LAYER SPECTROGRAM SHAPE ', x.shape)
        
        # 5th convolution
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)
        #print('OUTPUT OF CONV5 LAYER SPECTROGRAM SHAPE ', x.shape)
        
        # 6th convolution
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)
        #print('OUTPUT OF CONV6 LAYER SPECTROGRAM SHAPE ', x.shape)
        
        # ADD REGULARIZATION 
        x = self.dropout(x)
        
        x = x.reshape(batch_size,-1)
        #print('input of fc SPECTROGRAM layer shape',x.shape)
       
        # SECOND PASS THE FEATURES THROUGH A 1 LAYER MLP
        #print('INPUT FEATURES SHAPE', f.shape)
        f = self.fc2(f)
        #print('OUTPUT FC2 FEATURES SHAPE', f.shape)
        f = self.relu(f)
        #print('OUTPUT OF FC2 (FEATURES) AND RELU SHAPE', f.shape)
        f = self.dropout(f)
        
        
        
        # FOURTH CONCATENATE SIGNALS, REFERENCE AND FEATURES AND PASS THROUGH FINAL MLP
        # DUPLICATE XREF SIGNALS TO CONCATENATE WITH EACH SIGNAL OF THE BATCH THE SAME VALUES
        
    
        
        #print('input fc xref', xref.shape)
        #print('x', x.shape)
        #print('f',f.shape)
        z = torch.cat((x,f), 1)
        #print('CONCATENATED SIGNAL + XREF + FEATURES SHAPE',z.shape)
        output = self.fc1(z) 
        #print('OUTPUT OF lAST FULLY CONV LAYER SHAPE ', output.shape)
        
        output = self.logsoftmax(output) # BINARY OUTPUT
        #print('OUTPUT OF LOGSOFTMAX SHAPE ', output.shape)
        
        return output
           
    def trainloop(self,trainloader, validloader):
        
        # set model back to train model
        self.train()
        
        # Optimization Loop
        
        for e in range(int(self.epochs)):

            start_time = time.time()
            
            # Random data permutation at each epoch
            
            running_loss = 0.
            
            for (signals, features, labels) in trainloader:
        
                self.optim.zero_grad()  #TO RESET GRADIENTS!
                
                labels = labels.type(torch.LongTensor)
                signals, features, labels = signals.to(self.device), features.to(self.device), labels.to(self.device) 

                #signals = signals[:,:self.sequence_length] # COGER SOLO x OBSERVATIONS, NO 16MIL
                #xref = xref[:,:self.sequence_length]
                
                out = self.forward(x=signals, f=features)
                
                #daba un error de que se esperaba type long y found float

                #print('dim of output',out.shape)
                labels = labels.nonzero()[:,1] # to get the index of each class

               # print('dim of labels', labels.shape)
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
                
                for signals, features, labels in validloader:
                   
                    labels = labels.type(torch.LongTensor)
                    signals, features, labels = signals.to(self.device), features.to(self.device), labels.to(self.device) 

                    #signals = signals[:,:self.sequence_length]
                    #xref = xref[:,:self.sequence_length]
                    
                    out = self.forward(x=signals, f=features)
                    
                    labels = labels.nonzero()[:,1] # to get the index of each class

                    loss = self.criterion(out,labels)

                    running_loss += loss.item()   
                    
                self.valid_loss_during_training.append(running_loss/len(validloader))    
                if len(self.valid_loss_during_training)>1:
                    if self.valid_loss_during_training[-1]<np.min(self.valid_loss_during_training[:-1]):
                        torch.save(self.state_dict(), path + 'checkpoint_EarlyStopping_bestvalepoch_SERVER_static_2dcnn_potencia2_6_layers_xreference_subsampled_lr'+str(args.lr)+'_dropprob'+str(args.drop_prob)+'_epochs'+str(args.epochs)+'.pth')
    
            # set model back to train mode
            self.train()
            
                    
            if(e % 2 == 0): # Every 10 epochs (now 1 bc we are using only 5)

                print("Epoch %d. Training loss: %f, Validation loss: %f, Time per epoch: %f seconds" 
                      %(e,self.loss_during_training[-1],self.valid_loss_during_training[-1],
                       (time.time() - start_time)))

    def eval_performance(self,dataloader):
        
        loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            
            # set model to evaluation mode
            self.eval()

            for signals, features, labels in dataloader:
                
                signals, features, labels = signals.to(self.device), features.to(self.device), labels.to(self.device) 

                probs = self.forward(x=signals, f=features)
          
                labels = labels.nonzero()[:,1] # to get the index of each class

                top_p, top_class = probs.topk(1, dim=1)
                equals = (top_class == labels.view(signals.shape[0],1))
                accuracy += torch.mean(equals.type(torch.FloatTensor))
    
            return accuracy/len(dataloader)
    
    def get_auc(self, dataloader,  y, name=''):
        
        probs = np.zeros(y.shape)
        y_true = np.zeros(y.shape)
        
        aux = 0
        auc = []
        with torch.no_grad():
            self.eval()
            for (signals, features,labels) in dataloader:
                
                signals, features = signals.to(self.device), features.to(self.device) 
                logits = self.forward(x=signals, f=features).detach().cpu().numpy()
                
                probs[aux:aux+len(labels)] = np.exp(logits)
                y_true[aux:aux+len(labels)] = labels
                aux+=len(labels)

            morethan1class = [len(np.unique(y_true[:,i])) for i in range(y_true.shape[1])]
            classes = np.where(np.array(morethan1class)>1)[0]
            auc = roc_auc_score(y_true[:, classes], probs[:, classes], multi_class='ovo', average='macro')
            save_roc_multiclass(y_test = y_true, y_score = probs, name=name)
        return auc
        
print("Loading data...")     
# LOAD DATA TO TRAIN NN. TRANSFORM TO SPECTROGRAM
X_ws1, Y_ws1, feat_ws1 = preprocess_pickle(path_ws1, features=True)
x_ws1 = spectrogram(X_ws1)
trainloader, validloader, testloader,x_tr, y_tr, feat_tr, x_val, y_val, feat_val, x_test, y_test, feat_test = dataloader(x_ws1, Y_ws1, feat_ws1, val=True)


# TRAIN COMPLETE MODEL IN WS1 AND SAVE IT
my_DCNN = DCNN_MLP(in_channels=3,
                    out_channels=6, 
                    out_channels_cnn2 = 16,
                    out_channels_cnn3 = 26,
                    out_channels_cnn4 = 36,
                    out_channels_cnn5 = 46,
                    out_channels_cnn6 = 56,
                    kernel_size=2, 
                    kernel_maxpool = 2, 
                    nlabels=4, 
                    sequence_length = 2000,
                    in_features= feat_tr.shape[1], 
                    out_features= 10, 
                    batch_size=20, 
                    batch_size_ref=4,
                    epochs = args.epochs, 
                    drop_prob = args.drop_prob,
                    lr = args.lr)

if args.train:
    print("Training...")
    my_DCNN.trainloop(trainloader, validloader)
    torch.save(my_DCNN.state_dict(), path + 'checkpoint_SERVER_static_2dcnn_potencia2_6_layers_subsampled_lr'+str(args.lr)+'_dropprob'+str(args.drop_prob)+'_epochs'+str(args.epochs)+'.pth')


# EVALUATION METRICS FOR WS1 COMPLETE DATASET
else:

    # accuracy
    my_DCNN.load_state_dict(torch.load(path + 'checkpoint_EarlyStopping_bestvalepoch_SERVER_static_2dcnn_potencia2_6_layers_subsampled_lr'+str(args.lr)+'_dropprob'+str(args.drop_prob)+'_epochs'+str(args.epochs)+'.pth'))

    acc_train = my_DCNN.eval_performance(trainloader)
    acc_val = my_DCNN.eval_performance(validloader)
    acc_tst = my_DCNN.eval_performance(testloader)
    
    # loss
    loss_tr = my_DCNN.loss_during_training
    loss_valid = my_DCNN.valid_loss_during_training
    loss_curve(loss_tr, loss_valid, name= 'SERVER_loss_2dcnn_potencia2_6lay_xref_subsam_MULTICLASS_lr'+str(args.lr)+'_dropprob'+str(args.drop_prob)+'_epochs'+str(args.epochs), path=path)
    
    # AUC 
    auc_train = my_DCNN.get_auc(trainloader, y_tr, name='ROC_2DCNN_TRAIN')
    auc_val = my_DCNN.get_auc(validloader, y_val, name='ROC_2DCNN_VALID')
    auc_test = my_DCNN.get_auc(testloader, y_test, name='ROC_2DCNN_LSTM_TRAIN')
    
    # save accuracy and auc of results in txt
    f = open(path + 'results_complete_multiclass_2dcnn_potencia2_6lay_xref_subsam_ws1_lr'+str(args.lr)+'_dropprob'+str(args.drop_prob)+'_epochs'+str(args.epochs)+'.txt', 'w')
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
                                'AUC WS1', 'AUC WS2', 'AUC WS3', 'AUC WS4',
                                ])
    model = my_DCNN
    
    #  ITERATE AND SAVE RESULTS
    for conf in grid:
    
        # OBTAIN ALL DATA FOR EACH SUBSET
        X_ws1, Y_ws1, feat_ws1 = preprocess_pickle_configuration(path_ws1, lado=conf['Lado'], direction = conf['Direction'], corte=conf['Corte'], carga=conf['Load'],velocidad=conf['Velocidad'])
        X_ws2, Y_ws2, feat_ws2 = preprocess_pickle_configuration(path_ws2, lado=conf['Lado'], direction = conf['Direction'], corte=conf['Corte'], carga=conf['Load'],velocidad=conf['Velocidad'])
        X_ws3, Y_ws3, feat_ws3 = preprocess_pickle_configuration(path_ws3, lado=conf['Lado'], direction = conf['Direction'], corte=conf['Corte'], carga=conf['Load'],velocidad=conf['Velocidad'])
        X_ws4, Y_ws4, feat_ws4 = preprocess_pickle_configuration(path_ws4, lado=conf['Lado'], direction = conf['Direction'], corte=conf['Corte'], carga=conf['Load'],velocidad=conf['Velocidad'])
        
        #TRANSFORM DATA INTO SPECTROGRAMS
        x_ws1 = spectrogram(X_ws1)
        x_ws2 = spectrogram(X_ws2)
        x_ws3 = spectrogram(X_ws3)
        x_ws4 = spectrogram(X_ws4)
        
        # CREAMOS DATALOADER PARA CADA SUBSET, OBTENEMOS ACC Y AUC
        
        # WS1
        y_bin_ws1 = label_binarize(Y_ws1, classes=['eje sano', 'd1', 'd2','d3'])
        dataset_ws1 = TensorDataset(torch.Tensor(x_ws1), torch.Tensor(feat_ws1.values),torch.Tensor(y_bin_ws1)) 
        dataloader_ws1 = DataLoader(dataset_ws1, batch_size = 20, shuffle = True)  
           
        acc_ws1 = model.eval_performance(dataloader_ws1)
        auc_ws1 = model.get_auc(dataloader_ws1, y_bin_ws1, name= 'ROC_2DCNN_WS1_'+str(list(conf.items())))
        
        
        # WS2
        y_bin_ws2 = label_binarize(Y_ws2, classes=['eje sano', 'd1', 'd2','d3'])
        dataset_ws2 = TensorDataset(torch.Tensor(x_ws2), torch.Tensor(feat_ws2.values),torch.Tensor(y_bin_ws2)) 
        dataloader_ws2 = DataLoader(dataset_ws2, batch_size = 20, shuffle = True)
          
        acc_ws2 = model.eval_performance(dataloader_ws2)
        auc_ws2 = model.get_auc(dataloader_ws2, y_bin_ws2, name= 'ROC_2DCNN_WS2_'+str(list(conf.items())) )
        
        
        
        # WS3
        y_bin_ws3 = label_binarize(Y_ws3, classes=['eje sano', 'd1', 'd2','d3'])
        dataset_ws3 = TensorDataset(torch.Tensor(x_ws3), torch.Tensor(feat_ws3.values),torch.Tensor(y_bin_ws3)) 
        dataloader_ws3 = DataLoader(dataset_ws3, batch_size = 20, shuffle = True)
        
        acc_ws3 = model.eval_performance(dataloader_ws3)
        auc_ws3 = model.get_auc(dataloader_ws3, y_bin_ws3, name= 'ROC_2DCNN_WS3_'+str(list(conf.items())))
        
        
        # WS4
        y_bin_ws4 = label_binarize(Y_ws4, classes=['eje sano', 'd1', 'd2','d3'])
        dataset_ws4 = TensorDataset(torch.Tensor(x_ws4), torch.Tensor(feat_ws4.values),torch.Tensor(y_bin_ws4)) 
        dataloader_ws4 = DataLoader(dataset_ws4, batch_size = 20, shuffle = True)
        
        acc_ws4 = model.eval_performance(dataloader_ws4)
        auc_ws4 = model.get_auc(dataloader_ws4, y_bin_ws4, name= 'ROC_2DCNN_WS4_'+str(list(conf.items())))
        
        
        
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

results.to_csv(path + 'results_SERVER_2DCNN_COMPLETE_MODEL_MULTICLASS_subsampled_lr'+str(args.lr)+'_dropprob'+str(args.drop_prob)+'_epochs'+str(args.epochs)+'.csv')
