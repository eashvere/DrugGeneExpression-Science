import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
from sklearn.preprocessing import LabelBinarizer, normalize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from model import getSimpleNet
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


#TODO Eliminate low variation genes across all patients, PCI or ICA (Dimensional Reduction). Feature Selection maybe?

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_txt_pandas(path, sep='\t'):
    """
    Inputs: Path: Path to txt File, sep: the separation of txt file defaults to tab-delimit
    Output: dataf Pandas Dataframe
    """
    dataf = pd.read_csv(path, sep=sep, low_memory=False) #This outputs a pandas dataframe
    #dataf.reset_index(drop=True, inplace=True)
    return dataf


DATA_FILE_CLEAN = '/home/eashver/Desktop/project/geneexpres/input/BRCA.clin.merged.picked.txt'
DATA_FILE_HUGE = '/home/eashver/Desktop/project/geneexpres/input/RSEM_isoforms_normal_BRCA/BRCA.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_isoforms_normalized__data.data.txt'
DATA_FILE_ISOFORM_GENE = '/home/eashver/Desktop/project/geneexpres/input/isofrm_gene.txt'
DATA_FILE_MRNA = 'input/BRCA_mRNA/BRCA.transcriptome__agilentg4502a_07_3__unc_edu__Level_3__unc_lowess_normalization_gene_level__data.data.txt'

dataf = get_txt_pandas(DATA_FILE_MRNA)
#dataf.div(dataf.sum(axis=1), axis=0) #Normalize the DATATATATATAT
dataf_out = get_txt_pandas(DATA_FILE_CLEAN)
isoform_gene = get_txt_pandas(DATA_FILE_ISOFORM_GENE)

print(dataf_out.head(19))

print("Finished inputing the datafiles")


def get_info_hybrid_REF(hybrid_ref):
    """
    Inputs: hybrid_ref: the big version of the hybrid ref name
    Output: the isoform values and the stages! as an array. Or it is zero if there is no match!
    """
    isoform_values = list(dataf[hybrid_ref][1:])
    s = ("-".join(hybrid_ref.split('-')[:3])).lower()
    if s not in dataf_out.columns:
        return 0
    stage = dataf_out.iloc[6][s] #the cancer stage
    t_stage = dataf_out.iloc[7][s] #the size of the cancer
    n_stage = dataf_out.iloc[8][s] #whether the nearby lymph nodes have cancer
    m_stage = dataf_out.iloc[9][s] #whether the cancer has spread
    return [isoform_values, [stage, t_stage, n_stage, m_stage]]

def binarize(inputs):
    lb = LabelBinarizer()
    f = lb.fit_transform(inputs)
    #print(lb.classes_) #TODO Nan as a class??!??!?!?!?!?!?
    return f

#print(get_info_hybrid_REF('TCGA-3C-AAAU-01A-11R-A41B-07')[0])

inputs = []

stage = []
stagen = []
staget = []
stagem = []

for a in tqdm(dataf.columns[1:]):
    data = get_info_hybrid_REF(a)
    if data == 0:
        continue
    inputs.append(data[0])
    stage.append(data[1][0])
    staget.append(data[1][1])         
    stagen.append(data[1][2])
    stagem.append(data[1][3])
    

inputs = np.array(inputs).astype(np.float32)

#print(np.isnan(inputs).any())

stage = np.array(stage)
stagen = np.array(stagen)
staget = np.array(staget)
stagem = np.array(stagem)

stage = binarize(stage)
stagen = binarize(stagen)
staget = binarize(staget)
stagem = binarize(stagem)

inputs = np.nan_to_num(inputs) #set Nan to zero

pca_train = PCA(n_components=min(len(inputs), len(inputs[0])), svd_solver='auto')
inputs = pca_train.fit_transform(inputs)
#pca_test = PCA(n_components=min(len(stage), len(stage[0])), svd_solver='auto')
#stage = pca_test.fit_transform(stage)

trainx, testx, trainy, testy = train_test_split(inputs, stage, test_size=0.05, random_state=42)
trainx, validx, trainy, validy = train_test_split(trainx, trainy, test_size=0.3, random_state=42)

print(trainx[0])
print(trainy[0])

print("Finished creating the seperate lists")

class GeneExpressionDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def transform(self, x): 
        #print(np.count_nonzero(np.isnan(x))/len(x)) 
        #x = np.nan_to_num(x) #set Nan to zero
        x = x.reshape(1, -1) #One Sample
        x = normalize(x) #L2 Normalization from sklearn
        x = torch.from_numpy(x) #convert ndarray to torch Tensor 
        return x

    def __getitem__(self, index):
        return self.transform(self.x[index]), self.y[index]


trainset = GeneExpressionDataset(trainx, trainy)
validset = GeneExpressionDataset(validx, validy)
testset = GeneExpressionDataset(testx, testy)

trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
validloader = DataLoader(validset, batch_size=4, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

print("Finish Creating Datasets and DataLoaders")

net, criterion, optimizer = getSimpleNet(len(trainy[0]), len(trainx[0]))
net.train()

print("Starting Training!")

EPOCHS = 50

def evaluation(model, testloader, criterion):
    model.eval()
    test_loss = 0
    accuracy = 0
    i = 0

    for inputs, classes in testloader:
        inputs = inputs.to(device)
        classes = classes.to(device)
        output = model.forward(inputs)
        #print(output.shape)
        test_loss += criterion(output, torch.max(classes, 1)[1]).item()
        i+=1

        #ps = torch.exp(output)
        #equality = (labels.data == ps.max(dim=1)[1])
        #accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss/i#, accuracy

for epoch in range(EPOCHS):
    running_loss = 0.0
    valid_loss = 0.0
    j = 1
    accuracy = 0.0
    for i, data in enumerate(trainloader, 0):
        net.train()
        inputs, labels = data
        #print(inputs)
        #print(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        #print(outputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        valid_loss = evaluation(net, validloader, criterion)


        print("Epoch: {}, Processing mini-batch {}, Loss: {}, Valid Loss {}, Accuracy: {}".format(epoch+1, i, running_loss/j, valid_loss, accuracy), end="\r")
        j += 1
        if i % 200 == 199:
            j = 1
            running_loss = 0.0
    final_epoch_loss = evaluation(net, validloader, criterion)
    final_epoch_accuracy = 0.0
    print("\n")
    print("Epoch: {}, Valider Loss {}, Accuracyness {}".format(epoch+1, final_epoch_loss, final_epoch_accuracy), end="\r")
    print("\n")

print("Finished Training")
PATH = 'saves/Simplenet.pth'
if os.path.exists(PATH):
    yesorno = input("Do you want to replace the current model? Y[es] or N[o] or R[emove]: ").lower()
    if yesorno == 'y' or yesorno == 'yes':
        os.remove(PATH) 
        torch.save(net, PATH)
    elif yesorno == 'r' or yesorno == 'remove':
        print("You are not saving this model!")
    else:
        j = 0
        while os.path.exists(PATH):
            j += 1
            PATH = '/'.join('/'.split(PATH)[:-1]) + os.path.splitext(os.path.basename(PATH))[0] + str(j) + ".pth"
        torch.save(net, PATH)

test_loss = evaluation(net, testloader, criterion)
print("Test Loss: {}".format(test_loss))
