import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
from sklearn.preprocessing import LabelBinarizer, normalize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from model import getSimpleNet, getSimpleNetRegression
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CCLE_MAIN = "input/CCLE_RPKM/CCLE_DepMap_18q3_RNAseq_RPKM_20180719.txt"
CCLE_LABELS = "input/CCLE_NP24.2009_Drug_data_2015.02.24.csv"

main_gene = pd.read_csv(CCLE_MAIN, sep="\t", low_memory=False)
long_names = [x.split(" ")[0] for x in list(main_gene.columns.values)]
main_gene.columns = long_names


main_label = pd.read_csv(CCLE_LABELS, low_memory=False)

main_label = main_label[["CCLE Cell Line Name", "Primary Cell Line Name", "Compound", "IC50 (uM)"]]
#print(main_gene.head(19))


names = list()
drugs = list(set(list(main_label["Compound"].values)))
#print(long_names)

for drug in tqdm(drugs):
    p = []
    for index, row in main_label.iterrows():
        if row[2] == drug:
            if row[0] in long_names:
                p.append(row[0])
    p = np.array(p)
    names.append(p)

#print(names)
dict_of_ic50 = dict() # [Drug, Cell Line Name] -> IC50
for index, row in tqdm(main_label.iterrows()):
    dict_of_ic50[repr([row[2], row[0]])] = row[3]

def get_data(j, drug):
    x = list(main_gene[j][1:])
    y = dict_of_ic50[repr([drug, j])]

    return x, y


names = np.array(names)


class GeneExpressionDataset(Dataset):
    def __init__(self, x, drug):
        self.names = x
        self.drug = drug

    def __len__(self):
        return len(self.names)
    
    def transform(self, x): 
        #print(np.count_nonzero(np.isnan(x))/len(x)) 
        x = np.array(x).astype(dtype=np.float32)
        x = np.nan_to_num(x) #set Nan to zero
        x = x.reshape(1, -1) #One Sample
        x = normalize(x) #L2 Normalization from sklearn
        x = torch.from_numpy(x).float() #convert ndarray to torch Tensor 
        return x

    def __getitem__(self, index):
        name = self.names[index]
        x, y = get_data(name, self.drug)
        return self.transform(x), torch.tensor(y)



lol = 0
for drug in drugs:
    print("Start Training on: {}".format(drug))
    inputs = names[lol]

    trainx, testx = train_test_split(inputs, test_size=0.05, random_state=42)
    trainx, validx = train_test_split(trainx, test_size=0.3, random_state=42)

    trainset = GeneExpressionDataset(trainx, drug)
    validset = GeneExpressionDataset(validx, drug)
    testset = GeneExpressionDataset(testx, drug)

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

    print("Finish Creating Datasets and DataLoaders")

    xpll, _ = get_data("22RV1_PROSTATE", "Nilotinib")
    net, criterion, optimizer = getSimpleNetRegression(len(xpll), 30, lr=0.01)
    net.train()

    print("Starting Training!")

    EPOCHS = 10

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
            test_loss += criterion(output, classes).item()
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
            loss = criterion(outputs, labels)
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
    PATH = 'saves/Simplenet_{}.pth'.format(drug)
    if os.path.exists(PATH):
        yesorno = input("Do you want to replace the current model {}? Y[es] or N[o] or R[emove]: ".format(drug)).lower()
        if yesorno == 'y' or yesorno == 'yes':
            os.remove(PATH) 
            torch.save(net, PATH)
        elif yesorno == 'r' or yesorno == 'remove':
            print("You are not saving this model {}!".format(drug))
        else:
            j = 0
            while os.path.exists(PATH):
                j += 1
                PATH = '/'.join('/'.split(PATH)[:-1]) + os.path.splitext(os.path.basename(PATH))[0] + str(j) + ".pth"
            torch.save(net, PATH)

    test_loss = evaluation(net, testloader, criterion)
    print("Test Loss {}: {}".format(drug, test_loss))
    lol += 1
