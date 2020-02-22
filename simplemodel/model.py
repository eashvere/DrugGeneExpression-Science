import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#TODO Hyperparamter tuning?

class SimpleNet(nn.Module):
    def __init__(self, classes, inputs_size):
        super(SimpleNet, self).__init__()
        self.inputs_size = inputs_size
        self.fc1 = nn.Linear(self.inputs_size, 98)
        self.drop1 = nn.Dropout(p=0.9)
        self.fc2 = nn.Linear(98, 40)
        self.drop2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(98, classes)
    
    def forward(self, x):
        x = x.view(-1, self.inputs_size)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        #x = F.relu(self.fc2(x))
        #x = self.drop2(x)
        x = F.softmax(self.fc3(x), dim=1)
        return x

class SimpleNetRegression(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(SimpleNetRegression, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

def getSimpleNet(classes, input_size):
    net = SimpleNet(classes, input_size)
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-5)

    return net, criterion, optimizer

def getSimpleNetRegression(input_size, hidden, lr=0.2):
    net = SimpleNetRegression(input_size, hidden, 1)
    net.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True)

    return net, criterion, optimizer