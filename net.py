import torch
import torch.nn as nn
import torch.nn.init as init


class NN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU()                          
        self.fc2 = nn.Linear(hidden_size, output_size)
         
        self.init_weights()

    def init_weights(self):
        # Initialize weights of linear layers using He initialization
        init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

        
        
        # Initialize biases to zeros
        if self.fc1.bias is not None:
            init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            init.constant_(self.fc2.bias, 0)
        
    def forward(self, x):
       
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



