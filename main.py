import loader
import net
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

if __name__ == "__main__":

    X, Y = loader.loadData()

    print(X)



    '''input_size = 5
    hidden_size = 100
    output_size = 1
    model = net.NN_Model(input_size, hidden_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainLosses = []
    epochs = 5000

    for epoch in range(epochs):
        model.train()

        predictions = model(trainX)
        loss = criterion(predictions, trainY) 
        trainLosses.append(loss.item())
        
        loss.backward()
        optimizer.step()
    

    plt.plot(trainLosses)
    plt.show()'''
    
  



    

   