import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training = True):
    
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set=datasets.FashionMNIST('./data',train=True,
            download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False,
            transform=custom_transform)
    

    if training == True:
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
        
    return loader


def build_model():
    
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1 * 28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    return model


def train_model(model, train_loader, criterion, T):
    
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(T):
        
        model.train()
        
        size = len(train_loader.dataset)
        
        running_loss = 0.0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            
            inputs, labels = data
            batch_size = inputs.size()[0]
            
            opt.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += (loss.item()*batch_size)
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = (correct / size * 100)
    
        print("Train Epoch: " + str(epoch) + "  Accuracy: " + str(correct) + "/"  + str(size) + "(" + str(f'{accuracy:.2f}') + "%)" + " " + "Loss: " + str(f'{(running_loss / size):.3f}'))


def evaluate_model(model, test_loader, criterion, show_loss = True):
    
    model.eval()
    
    size = len(test_loader.dataset)
    running_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            
            images, labels = data
            batch_size = images.size()[0]
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            running_loss += (loss.item() * batch_size)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = (correct / total) * 100
    
    if (show_loss == False):
        print("Accuracy: " + str(f'{accuracy:.2f}') + "%")
    else:
        print("Average loss: " + str(f'{(running_loss / size):.4f}'))
        print("Accuracy: " + str(f'{accuracy:.2f}') + "%")


def predict_label(model, test_images, index):
    
    test_image = test_images[index]
    
    logits = model(test_image)
    
    prob = prob = F.softmax(logits, dim=1)
    
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt'
                ,'Sneaker','Bag','Ankle Boot']
    probs = []
    for p in prob[0]:
        probs.append(p.item() * 100)
    sorted_pairings = sorted(zip(probs, class_names), reverse=True)
    
    print(str(sorted_pairings[0][1]) + ": " + str(f'{sorted_pairings[0][0]:.2f}') + "%")
    print(str(sorted_pairings[1][1]) + ": " + str(f'{sorted_pairings[1][0]:.2f}') + "%")
    print(str(sorted_pairings[2][1]) + ": " + str(f'{sorted_pairings[2][0]:.2f}') + "%")
    

if __name__ == '__main__':

    criterion = nn.CrossEntropyLoss()
    
    train_loader = get_data_loader()
    
    test_loader = get_data_loader(False)
    
    model = build_model()
                         
    train_model(model, train_loader, criterion, 5)
    
    evaluate_model(model, test_loader, criterion, show_loss = False)
    
    evaluate_model(model, test_loader, criterion, show_loss = True)
    
    pred_set, _ = next(iter(test_loader))
    predict_label(model, pred_set, 1)
    
