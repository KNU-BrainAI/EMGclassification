import torch
import torch.nn as nn

#%% functions
criterion = nn.CrossEntropyLoss()

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


#%% default
def train_step(batch_item, epoch, batch, training,model, optimizer, device):
    inputs = batch_item['input_data'].to(device)
    labels = batch_item['label'].to(device)
    criterion = nn.CrossEntropyLoss()
    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(): output= model(inputs)
        loss = criterion(output, labels)
        #print(labels)
        #print(torch.argmax(output,dim=1))
        accuracy = calculate_accuracy(output, labels)
        loss.backward()
        optimizer.step()
    else:
        model.eval()
        with torch.no_grad(): output= model(inputs)
        #print(labels)
        #print(torch.argmax(output,dim=1))
        loss = criterion(output, labels)
        accuracy = calculate_accuracy(output, labels)
        
    return loss, accuracy