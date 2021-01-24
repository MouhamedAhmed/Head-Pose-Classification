import numpy as np
from datetime import datetime 
import torch
from tqdm import tqdm
from statistics import mean 


###############
# batch accuracy
def get_accuracy(y_hat, y_true):
    _, predicted_labels = torch.max(y_hat, 1)
    accuracy = (predicted_labels == y_true).sum()/y_true.size(0)
    return accuracy.item()


###############
# train iteration
def train(loader, batch_size, model, cross_entropy_loss_criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    epoch_loss = 0

    print("start training epoch...")
    pbar = tqdm(loader)
    for x,y in pbar:
        
        x = x.to(device).transpose(1,3).transpose(2,3).float()
        y_true = y.to(device)

        optimizer.zero_grad()

        y_hat = model(x)
        if len(list(y_hat.size())) < 2:
            y_hat = torch.unsqueeze(y_hat, 0)

        cross_entropy_loss = cross_entropy_loss_criterion(y_hat, y_true)
        
        epoch_loss += cross_entropy_loss.item()

        # Backward pass
        cross_entropy_loss.backward()
        optimizer.step()

        # accuracy
        pbar.set_description(f'Acc = : {get_accuracy(y_hat, y_true):.4f}')

    epoch_loss = epoch_loss / len(loader)
    
    return model, optimizer, epoch_loss


# validate 
def validate(loader, batch_size, model, cross_entropy_loss_criterion, device):
    '''
    Function for the validation step of the training loop
    '''
    print("start validation epoch...")
    model.eval()
    epoch_loss = 0

    batches_accuracy = []
    pbar = tqdm(loader)
    for x,y in pbar:
        x = x.to(device).transpose(1,3).transpose(2,3).float()
        y_true = y.to(device)

        y_hat = model(x)

        if len(list(y_hat.size())) < 2:
            y_hat = torch.unsqueeze(y_hat, 0)

        cross_entropy_loss = cross_entropy_loss_criterion(y_hat, y_true) 

        epoch_loss += cross_entropy_loss.item()

        # accuracy
        accuracy = get_accuracy(y_hat, y_true)
        pbar.set_description(f'Acc = : {accuracy:.4f}')
        batches_accuracy.append(accuracy)

    epoch_loss = epoch_loss / len(loader)
    
    return model, epoch_loss, mean(batches_accuracy)


def training_loop(model, cross_entropy_loss_criterion, batch_size, optimizer, scheduler, epochs, train_loader,
                                                             test_loader, device, print_every=1):
    '''
    Function defining the entire training loop
    '''

    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    
    # delete contents of out.txt
    with open('training_log.txt', 'w') as outfile:
        outfile.write('')
    # Train model
    for epoch in range(epochs):

        # training
        model, optimizer, train_loss = train(train_loader, batch_size, model, cross_entropy_loss_criterion, optimizer, device)
        train_losses.append(train_loss)
        

        if epoch % print_every == (print_every - 1):
            # validation
            with torch.no_grad():
                model, valid_loss, valid_accuracy = validate(test_loader, batch_size, model, cross_entropy_loss_criterion, device)
                valid_losses.append(valid_loss)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Valid acc: {valid_accuracy:.4f}\t'
                  )
            
            with open('training_log.txt', 'a+') as outfile:
                outfile.write(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Valid acc: {valid_accuracy:.4f}\t'
                  )
                outfile.write('\n')
                
        torch.save(model, 'model.pkl')
        
        scheduler.step()

    return model, optimizer, train_losses, valid_losses



