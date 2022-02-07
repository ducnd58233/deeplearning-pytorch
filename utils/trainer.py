import torch
import torch.nn as nn
from tqdm import tqdm
from time import sleep

def train(model, data_loader, optimizer, loss_criteria, num_classes = 1):
    model.train()
    train_loss = 0
    correct = 0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch = 0
    length = len(data_loader.dataset)   
    
    with tqdm(data_loader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description("Train")      
            batch += 1
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            out = model(data).to(device)
            loss = loss_criteria(out, target).to(device)
            train_loss += loss.item()

            if (num_classes == 1):
                _, predicted = torch.max(out.data, 1)
                correct += torch.sum(target == predicted).item()
            else:
                predicted = out.argmax(dim=1)
                correct += torch.sum(target == predicted)

            loss.backward()
            optimizer.step()
            
            acc = 100. * correct / length
            guess = f"{correct}/{length}"
            string = f"loss: {100. * loss:.6f}%, accuracy: {acc:.6f}% [{guess}]" 
            tepoch.set_postfix_str(string)
            sleep(0.01)
     
    avg_loss = train_loss / (batch + 1)  
    return avg_loss
        
        
def test(model, data_loader, loss_criteria, num_classes = 1, type = "Test"):
    model.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch = 0
    length = len(data_loader.dataset)
    
    with tqdm(data_loader, unit="batch") as tepoch:
        with torch.no_grad():
            for data, target in tepoch:
                tepoch.set_description(type)
                batch += 1
                data = data.to(device)
                target = target.to(device)
                
                out = model(data).to(device)
                loss = loss_criteria(out, target).to(device)
                test_loss += loss.item()
                
                if (num_classes == 1):
                    _, predicted = torch.max(out.data, 1)
                    correct += torch.sum(target == predicted).item()
                else:
                    predicted = out.argmax(dim=1)
                    correct += torch.sum(target == predicted)
                    
                
                acc = 100. * correct / length
                guess = f"{correct}/{length}"
                string = f"loss: {100. * loss:.6f}%, accuracy: {acc:.6f}% [{guess}]"   
                tepoch.set_postfix_str(string)
                sleep(0.01)
                
    avg_loss = test_loss / (batch + 1)           
    return avg_loss