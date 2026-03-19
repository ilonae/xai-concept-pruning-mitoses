import copy
import torch
import os
import cv2
import sys
import torchvision


from torchvision import transforms
from torch import nn
from sklearn.model_selection import train_test_split
from modules.utils import create_classification_report, measure_global_sparsity

sys.path.append('code')
from models.pytorch_models import PretrainedCNN
from modules.concept_attribution import retrieve_prunable_concept_diff, retrieve_prunable_true_concepts
from modules.concept_visualization import gen_heatmaps_concepts, print_classification
from modules.concept_pruning import prune_selected_concepts
from modules.lamb import Lamb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.cuda.empty_cache()


def evaluate_model(eval_model):
    # All Mitoses
    path = 'examples_ds_from_MP.train.HTW.train/True'
    model_pred =0
    for file in os.listdir(path):
        path_to_file = os.path.join(path,file)  
        img = cv2.imread(path_to_file)
        # Preprocess for the model
        original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = original[144:144+224,144:144+224]
        img = img2 / 255.
        img -= [0.86121925, 0.86814779, 0.88314296] # MEAN-Values
        img /= [0.13475281, 0.10909398, 0.09926313] # STD-Values
        imgtensor = torch.tensor(img.swapaxes(1,2).swapaxes(0,1),dtype=torch.float32)[None]
        pred = eval_model.predict(imgtensor.to(device),logits=True).detach().cpu()
        #print(pred)
        pred = float(torch.nn.functional.softmax(pred, dim=1)[0,1].numpy())
        #print(pred)
        model_pred+=pred      
    model_pred = model_pred / len(os.listdir(path))
    return model_pred

def finetune_model(train_loader,test_loader,model,optimizer,epochs):
    
    train_len = len(train_loader)    
    test_len = len(test_loader)
    for epoch in range(epochs):
        train_loss = 0.0
        total_val_loss = 0.0
        model_pred =0
        # Training the model
        model.train()
        counter = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            log_prob = torch.nn.functional.log_softmax(output, dim=1)
            loss = torch.nn.functional.nll_loss(log_prob, labels)
            #optimizer.zero_grad()
            #outputs = model.forward(inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        # Evaluating the model
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)  
                
                
                output = model(inputs)
                val_log_prob = torch.nn.functional.log_softmax(output, dim=1)
                val_loss = torch.nn.functional.nll_loss(val_log_prob, labels)
            
                total_val_loss += val_loss.item() * inputs.size(0)

        train_loss = train_loss/train_len
        valid_loss = total_val_loss/test_len
        print('[%d] Training Loss: %.6f, Validation Loss: %.6f'  % (epoch + 1, train_loss, valid_loss))
    torch.cuda.empty_cache()
    return model

def dataset_split(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = torch.utils.data.Subset(dataset, train_idx)
    datasets['val'] = torch.utils.data.Subset(dataset, val_idx)
    return datasets


def init_datasets(path):
    
    train_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0),
                                        #transforms.Resize(317),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomRotation(30),
                                        #transforms.CenterCrop(224),
                                        transforms.Normalize(
                                            mean=[0.86121925, 0.86814779, 0.88314296], 
                                            std=[0.13475281, 0.10909398, 0.09926313])
                                        ])

    test_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        #transforms.Resize(317),
                                        # transforms.CenterCrop(224),
                                        transforms.Normalize(
                                            mean=[0.86121925, 0.86814779, 0.88314296], 
                                            std=[0.13475281, 0.10909398, 0.09926313]),
                                       ])


    train_data = torchvision.datasets.ImageFolder(path, transform=train_transforms)
    test_data = torchvision.datasets.ImageFolder(path, transform=test_transforms)

    datasets = dataset_split(train_data)

    dataloaders = {x:torch.utils.data.DataLoader(datasets[x],32, shuffle=True, num_workers=4) for x in ['train','val']}

    test_loader = torch.utils.data.DataLoader(test_data)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    return train_loader, val_loader,test_loader




if __name__ == "__main__":

    state_dict = torch.load('state_dict.pth', weights_only=False)['models_state_dict']
    path = 'examples_ds_from_MP.train.HTW.train'
    model = PretrainedCNN('vgg', num_classes=2)
    model.load_state_dict(state_dict[0])
    model = model.eval()
    #print(model)
    optimizer = Lamb(model.parameters(), lr=0.00025, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    model.to(device)

    train_loader, val_loader, test_loader = init_datasets(path)
     
    f = open('./result_log.txt', 'w') 
    f.write("Concept Pruning log")
    f.write("\n=========================================================================================")
    pred= evaluate_model(model)
    print("\nInitial pred: {}".format(pred))
    f.write("\nInitial pred: {}".format(pred))
    num_zeros, num_elements, sparsity = measure_global_sparsity( model,weight=True,bias=False,conv2d_use_mask=False,linear_use_mask=False)
    classification_report = create_classification_report(model=model, test_loader=test_loader, device=device)
    print("Classification Report:")
    f.write("\nClassification Report: {}".format(classification_report))
    print(classification_report)
    #print_classification(classification_report)
    print("Global Sparsity:")
    f.write("\nGlobal Sparsity: {}".format(sparsity))
    print("{:.2f}".format(sparsity))
    pruned_model = copy.deepcopy(model)

    for i in range(0,10):
        f.write("\n=========================================================================================")
        f.write("\niteration {} , finding irrelevant concepts...".format(i))
        prunable_channels=retrieve_prunable_true_concepts(i,False, pruned_model, device)
        #prunable_channels={'features.0': [0, 8, 15, 44, 27],
        #                    'features.4': [0, 8, 15, 44, 2,
        #                    80, 120, 42, 100],
        #                    'features.8': [0, 8, 15, 44, 27],
        #                    'features.11': [0, 8, 15, 44, 27],
        #                    'classifier.0': [0, 8, 15, 44, 27],
        #                    'classifier.3': [0, 8, 15, 44, 27],
        #                    'features.22': [0, 8, 15, 44, 27]}
        print("Voted least relevant concept ids to prune (max vote ids): {}".format(prunable_channels))
        f.write("\nVoted least relevant concept ids to prune (max vote ids): {}".format(prunable_channels))
    
        #print("Visualising irrelevant concepts...")
        #f.write("Visualising irrelevant concepts...")
        #gen_heatmaps_concepts( prunable_channels,i,device,pruned_model)
        
        pruned_model = prune_selected_concepts(prunable_channels,pruned_model)
        pred = evaluate_model(pruned_model)
        print("iteration {} , pred after pruning:  {}".format(i, pred))
        f.write("\niteration {} , pred after pruning:  {}".format(i, pred))
        pruned_model = finetune_model(train_loader,val_loader,pruned_model,optimizer,epochs)
        #model = prune_selected_concepts(prunable_channels,pruned_model)
        pred = evaluate_model(pruned_model)
        print("iteration {} , pred after pruning AND finetuning:  {}".format(i, pred))
        f.write("\niteration {} , pred after pruning AND finetuning:  {}".format(i, pred))
        #classification_report = create_classification_report(model=pruned_model, test_loader=test_loader, device=device)
        #print_classification(classification_report)
        num_zeros, num_elements, sparsity = measure_global_sparsity( pruned_model,weight=True,bias=False,conv2d_use_mask=False,linear_use_mask=False)
        #print("Classification Report:")
        #print(classification_report)
        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))
        f.write("\n=========================================================================================")
    f.close()
    #print_classification(classification_report)


