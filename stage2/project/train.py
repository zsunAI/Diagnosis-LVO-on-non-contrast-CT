import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision.models import resnet18
from tqdm import tqdm
import os
import gc
from dataset import NiftiDataset
from McResnet18 import ResNet,BasicBlock,resnet18s

# 
train_df = pd.read_csv('...YOURDIR/data/train_labels.csv')
val_df = pd.read_csv('...YOURDIR/data/val_labels.csv')
train_df['Patient ID'] = train_df['Patient ID'].astype(str)
val_df['Patient ID'] = val_df['Patient ID'].astype(str)

# data dir
positive_dir = '...YOURDIR/data/train_nii_close'
negative_dir = '...YOURDIR/data/train_nii_nc'
best_model_path = '...YOURDIR/ncct/output/best_model.pth'

# Set parameters
learning_rate = 0.00001
l2_reg = 0.001
num_epochs = 50
batch_size = 2
num_classes = 1  # 2 class
target_dim = (512, 512, 200)

# Create datasets and data loaders
train_dataset = NiftiDataset(train_df, positive_dir, negative_dir,target_dim)
val_dataset = NiftiDataset(val_df, positive_dir, negative_dir,target_dim)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def initialize_model_with_custom_weights(num_classes, fc_modify=False, best_model_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18s(num_classes=num_classes) # Initialize the model
    # init...
    
    if fc_modify:
        # If it exists, load pre trained weights
        pretrained_model = resnet18s()  # Initialize a non-specialized resnet18s for weights extraction
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()

        # Only retain layers that match the current model and do not overwrite specific custom layers(’fc‘)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('fc')}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    if best_model_path and os.path.exists(best_model_path):
        # Use a utility to handle different model states
        state_dict = torch.load(best_model_path)
        
        # Fix module prefix issues
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)  # Use strict=False to handle unexpected/missing keys
        print("Loaded model from best model path.")

    # Distribute the model to multiple GPUs (if available)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    model.to(device)

    return model


model = initialize_model_with_custom_weights(
    num_classes=1,
    fc_modify=True, # False
    best_model_path=best_model_path
)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
scaler = torch.cuda.amp.GradScaler()
best_accuracy = 0.0

# train
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    for inputs, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1) if num_classes == 1 else labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if num_classes == 1:  # binary classification
            predictions = torch.round(torch.sigmoid(outputs))
        else:  # Multi category classification
            _, predictions = torch.max(outputs, 1)

        correct_predictions += (predictions.squeeze() == labels.squeeze()).sum().item()
        total_samples += labels.size(0)

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / total_samples
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

    # val
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_probs = []  # Storage probability
    all_preds = []  # Store all predicted labels
    all_ids = []
    all_labels = []
    patient_ids = []
    with torch.no_grad():
        for inputs, labels, patient_ids in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1) if num_classes == 1 else labels)

            val_loss += loss.item()

            # Calculate probability
            probs = torch.sigmoid(outputs) if num_classes == 1 else torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            # Calculate predicted labels
            predicted = (probs > 0.5).float() if num_classes == 1 else torch.argmax(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())

            all_ids.extend(patient_ids)
            all_labels.extend(labels.cpu().numpy())

            correct_predictions += (predicted.squeeze() == labels.squeeze()).sum().item()
            total_predictions += labels.size(0)  

    # save
    os.makedirs('.../ncct/output/excel/', exist_ok=True)
    results_df = pd.DataFrame({
    'Patient ID': all_ids,
    'Ground Truth': all_labels,
    'Probability': [prob[0] if num_classes == 1 else prob for prob in all_probs],
    'Prediction': [pred[0] if num_classes == 1 else pred for pred in all_preds]
    })
    results_df.to_csv(f'.../ncct/output/excel/validation_results_epoch_{epoch+1}.csv', index=False)

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved best model with accuracy: {best_accuracy:.4f}')

    torch.cuda.empty_cache()
    gc.collect()

print('Training complete.')
