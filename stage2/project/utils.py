# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 16:27:59 2025

@author: admin
"""

# utils.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def save_model(model, path):
    """Saves the model's state dictionary to a specified path.
    
    Args:
        model: The model to be saved.
        path (str): The file path where the model will be saved.
    """
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(model, path):
    """Loads the model's state dictionary from a specified path.
    
    Args:
        model: The model to load the state dictionary into.
        path (str): The file path from which the model will be loaded.
    
    Returns:
        model: The model with the loaded state dictionary.
    """
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f'Model loaded from {path}')
    return model

def plot_loss_accuracy(history):
    """Plots the training and validation loss and accuracy over epochs.
    
    Args:
        history (dict): A dictionary containing loss and accuracy values.
    """
    epochs = range(1, len(history['loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def calculate_metrics(y_true, y_pred):
    """Calculates and prints accuracy, confusion matrix, and classification report.
    
    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
    """
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

def create_dir_if_not_exists(directory):
    """Checks if a directory exists; if not, creates it.
    
    Args:
        directory (str): The directory path to check or create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Directory created: {directory}')
    else:
        print(f'Directory already exists: {directory}')
