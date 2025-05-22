import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import psutil

class ModelUtils:

    @staticmethod
    def calculate_inference_speed(model, input_data, device='cpu', iterations=100):
        model.to(device)
        input_data = input_data.to(device)
        model.eval()

        with torch.no_grad():
            _ = model(input_data)

        start_time = time.time()
        with torch.no_grad():
            count = 0
            while count < iterations:
                _ = model(input_data)
                count += 1
        end_time = time.time()

        avg_time = (end_time - start_time) / iterations
        print(f"Average inference time per input: {avg_time:.6f} seconds")
        return avg_time

    @staticmethod
    def calculate_metrics(y_true_labels, predicted_labels):
        acc = accuracy_score(y_true_labels, predicted_labels)
        precision = precision_score(y_true_labels, predicted_labels, average='weighted')
        recall = recall_score(y_true_labels, predicted_labels, average='weighted')
        f1 = f1_score(y_true_labels, predicted_labels, average='weighted')
        cm = confusion_matrix(y_true_labels, predicted_labels)

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        return acc, precision, recall, f1, cm

    @staticmethod
    def calculate_model_memory(model):
        param_size = 0
        for parameter in model.parameters():
            param_size += parameter.nelement() * parameter.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = (param_size + buffer_size) / (1024 ** 2)
        print(f"Model size: {total_size:.2f} MB")
        return total_size

    @staticmethod
    def summarize_model(model):
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_non_trainable_params = num_params - num_trainable_params

        print(f"Number of layers: {len(list(model.children()))}")
        print(f"Total parameters: {num_params}")
        print(f"Trainable parameters: {num_trainable_params}")
        print(f"Non-trainable parameters: {num_non_trainable_params}")
        
        return {
            "layers": len(list(model.children())),
            "total_params": num_params,
            "trainable_params": num_trainable_params,
            "non_trainable_params": num_non_trainable_params
        }

    @staticmethod
    def plot_learning_curves(train_accuracy, val_accuracy, train_loss, val_loss):
        epochs = range(1, len(train_accuracy) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_accuracy, label='Training Accuracy')
        plt.plot(epochs, val_accuracy, label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_precision_recall(y_true_labels, y_scores):
        precision, recall, _ = precision_recall_curve(y_true_labels, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.show()