import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import psutil

class FunkyUtils:

    @staticmethod
    def measure_speedy_inference(machine, data_tensor, device='cpu', iterations=100):
        """
        Function to calculate the inference speed of a model.
        """
        machine.to(device)
        data_tensor = data_tensor.to(device)
        machine.eval()  # Set the model to evaluation mode

        # Warm up the model
        with torch.no_grad():
            _ = machine(data_tensor)

        # Timing the inference
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = machine(data_tensor)
        end_time = time.time()

        avg_inference_time = (end_time - start_time) / iterations
        print(f"Average inference time per input: {avg_inference_time:.6f} seconds")
        return avg_inference_time

    @staticmethod
    def compute_metrics_rare(y_truth, y_prediction):
        """
        Function to calculate accuracy, precision, recall, F1-score, and confusion matrix.
        """
        acc = accuracy_score(y_truth, y_prediction)
        precision = precision_score(y_truth, y_prediction, average='weighted')
        recall = recall_score(y_truth, y_prediction, average='weighted')
        f1 = f1_score(y_truth, y_prediction, average='weighted')
        cm = confusion_matrix(y_truth, y_prediction)

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        return acc, precision, recall, f1, cm

    @staticmethod
    def estimate_model_mass(machine):
        """
        Function to calculate the size of the model in terms of memory usage.
        """
        param_size = 0
        for param in machine.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in machine.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB
        print(f"Model size: {total_size:.2f} MB")
        return total_size

    @staticmethod
    def machine_summary(machine):
        """
        Function to generate a summary of the model's parameters (layers, total parameters, trainable vs non-trainable).
        """
        num_params = sum(p.numel() for p in machine.parameters())
        num_trainable_params = sum(p.numel() for p in machine.parameters() if p.requires_grad)
        num_non_trainable_params = num_params - num_trainable_params

        print(f"Number of layers: {len(list(machine.children()))}")
        print(f"Total parameters: {num_params}")
        print(f"Trainable parameters: {num_trainable_params}")
        print(f"Non-trainable parameters: {num_non_trainable_params}")
        
        return {
            "layers": len(list(machine.children())),
            "total_params": num_params,
            "trainable_params": num_trainable_params,
            "non_trainable_params": num_non_trainable_params
        }

    @staticmethod
    def visualize_learning_curve(train_accuracy, val_accuracy, train_error, val_error):
        """
        Function to plot learning curves for training and validation accuracy/loss over time.
        """
        epochs = range(1, len(train_accuracy) + 1)

        plt.figure(figsize=(12, 5))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_accuracy, label='Training Accuracy')
        plt.plot(epochs, val_accuracy, label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_error, label='Training Loss')
        plt.plot(epochs, val_error, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_precision_recall_curve(y_truth, y_scores):
        """
        Function to plot the precision-recall curve, which is useful for imbalanced datasets.
        """
        precision, recall, _ = precision_recall_curve(y_truth, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.show()
