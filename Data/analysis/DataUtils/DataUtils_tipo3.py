import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

class DataUtils:

    @staticmethod
    def load_and_preview(file_path, rows=5):
        data = pd.read_csv(file_path)
        print(f"First {rows} rows of the dataset:")
        print(data.head(rows))
        return data

    @staticmethod
    def calculate_basic_statistics(data):
        print("Basic Statistics:")
        stats_summary = data.describe()
        print(stats_summary)
        return stats_summary

    @staticmethod
    def analyze_missing_data(data):
        missing_data = data.isnull().sum().sort_values(ascending=False)
        missing_percent = (missing_data / len(data)) * 100
        missing_info = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percent})
        print(missing_info)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_info.index, y=missing_info['Percentage'])
        plt.title("Percentage of Missing Data by Feature")
        plt.xticks(rotation=90)
        plt.ylabel('Percentage')
        plt.show()

        return missing_info

    @staticmethod
    def analyze_data_distribution(data, feature):
        plt.figure(figsize=(8, 6))
        sns.histplot(data[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def perform_correlation_analysis(data, method='pearson'):
        correlation_matrix = data.corr(method=method)
        print(f"Correlation Matrix ({method}):")
        print(correlation_matrix)

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title(f'Correlation Matrix ({method})')
        plt.show()

        return correlation_matrix

    @staticmethod
    def detect_outliers_z(data, threshold=3):
        z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
        outliers = np.where(z_scores > threshold)
        print(f"Outliers detected using Z-score (threshold={threshold}):")
        print(outliers)
        return outliers

    @staticmethod
    def split_data(data, target_col, test_size=0.2, random_state=42):
        X = data.drop(columns=[target_col])
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print(f"Data successfully split: {len(X_train)} training samples, {len(X_test)} testing samples")
        return X_train, X_test, y_train, y_test

    @staticmethod
    def encode_labels(data, columns):
        encoder = LabelEncoder()
        for col in columns:
            data[col] = encoder.fit_transform(data[col].astype(str))
        print(f"Label encoding applied to columns: {columns}")
        return data

    @staticmethod
    def scale_features(X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Feature scaling completed.")
        return X_train_scaled, X_test_scaled

    @staticmethod
    def plot_categorical(data, feature):
        plt.figure(figsize=(8, 6))
        sns.countplot(x=feature, data=data)
        plt.title(f'Count Plot of {feature}')
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def correlation_with_target_variable(data, target_col):
        correlation = data.corr()[target_col].sort_values(ascending=False)
        print(f"Correlation of features with the target '{target_col}':")
        print(correlation)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlation.index, y=correlation.values)
        plt.title(f'Correlation of Features with {target_col}')
        plt.xticks(rotation=90)
        plt.show()

        return correlation