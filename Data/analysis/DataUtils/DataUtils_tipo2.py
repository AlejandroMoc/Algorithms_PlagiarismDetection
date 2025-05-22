import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

class DataProcessor:

    @staticmethod
    def acquire_and_inspect_data(file_path, rows=5):
        data_set = pd.read_csv(file_path)
        print(f"First {rows} rows of the dataset:")
        print(data_set.head(rows))
        return data_set

    @staticmethod
    def fundamental_numerics(data_set):
        print("Fundamental Numerics:")
        stats_summary = data_set.describe()
        print(stats_summary)
        return stats_summary

    @staticmethod
    def examine_missing_values(data_set):
        missing_values = data_set.isnull().sum().sort_values(ascending=False)
        missing_percentage = (missing_values / len(data_set)) * 100
        missing_info = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
        print(missing_info)

        # Plot missing data
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_info.index, y=missing_info['Percentage'])
        plt.title("Percentage of Missing Data by Feature")
        plt.xticks(rotation=90)
        plt.ylabel('Percentage')
        plt.show()

        return missing_info

    @staticmethod
    def examine_data_distribution(data_set, feature):
        plt.figure(figsize=(8, 6))
        sns.histplot(data_set[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def analyze_correlation(data_set, method='pearson'):
        correlation_matrix = data_set.corr(method=method)
        print(f"Correlation Matrix ({method}):")
        print(correlation_matrix)

        # Heatmap visualization
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title(f'Correlation Matrix ({method})')
        plt.show()

        return correlation_matrix

    @staticmethod
    def detect_anomalies_z_score(data_set, threshold=3):
        z_scores = np.abs(stats.zscore(data_set.select_dtypes(include=[np.number])))
        anomalies = np.where(z_scores > threshold)
        print(f"Anomalies detected using Z-score (threshold={threshold}):")
        print(anomalies)
        return anomalies

    @staticmethod
    def split_training_testing_data(data_set, target_column, test_size=0.2, random_state=42):
        X_data = data_set.drop(columns=[target_column])
        y_data = data_set[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=random_state)
        print(f"Data successfully split: {len(X_train)} training samples, {len(X_test)} testing samples")
        return X_train, X_test, y_train, y_test

    @staticmethod
    def encode_labels(data_set, columns):
        encoder = LabelEncoder()
        for col in columns:
            data_set[col] = encoder.fit_transform(data_set[col].astype(str))
        print(f"Label encoding applied to columns: {columns}")
        return data_set

    @staticmethod
    def scale_features(X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Feature scaling completed.")
        return X_train_scaled, X_test_scaled

    @staticmethod
    def plot_category(data_set, feature):
        plt.figure(figsize=(8, 6))
        sns.countplot(x=feature, data=data_set)
        plt.title(f'Count Plot of {feature}')
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def correlate_with_target(data_set, target_column):
        correlation = data_set.corr()[target_column].sort_values(ascending=False)
        print(f"Correlation of features with the target '{target_column}':")
        print(correlation)

        # Plotting the correlation with target
        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlation.index, y=correlation.values)
        plt.title(f'Correlation of Features with {target_column}')
        plt.xticks(rotation=90)
        plt.show()

        return correlation