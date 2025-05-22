# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
source_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data_set = pd.read_csv(source_url)

# Preprocess the data
processed_data = data_set[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
processed_data = processed_data.dropna()
processed_data['Sex'] = processed_data['Sex'].map({'male': 0, 'female': 1})
features = processed_data[['Pclass', 'Sex', 'Age', 'Fare']]
target = processed_data['Survived']

# Split the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)
model_decision_tree = DecisionTreeClassifier()
model_decision_tree.fit(features_train, target_train)
target_predicted = model_decision_tree.predict(features_test)
accuracy_result = accuracy_score(target_test, target_predicted)
print(f'Accuracy: {accuracy_result:.2f}')

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plot_tree(model_decision_tree, feature_names=features.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.show()

import interactive_decision_tree as idt  # noqa: E402
idt.create_tree(tree_model=model_decision_tree,
                X=features_train,
                target_names=['Not Survived', 'Survived'],
                target_colors=['red', 'green'],
                save_path='titanic-tree.html')

idt.create_sankey(tree_model=model_decision_tree,
                X=features_train,
                target_names=['Not Survived', 'Survived'],
                target_colors=['red', 'green'],
                save_path='titanic-sankey.html')