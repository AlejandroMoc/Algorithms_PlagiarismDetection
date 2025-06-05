# Import necessary libraries
import panda as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
link = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
info = pd.read_csv(link)

# Preprocess the data
info = info[['Survived', 'Pclass', 'Gender', 'Age', 'Price']]
info = info.dropna()
info['Gender'] = info['Gender'].map({'male': 0, 'female': 1})
X_data = info[['Pclass', 'Gender', 'Age', 'Price']]
y_data = info['Survived']

# Split the data into training and testing sets
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
machine = DecisionTreeClassifier()
machine.fit(X_train_data, y_train_data)
y_prediction = machine.predict(X_test_data)
correctness = accuracy_score(y_test_data, y_prediction)
print(f'Correctness: {correctness:.2f}')


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plot_tree(machine, feature_names=X_data.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.show()


import interactive_decision_tree as idt  # noqa: E402
idt.create_tree(tree_model=machine,
                X=X_train_data,
                target_names=['Not Survived', 'Survived'],
                target_colors=['red', 'green'],
                save_path='titanic-tree.html')

idt.create_sankey(tree_model=machine,
                X=X_train_data,
                target_names=['Not Survived', 'Survived'],
                target_colors=['red', 'green'],
                save_path='titanic-sankey.html')