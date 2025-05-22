python
from collections import Counter
import numpy as np

class Nodo:
    """Represents a single node in the decision tree."""
    def __init__(self, caracteristica=None, umbral=None, izquierda=None, derecha=None, *, valor=None):
        self.caracteristica = caracteristica
        self.umbral = umbral
        self.izquierda = izquierda
        self.derecha = derecha
        self.valor = valor

    def es_nodo_hoja(self):
        """Check if the node is a leaf node."""
        return self.valor is not None


class ClasificadorArbolDecision:
    """
    A simple implementation of a Decision Tree Classifier.
    
    Parameters:
    -----------
    max_profundidad : int
        The maximum depth of the tree.
    """
    
    def __init__(self, max_profundidad=None):
        self.max_profundidad = max_profundidad
        self.arbol = None

    def ajustar(self, X, y):
        """
        Build the decision tree classifier from the training set (X, y).
        
        Parameters:
        -----------
        X : np.ndarray
            Training features, shape (n_samples, n_features).
        y : np.ndarray
            Training labels, shape (n_samples,).
        """
        self.arbol = self._construir_arbol(X, y)

    def predecir(self, X):
        """
        Predict class for X.
        
        Parameters:
        -----------
        X : np.ndarray
            Samples, shape (n_samples, n_features).
        
        Returns:
        --------
        np.ndarray
            Predicted classes, shape (n_samples,).
        """
        return np.array([self._traverse_arbol(x, self.arbol) for x in X])

    def _construir_arbol(self, X, y, profundidad=0):
        """
        Recursively build the decision tree.
        
        Parameters:
        -----------
        X : np.ndarray
            Features for the current node.
        y : np.ndarray
            Labels for the current node.
        profundidad : int
            The depth of the current node.
        
        Returns:
        --------
        Node
            The root node of the decision tree.
        """
        num_muestras, num_caracteristicas = X.shape
        num_etiquetas = len(Counter(y))

        # Stopping criteria
        if num_etiquetas == 1 or (self.max_profundidad is not None and profundidad >= self.max_profundidad):
            valor_nodo_hoja = Counter(y).most_common(1)[0][0]
            return Nodo(valor=valor_nodo_hoja)

        # Find the best split
        mejor_caracteristica, mejor_umbral = self._mejor_split(X, y, num_caracteristicas)
        if mejor_caracteristica is None:
            valor_nodo_hoja = Counter(y).most_common(1)[0][0]
            return Nodo(valor=valor_nodo_hoja)

        # Grow the children recursively
        indices_izquierda = X[:, mejor_caracteristica] <= mejor_umbral
        izquierda = self._construir_arbol(X[indices_izquierda], y[indices_izquierda], profundidad + 1)
        derecha = self._construir_arbol(X[~indices_izquierda], y[~indices_izquierda], profundidad + 1)
        return Nodo(caracteristica=mejor_caracteristica, umbral=mejor_umbral, izquierda=izquierda, derecha=derecha)

    def _mejor_split(self, X, y, num_caracteristicas):
        """
        Find the best feature and threshold to split on using Gini impurity.
        
        Parameters:
        -----------
        X : np.ndarray
            Features for the current node.
        y : np.ndarray
            Labels for the current node.
        num_caracteristicas : int
            The number of features to consider for splitting.
        
        Returns:
        --------
        Tuple[int, float]
            The best feature index and the best threshold to split on.
        """
        mejor_caracteristica, mejor_umbral, mejor_gini = None, None, 1

        for caracteristica in range(num_caracteristicas):
            umbrales, clases = zip(*sorted(zip(X[:, caracteristica], y)))
            num_izquierda = Counter()
            num_derecha = Counter(y)
            
            for i in range(1, len(y)):
                c = clases[i - 1]
                num_izquierda[c] += 1
                num_derecha[c] -= 1

                gini_izquierda = self._gini_impureza(num_izquierda, i)
                gini_derecha = self._gini_impureza(num_derecha, len(y) - i)
                gini = (i * gini_izquierda + (len(y) - i) * gini_derecha) / len(y)

                if umbrales[i] == umbrales[i - 1]:
                    continue
                if gini < mejor_gini:
                    mejor_caracteristica, mejor_umbral, mejor_gini = caracteristica, umbrales[i], gini
        return mejor_caracteristica, mejor_umbral

    def _gini_impureza(self, conteo_clases, total_conteo):
        """
        Calculate the Gini impurity for a node.
        
        Parameters:
        -----------
        class_counts : Counter
            Counts of each class in the node.
        total_count : int
            Total number of samples in the node.
        
        Returns:
        --------
        float
            Gini impurity of the node.
        """
        return 1.0 - sum((count / total_conteo) ** 2 for count in conteo_clases.values())

    def _traverse_arbol(self, x, nodo):
        """
        Traverse the tree to make a prediction.
        
        Parameters:
        -----------
        x : np.ndarray
            Single sample for which prediction is to be made.
        nodo : Node
            The current node in the tree.
        
        Returns:
        --------
        int
            Predicted class label.
        """
        if nodo.es_nodo_hoja():
            return nodo.valor
        if x[nodo.caracteristica] <= nodo.umbral:
            return self._traverse_arbol(x, nodo.izquierda)
        return self._traverse_arbol(x, nodo.derecha)


# Example usage
if __name__ == "__main__":
    # Toy dataset
    X_entrenamiento = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_entrenamiento = np.array([0, 0, 1, 1])

    # Initialize and fit the model
    clf = ClasificadorArbolDecision(max_profundidad=3)
    clf.ajustar(X_entrenamiento, y_entrenamiento)

    # Predictions
    X_prueba = np.array([[1.5, 2.5], [3.5, 4.5]])
    predicciones = clf.predecir(X_prueba)
    print("Predictions:", predicciones)