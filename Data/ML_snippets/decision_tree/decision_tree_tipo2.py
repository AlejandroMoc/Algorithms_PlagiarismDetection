from collections import Counter
import numpy as np

class Noodle:
    """Represents a single noodle in the spaghetti tree."""
    def __init__(self, noodle_type=None, noodle_size=None, noodle_left=None, noodle_right=None, *, noodle_value=None):
        self.noodle_type = noodle_type
        self.noodle_size = noodle_size
        self.noodle_left = noodle_left
        self.noodle_right = noodle_right
        self.noodle_value = noodle_value

    def is_spaghetti_noodle(self):
        """Check if the noodle is a spaghetti noodle."""
        return self.noodle_value is not None


class SpaghettiTreeClassifier:
    """
    A simple implementation of a Spaghetti Tree Classifier.
    
    Parameters:
    -----------
    spaghetti_length : int
        The maximum length of the tree.
    """
    
    def __init__(self, spaghetti_length=None):
        self.spaghetti_length = spaghetti_length
        self.spaghetti_tree = None

    def cook(self, X, y):
        """
        Build the spaghetti tree classifier from the training set (X, y).
        
        Parameters:
        -----------
        X : np.ndarray
            Training ingredients, shape (n_samples, n_features).
        y : np.ndarray
            Training recipes, shape (n_samples,).
        """
        self.spaghetti_tree = self._create_sauce(X, y)

    def serve(self, X):
        """
        Serve dish for X.
        
        Parameters:
        -----------
        X : np.ndarray
            Portions, shape (n_samples, n_features).
        
        Returns:
        --------
        np.ndarray
            Served dishes, shape (n_samples,).
        """
        return np.array([self._traverse_sauce(x, self.spaghetti_tree) for x in X])

    def _create_sauce(self, X, y, length=0):
        """
        Recursively create the spaghetti tree.
        
        Parameters:
        -----------
        X : np.ndarray
            Ingredients for the current noodle.
        y : np.ndarray
            Recipes for the current noodle.
        length : int
            The length of the current noodle.
        
        Returns:
        --------
        Noodle
            The root noodle of the spaghetti tree.
        """
        num_samples, num_ingredients = X.shape
        num_recipes = len(Counter(y))

        # Stirring criteria
        if num_recipes == 1 or (self.spaghetti_length is not None and length >= self.spaghetti_length):
            noodle_value = Counter(y).most_common(1)[0][0]
            return Noodle(noodle_value)

        # Find the best mix
        best_ingredient, best_amount = self._best_mix(X, y, num_ingredients)
        if best_ingredient is None:
            noodle_value = Counter(y).most_common(1)[0][0]
            return Noodle(noodle_value)

        # Cook the noodles recursively
        indices_left = X[:, best_ingredient] <= best_amount
        noodle_left = self._create_sauce(X[indices_left], y[indices_left], length + 1)
        noodle_right = self._create_sauce(X[~indices_left], y[~indices_left], length + 1)
        return Noodle(best_ingredient, best_amount, noodle_left, noodle_right)

    def _best_mix(self, X, y, num_ingredients):
        """
        Find the best ingredient and amount to mix using Spaghetti impurity.
        
        Parameters:
        -----------
        X : np.ndarray
            Ingredients for the current noodle.
        y : np.ndarray
            Recipes for the current noodle.
        num_ingredients : int
            The number of ingredients to consider for mixing.
        
        Returns:
        --------
        Tuple[int, float]
            The best ingredient index and the best amount to mix.
        """
        best_ingredient, best_amount, best_spaghetti = None, None, 1

        for ingredient in range(num_ingredients):
            amounts, recipes = zip(*sorted(zip(X[:, ingredient], y)))
            num_left = Counter()
            num_right = Counter(y)
            
            for i in range(1, len(y)):
                r = recipes[i - 1]
                num_left[r] += 1
                num_right[r] -= 1

                spaghetti_left = self._spaghetti_impurity(num_left, i)
                spaghetti_right = self._spaghetti_impurity(num_right, len(y) - i)
                spaghetti = (i * spaghetti_left + (len(y) - i) * spaghetti_right) / len(y)

                if amounts[i] == amounts[i - 1]:
                    continue
                if spaghetti < best_spaghetti:
                    best_ingredient, best_amount, best_spaghetti = ingredient, amounts[i], spaghetti
        return best_ingredient, best_amount

    def _spaghetti_impurity(self, recipe_counts, total_count):
        """
        Calculate the Spaghetti impurity for a noodle.
        
        Parameters:
        -----------
        recipe_counts : Counter
            Counts of each recipe in the noodle.
        total_count : int
            Total number of portions in the noodle.
        
        Returns:
        --------
        float
            Spaghetti impurity of the noodle.
        """
        return 1.0 - sum((count / total_count) ** 2 for count in recipe_counts.values())

    def _traverse_sauce(self, x, noodle):
        """
        Traverse the sauce to make a dish.
        
        Parameters:
        -----------
        x : np.ndarray
            Single portion for which dish is to be made.
        noodle : Noodle
            The current noodle in the sauce.
        
        Returns:
        --------
        int
            Served recipe label.
        """
        if noodle.is_spaghetti_noodle():
            return noodle.noodle_value
        if x[noodle.noodle_type] <= noodle.noodle_size:
            return self._traverse_sauce(x, noodle.noodle_left)
        return self._traverse_sauce(x, noodle.noodle_right)


# Example usage
if __name__ == "__main__":
    # Spaghetti dataset
    X_pasta = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_pasta = np.array([0, 0, 1, 1])

    # Initialize and cook the recipe
    chef = SpaghettiTreeClassifier(spaghetti_length=3)
    chef.cook(X_pasta, y_pasta)

    # Served dishes
    X_serve = np.array([[1.5, 2.5], [3.5, 4.5]])
    dishes = chef.serve(X_serve)
    print("Served dishes:", dishes)