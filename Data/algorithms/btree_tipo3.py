class NodoB:
    def __init__(self, grado, hoja=False):
        self.grado = grado  # Grado mínimo
        self.claves = []  # Lista de claves
        self.hijos = []  # Lista de punteros a hijos
        self.hoja = hoja  # Booleano para indicar si el nodo es una hoja

    def __str__(self):
        """Visualiza la estructura del nodo para facilitar la depuración."""
        return f"Claves: {self.claves}, Hoja: {self.hoja}"

    def recorrer(self):
        """
        Recorre todos los nodos en un subárbol con raíz en este nodo e imprime las claves.
        """
        i = 0
        while i < len(self.claves):
            if not self.hoja:
                self.hijos[i].recorrer()
            print(self.claves[i], end=" ")
            i += 1
        if not self.hoja:
            self.hijos[len(self.claves)].recorrer()

    def buscar(self, k):
        """
        Busca una clave en el subárbol con raíz en este nodo.
        """
        i = 0
        while i < len(self.claves) and k > self.claves[i]:
            i += 1

        if i < len(self.claves) and self.claves[i] == k:
            return self

        if self.hoja:
            return None

        return self.hijos[i].buscar(k)


class ArbolB:
    def __init__(self, grado):
        """
        Inicializa un árbol B con el grado mínimo `grado`.
        """
        self.raiz = None
        self.grado = grado  # Grado mínimo

    def __str__(self):
        """Visualiza toda la estructura del árbol para facilitar la depuración."""
        if self.raiz is not None:
            return str(self.raiz)
        return "Árbol vacío"

    def recorrer(self):
        """
        Recorre todo el árbol.
        """
        if self.raiz is not None:
            self.raiz.recorrer()

    def buscar(self, k):
        """
        Busca una clave `k` en el árbol B.
        """
        if self.raiz is not None:
            return self.raiz.buscar(k)
        return None

    def insertar(self, k):
        """
        Inserta una clave `k` en el árbol B.
        """
        if self.raiz is None:
            self.raiz = NodoB(self.grado, hoja=True)
            self.raiz.claves = [k]
        else:
            if len(self.raiz.claves) == 2 * self.grado - 1:
                s = NodoB(self.grado, hoja=False)
                s.hijos.insert(0, self.raiz)
                self.dividir_hijo(s, 0)
                i = 0
                if s.claves[0] < k:
                    i += 1
                self.insertar_no_lleno(s.hijos[i], k)
                self.raiz = s
            else:
                self.insertar_no_lleno(self.raiz, k)

    def dividir_hijo(self, padre, i):
        """
        Divide al hijo `i` del nodo `padre`.
        """
        grado = self.grado
        y = padre.hijos[i]
        z = NodoB(grado, y.hoja)
        padre.hijos.insert(i + 1, z)
        padre.claves.insert(i, y.claves[grado - 1])
        z.claves = y.claves[grado:(2 * grado - 1)]
        y.claves = y.claves[0:(grado - 1)]

        if not y.hoja:
            z.hijos = y.hijos[grado:(2 * grado)]
            y.hijos = y.hijos[0:grado]

    def insertar_no_lleno(self, nodo, k):
        """
        Inserta una clave `k` en un nodo no lleno.
        """
        i = len(nodo.claves) - 1
        if nodo.hoja:
            nodo.claves.append(None)
            while i >= 0 and k < nodo.claves[i]:
                nodo.claves[i + 1] = nodo.claves[i]
                i -= 1
            nodo.claves[i + 1] = k
        else:
            while i >= 0 and k < nodo.claves[i]:
                i -= 1
            i += 1
            if len(nodo.hijos[i].claves) == 2 * self.grado - 1:
                self.dividir_hijo(nodo, i)
                if k > nodo.claves[i]:
                    i += 1
            self.insertar_no_lleno(nodo.hijos[i], k)

# Ejemplo de uso:
if __name__ == "__main__":
    arbol_b = ArbolB(3)  # Crea un árbol B con grado mínimo 3
    arbol_b.insertar(10)
    arbol_b.insertar(20)
    arbol_b.insertar(5)
    arbol_b.insertar(6)
    arbol_b.insertar(12)
    arbol_b.insertar(30)
    arbol_b.insertar(7)
    arbol_b.insertar(17)

    print("Recorrido del Árbol B:")
    arbol_b.recorrer()  # Imprime las claves del árbol en orden ordenado
    print("\nBuscando la clave 6 en el árbol:")
    resultado = arbol_b.buscar(6)
    print(f"Clave {6} encontrada: {resultado is not None}")