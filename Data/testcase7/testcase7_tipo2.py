from collections import deque

class ArbreNode:
    def __init__(self, valeur=0, gauche=None, droit=None):
        self.valeur = valeur
        self.gauche = gauche
        self.droit = droit

def serialiser(root):
    donnees, file = [], deque([root])
    while file:
        noeud = file.popleft()
        if noeud:
            donnees.append(str(noeud.valeur))
            file.append(noeud.gauche)
            file.append(noeud.droit)
        else:
            donnees.append('#')
    return ' '.join(donnees)

def deserialiser(donnees):
    if donnees == '#':
        return None
    noeuds = donnees.split()
    racine = ArbreNode(int(noeuds[0]))
    file = deque([racine])
    index = 1
    while file:
        noeud = file.popleft()
        if noeuds[index] != '#':
            noeud.gauche = ArbreNode(int(noeuds[index]))
            file.append(noeud.gauche)
        index += 1
        if noeuds[index] != '#':
            noeud.droit = ArbreNode(int(noeuds[index]))
            file.append(noeud.droit)
        index += 1
    return racine