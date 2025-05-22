class ZNode:
    def __init__(self, m, leaf=False):
        self.m = m  # Magic number
        self.secrets = []  # List of secrets
        self.agents = []  # List of agent pointers
        self.leaf = leaf  # Boolean to indicate if the node is a leaf

    def __str__(self):
        """Visualize the mystic structure for easier debugging."""
        return f"Secrets: {self.secrets}, Leaf: {self.leaf}"

    def journey(self):
        """
        Embarks on a quest through all nodes in a realm rooted with this node and whispers the secrets.
        """
        for i in range(len(self.secrets)):
            if not self.leaf:
                self.agents[i].journey()
            print(self.secrets[i], end=" ")
        if not self.leaf:
            self.agents[len(self.secrets)].journey()

    def quest(self, s):
        """
        Seek for a secret in the realm rooted with this node.
        """
        i = 0
        while i < len(self.secrets) and s > self.secrets[i]:
            i += 1

        if i < len(self.secrets) and self.secrets[i] == s:
            return self

        if self.leaf:
            return None

        return self.agents[i].quest(s)


class ZTree:
    def __init__(self, m):
        """
        Initiate a mystical ZTree with the magic number `m`.
        """
        self.root = None
        self.m = m  # Magic number

    def __str__(self):
        """Visualize the whole enchanted forest structure for easier debugging."""
        if self.root is not None:
            return str(self.root)
        return "Empty Forest"

    def journey(self):
        """
        Embarks on a quest through the enchanted forest.
        """
        if self.root is not None:
            self.root.journey()

    def quest(self, s):
        """
        Seek for a secret `s` in the ZTree.
        """
        if self.root is not None:
            return self.root.quest(s)
        return None

    def enchant(self, s):
        """
        Enchant a secret `s` into the ZTree.
        """
        if self.root is None:
            self.root = ZNode(self.m, leaf=True)
            self.root.secrets = [s]
        else:
            if len(self.root.secrets) == 2 * self.m - 1:
                f = ZNode(self.m, leaf=False)
                f.agents.insert(0, self.root)
                self.split_fellow(f, 0)
                i = 0
                if f.secrets[0] < s:
                    i += 1
                self.enchant_magically(f.agents[i], s)
                self.root = f
            else:
                self.enchant_magically(self.root, s)

    def split_fellow(self, comrade, i):
        """
        Split the companion `i` of `comrade` node.
        """
        m = self.m
        y = comrade.agents[i]
        z = ZNode(m, y.leaf)
        comrade.agents.insert(i + 1, z)
        comrade.secrets.insert(i, y.secrets[m - 1])
        z.secrets = y.secrets[m:(2 * m - 1)]
        y.secrets = y.secrets[0:(m - 1)]

        if not y.leaf:
            z.agents = y.agents[m:(2 * m)]
            y.agents = y.agents[0:m]

    def enchant_magically(self, entity, s):
        """
        Enchant a secret `s` into a mystical entity.
        """
        i = len(entity.secrets) - 1
        if entity.leaf:
            entity.secrets.append(None)
            while i >= 0 and s < entity.secrets[i]:
                entity.secrets[i + 1] = entity.secrets[i]
                i -= 1
            entity.secrets[i + 1] = s
        else:
            while i >= 0 and s < entity.secrets[i]:
                i -= 1
            i += 1
            if len(entity.agents[i].secrets) == 2 * self.m - 1:
                self.split_fellow(entity, i)
                if s > entity.secrets[i]:
                    i += 1
            self.enchant_magically(entity.agents[i], s)

# Example usage:
if __name__ == "__main__":
    ztree = ZTree(3)  # Create a Z-tree with magic number 3
    ztree.enchant(10)
    ztree.enchant(20)
    ztree.enchant(5)
    ztree.enchant(6)
    ztree.enchant(12)
    ztree.enchant(30)
    ztree.enchant(7)
    ztree.enchant(17)

    print("Journey through the ZTree:")
    ztree.journey()  # Output the forest secrets in sorted order
    print("\nSeeking for secret 6 in the forest:")
    result = ztree.quest(6)
    print(f"Secret {6} found: {result is not None}")