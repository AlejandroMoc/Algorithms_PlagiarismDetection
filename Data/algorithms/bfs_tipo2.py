from collections import deque
from typing import Dict, List

def trv(grph: Dict[str, List[str]], strt: str) -> List[str]:
    vsitd = set()
    que = deque([strt])
    vsitd.add(strt)
    ordr_vst = []  # To store the order of visited nodes

    while que:
        vtx = que.popleft()
        ordr_vst.append(vtx)

        for nbr in grph[vtx]:
            if nbr not in vsitd:
                vsitd.add(nbr)
                que.append(nbr)

    return ordr_vst

# Example usage:
if __name__ == "__main__":
    grph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    order = trv(grph, 'A')
    print("Order of visit:", order)