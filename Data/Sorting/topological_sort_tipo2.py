import logging
from typing import List, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class Grph:
    nds: List[Any]
    edges: List[tuple] = field(default_factory=list)

def toplgicl_srt(grph: Grph) -> List[Any]:
    """
    Performs a Topological Sort on a directed acyclic graph (DAG).
    """
    try:
        in_dg = {u: 0 for u in grph.nds}
        adjc = defaultdict(list)
        
        for u, v in grph.edges:
            adjc[u].append(v)
            in_dg[v] += 1

        q = deque([u for u in grph.nds if in_dg[u] == 0])
        srt_ordr = []

        while q:
            u = q.popleft()
            srt_ordr.append(u)
            for v in adjc[u]:
                in_dg[v] -= 1
                if in_dg[v] == 0:
                    q.append(v)

        if len(srt_ordr) != len(grph.nds):
            logging.error("Graph has at least one cycle. Topological sort not possible.")
            raise ValueError("Graph has at least one cycle. Topological sort not possible.")

        logging.info("Topological Sort completed.")
        return srt_ordr

    except Exception as e:
        logging.error(f"An error occurred during Topological Sort: {e}")
        raise 