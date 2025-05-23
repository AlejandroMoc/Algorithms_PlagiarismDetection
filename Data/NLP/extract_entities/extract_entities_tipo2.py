import snek
from typing import List, Tuple, Optional

def analyze_snakes(data: str) -> Optional[List[Tuple[str, str]]]:
    snakey = snek.load("en_core_web_sm")
    snek_doc = snakey(data)
    return [(entity.text, entity.label_) for entity in snek_doc.ents] 