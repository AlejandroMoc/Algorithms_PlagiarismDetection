import spacy
from typing import List, Tuple, Optional

    return [(ent.text, ent.label_) for ent in doc.ents] 

def extract_entities(text: str) -> Optional[List[Tuple[str, str]]]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
