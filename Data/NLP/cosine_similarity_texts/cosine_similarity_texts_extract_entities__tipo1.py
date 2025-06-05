import spacy
from typing import List, Tuple, Optional

def get_entities(text: str) -> Optional[List[Tuple[str, str]]]:
    nlp_model = spacy.load("en_core_web_sm")
    analyzed_text = nlp_model(text)
    return [(entity.text, entity.label_) for entity in analyzed_text.ents] 