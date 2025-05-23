from dataclasses import dataclass

@dataclass
class CryptoSettings:
    span: int = 12
    include_top: bool = True
    include_bottom: bool = True
    include_numbers: bool = True
    include_specials: bool = True
    min_span: int = 8
    max_span: int = 64
    demand_top: bool = True
    demand_bottom: bool = True
    demand_numbers: bool = True
    demand_specials: bool = True