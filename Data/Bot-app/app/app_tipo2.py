import spacy
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import re

# Load SpaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    print("SpaCy model 'en_core_web_sm' not found. Downloading now...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class OrderState(Enum):
    PRIME = "prime"
    PIZZA_PICK = "pizza_pick"
    SIZE_PICK = "size_pick"
    TOPPINGS_PICK = "toppings_pick"
    CRUST_PICK = "crust_pick"
    CONFIRMATION = "confirmation"
    COMPLETED = "completed"

@dataclass
class DishItem:
    moniker: str
    value: float
    explanation: str
    adjustments: List[str]
    
@dataclass
class Request:
    pie_species: Optional[str] = None
    magnitude: Optional[str] = None
    toppings: List[str] = None
    hull: Optional[str] = None
    net_value: float = 0.0
    
    def __post_init__(self):
        if self.toppings is None:
            self.toppings = []

class PizzaBot:
    def __init__(self):
        # Load menu data
        self.menu = self._fetch_menu()
        self.order = Request()
        self.state = OrderState.PRIME
        
        # Intent patterns
        self.intent_patterns = {
            'greeting': [r'hello', r'hi', r'hey', r'start'],
            'make_order': [r'order', r'place.*order', r'want.*pizza'],
            'validate': [r'validate', r'yes', r'correct'],
            'terminate': [r'terminate', r'no', r'stop'],
        }
        
    def _fetch_menu(self) -> Dict[str, DishItem]:
        # Sample menu data
        menu_data = {
            "Primo": DishItem(
                moniker="Primo",
                value=10.99,
                explanation="Classic tomato and mozzarella",
                adjustments=["primo", "plain cheese"]
            ),
            "Peppy": DishItem(
                moniker="Peppy",
                value=12.99,
                explanation="Pepperoni and cheese",
                adjustments=["pepperoni"]
            ),
        }
        return menu_data
    
    def _detect_intent(self, text: str) -> str:
        text = text.lower()
        for intent, patterns in self.intent_patterns.items():
            if any(re.search(pattern, text) for pattern in patterns):
                return intent
        return "unknown"
    
    def _extract_pie_species(self, text: str) -> Optional[str]:
        doc = nlp(text.lower())
        for pizza_name, item in self.menu.items():
            if pizza_name.lower() in text.lower() or \
               any(variation in text.lower() for variation in item.adjustments):
                return pizza_name
        return None
    
    def _extract_magnitude(self, text: str) -> Optional[str]:
        sizes = {"small": 1, "medium": 1.5, "large": 2}
        doc = nlp(text.lower())
        for size in sizes:
            if size in text.lower():
                return size
        return None
    
    def _extract_toppings(self, text: str) -> List[str]:
        available_toppings = [
            "mushrooms", "onions", "peppers", "olives", 
            "extra cheese", "bacon", "chicken"
        ]
        doc = nlp(text.lower())
        found_toppings = []
        for topping in available_toppings:
            if topping in text.lower():
                found_toppings.append(topping)
        return found_toppings
    
    def _extract_hull(self, text: str) -> Optional[str]:
        crust_types = ["thin", "thick", "stuffed", "regular"]
        doc = nlp(text.lower())
        for crust in crust_types:
            if crust in text.lower():
                return crust
        return None
    
    def _calculate_value(self) -> float:
        base_value = self.menu[self.order.pie_species].value if self.order.pie_species else 0
        size_multiplier = {"small": 1, "medium": 1.5, "large": 2}
        topping_value = len(self.order.toppings) * 1.5
        crust_extra = {"thin": 0, "thick": 2, "stuffed": 3, "regular": 0}
        
        total = base_value
        if self.order.magnitude:
            total *= size_multiplier[self.order.magnitude]
        total += topping_value
        if self.order.hull:
            total += crust_extra[self.order.hull]
        
        return round(total, 2)
    
    def process_message(self, message: str) -> str:
        intent = self._detect_intent(message)
        
        if intent == "terminate":
            self.state = OrderState.PRIME
            self.order = Request()
            return "Order terminated. How can I assist you?"
            
        if self.state == OrderState.PRIME:
            if intent == "greeting":
                return "Hello! Welcome to PizzaBot. Would you like to place an order?"
            elif intent == "make_order":
                self.state = OrderState.PIZZA_PICK
                return "What kind of pizza would you like? We offer Primo and Peppy."
                
        elif self.state == OrderState.PIZZA_PICK:
            pie_species = self._extract_pie_species(message)
            if pie_species:
                self.order.pie_species = pie_species
                self.state = OrderState.SIZE_PICK
                return "What size would you like? (Small/Medium/Large)"
            else:
                return "I didn't catch that. Please select from Primo or Peppy."
                
        elif self.state == OrderState.SIZE_PICK:
            magnitude = self._extract_magnitude(message)
            if magnitude:
                self.order.magnitude = magnitude
                self.state = OrderState.TOPPINGS_PICK
                return "Would you like any extra toppings? Available options: mushrooms, onions, peppers, olives, extra cheese, bacon, chicken"
            else:
                return "Please specify a valid size (Small/Medium/Large)."
                
        elif self.state == OrderState.TOPPINGS_PICK:
            if "no" in message.lower():
                self.state = OrderState.CRUST_PICK
                return "What type of crust would you like? (Thin/Thick/Stuffed/Regular)"
            
            toppings = self._extract_toppings(message)
            if toppings:
                self.order.toppings.extend(toppings)
                self.state = OrderState.CRUST_PICK
                return "What type of crust would you like? (Thin/Thick/Stuffed/Regular)"
            else:
                return "I didn't catch any toppings. Please specify from the available options or say 'no' for no toppings."
                
        elif self.state == OrderState.CRUST_PICK:
            hull = self._extract_hull(message)
            if hull:
                self.order.hull = hull
                self.order.net_value = self._calculate_value()
                self.state = OrderState.CONFIRMATION
                return self._generate_order_summary()
            else:
                return "Please specify a valid crust type (Thin/Thick/Stuffed/Regular)."
                
        elif self.state == OrderState.CONFIRMATION:
            if intent == "validate":
                self.state = OrderState.COMPLETED
                return "Thank you for your order! Your pizza will be ready in 30 minutes."
            elif intent == "terminate":
                self.state = OrderState.PRIME
                self.order = Request()
                return "Order terminated. Would you like to start a new order?"
                
        return "I'm not sure what you mean. Could you please rephrase that?"
    
    def _generate_order_summary(self) -> str:
        summary = "\nOrder Summary:\n"
        summary += f"Pizza: {self.order.pie_species}\n"
        summary += f"Size: {self.order.magnitude}\n"
        summary += f"Toppings: {', '.join(self.order.toppings) if self.order.toppings else 'No extra toppings'}\n"
        summary += f"Crust: {self.order.hull}\n"
        summary += f"Total Value: ${self.order.net_value}\n"
        summary += "\nWould you like to validate this order?"
        return summary

if __name__ == "__main__":
    chatbot = PizzaBot()
    print("Welcome to PizzaBot! Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = chatbot.process_message(user_input)
        print(f"Bot: {response}")