import unittest
from app import DominosChatbot, OrderState, Order

class TestDominosChatbot(unittest.TestCase):
    def setUp(self):
        self.bot = DominosChatbot()
        
    def test_intent_detection(self):
        self.assertEqual(self.bot._detect_intent("hello there"), "greeting")
        self.assertEqual(self.bot._detect_intent("hi"), "greeting")
        
        self.assertEqual(self.bot._detect_intent("I want to order a pizza"), "order")
        self.assertEqual(self.bot._detect_intent("place order"), "order")
        
        self.assertEqual(self.bot._detect_intent("yes please"), "confirm")
        self.assertEqual(self.bot._detect_intent("confirm"), "confirm")
        
        self.assertEqual(self.bot._detect_intent("no thanks"), "cancel")
        self.assertEqual(self.bot._detect_intent("cancel"), "cancel")
        
        self.assertEqual(self.bot._detect_intent("random text"), "unknown")
    
    def test_pizza_type_extraction(self):
        self.assertEqual(self.bot._extract_pizza_type("I want a Margherita"), "Margherita")
        self.assertEqual(self.bot._extract_pizza_type("Pepperoni please"), "Pepperoni")
        
        self.assertEqual(self.bot._extract_pizza_type("plain cheese pizza"), "Margherita")
        
        self.assertIsNone(self.bot._extract_pizza_type("I want something else"))
    
    def test_size_extraction(self):
        self.assertEqual(self.bot._extract_size("I want a small pizza"), "small")
        self.assertEqual(self.bot._extract_size("medium size please"), "medium")
        self.assertEqual(self.bot._extract_size("large"), "large")
        
        self.assertIsNone(self.bot._extract_size("I want a pizza"))
    
    def test_toppings_extraction(self):
        self.assertEqual(
            self.bot._extract_toppings("add mushrooms"),
            ["mushrooms"]
        )
        
        self.assertEqual(
            set(self.bot._extract_toppings("mushrooms and onions with extra cheese")),
            {"mushrooms", "onions", "extra cheese"}
        )
        
        self.assertEqual(self.bot._extract_toppings("no toppings"), [])
    
    def test_crust_extraction(self):
        self.assertEqual(self.bot._extract_crust("thin crust"), "thin")
        self.assertEqual(self.bot._extract_crust("thick crust please"), "thick")
        self.assertEqual(self.bot._extract_crust("stuffed"), "stuffed")
        self.assertEqual(self.bot._extract_crust("regular crust"), "regular")
        
        self.assertIsNone(self.bot._extract_crust("no preference"))
    
    def test_price_calculation(self):
        self.bot.order = Order(
            pizza_type="Margherita",
            size="small",
            toppings=[],
            crust="regular"
        )
        self.assertEqual(self.bot._calculate_price(), 10.99)
        
        self.bot.order.size = "large"
        self.assertEqual(self.bot._calculate_price(), 21.98)
        
        self.bot.order.toppings = ["mushrooms", "onions"]
        self.assertEqual(self.bot._calculate_price(), 24.98)
        
        self.bot.order.crust = "stuffed"
        self.assertEqual(self.bot._calculate_price(), 27.98)
    
    def test_order_flow(self):
        response = self.bot.process_message("hello")
        self.assertEqual(self.bot.state, OrderState.INITIAL)
        self.assertIn("Welcome to Domino's", response)
        
        response = self.bot.process_message("I want to order")
        self.assertEqual(self.bot.state, OrderState.PIZZA_SELECTION)
        self.assertIn("What type of pizza", response)
        
        response = self.bot.process_message("Margherita")
        self.assertEqual(self.bot.state, OrderState.SIZE_SELECTION)
        self.assertIn("What size", response)
        
        response = self.bot.process_message("large")
        self.assertEqual(self.bot.state, OrderState.TOPPINGS_SELECTION)
        self.assertIn("extra toppings", response)
        
        response = self.bot.process_message("mushrooms and onions")
        self.assertEqual(self.bot.state, OrderState.CRUST_SELECTION)
        self.assertIn("type of crust", response)
        
        response = self.bot.process_message("thin")
        self.assertEqual(self.bot.state, OrderState.CONFIRMATION)
        self.assertIn("Order Summary", response)
        
        response = self.bot.process_message("confirm")
        self.assertEqual(self.bot.state, OrderState.COMPLETED)
        self.assertIn("Thank you for your order", response)

if __name__ == '__main__':
    unittest.main()