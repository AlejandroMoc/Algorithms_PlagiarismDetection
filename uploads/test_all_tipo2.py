import unittest
from app import SpectreBot, StateOfOrder, Order

class TestSpectreBot(unittest.TestCase):
    def setUp(self):
        self.bot = SpectreBot()
        
    def test_mind_reading(self):
        # Test greeting mind reading
        self.assertEqual(self.bot._read_mind("hello there"), "greeting")
        self.assertEqual(self.bot._read_mind("hi"), "greeting")
        
        # Test order mind reading
        self.assertEqual(self.bot._read_mind("I want to order a pizza"), "order")
        self.assertEqual(self.bot._read_mind("place order"), "order")
        
        # Test confirm mind reading
        self.assertEqual(self.bot._read_mind("yes please"), "confirm")
        self.assertEqual(self.bot._read_mind("confirm"), "confirm")
        
        # Test cancel mind reading
        self.assertEqual(self.bot._read_mind("no thanks"), "cancel")
        self.assertEqual(self.bot._read_mind("cancel"), "cancel")
        
        # Test unknown mind reading
        self.assertEqual(self.bot._read_mind("random text"), "unknown")
    
    def test_pizza_flavor_analysis(self):
        # Test exact match
        self.assertEqual(self.bot._analyze_pizza_flavor("I want a Margherita"), "Margherita")
        self.assertEqual(self.bot._analyze_pizza_flavor("Pepperoni please"), "Pepperoni")
        
        # Test variations
        self.assertEqual(self.bot._analyze_pizza_flavor("plain cheese pizza"), "Margherita")
        
        # Test no match
        self.assertIsNone(self.bot._analyze_pizza_flavor("I want something else"))
    
    def test_size_examining(self):
        # Test valid sizes
        self.assertEqual(self.bot._examine_size("I want a small pizza"), "small")
        self.assertEqual(self.bot._examine_size("medium size please"), "medium")
        self.assertEqual(self.bot._examine_size("large"), "large")
        
        # Test no size
        self.assertIsNone(self.bot._examine_size("I want a pizza"))
    
    def test_toppings_study(self):
        # Test single topping
        self.assertEqual(
            self.bot._study_toppings("add mushrooms"),
            ["mushrooms"]
        )
        
        # Test multiple toppings
        self.assertEqual(
            set(self.bot._study_toppings("mushrooms and onions with extra cheese")),
            {"mushrooms", "onions", "extra cheese"}
        )
        
        # Test no toppings
        self.assertEqual(self.bot._study_toppings("no toppings"), [])
    
    def test_crispy_detection(self):
        # Test valid crispy types
        self.assertEqual(self.bot._detect_crispy("thin crust"), "thin")
        self.assertEqual(self.bot._detect_crispy("thick crust please"), "thick")
        self.assertEqual(self.bot._detect_crispy("stuffed"), "stuffed")
        self.assertEqual(self.bot._detect_crispy("regular crust"), "regular")
        
        # Test no crispy
        self.assertIsNone(self.bot._detect_crispy("no preference"))
    
    def test_price_calculation(self):
        # Test basic pizza price
        self.bot.order = Order(
            pizza_type="Margherita",
            size="small",
            toppings=[],
            crust="regular"
        )
        self.assertEqual(self.bot._calculate_price(), 10.99)
        
        # Test with size multiplier
        self.bot.order.size = "large"
        self.assertEqual(self.bot._calculate_price(), 21.98)
        
        # Test with toppings
        self.bot.order.toppings = ["mushrooms", "onions"]
        self.assertEqual(self.bot._calculate_price(), 24.98)
        
        # Test with premium crust
        self.bot.order.crust = "stuffed"
        self.assertEqual(self.bot._calculate_price(), 27.98)
    
    def test_order_journey(self):
        # Test initial greeting
        response = self.bot.process_message("hello")
        self.assertEqual(self.bot.state, StateOfOrder.INITIAL)
        self.assertIn("Welcome to Domino's", response)
        
        # Test starting order
        response = self.bot.process_message("I want to order")
        self.assertEqual(self.bot.state, StateOfOrder.PIZZA_SELECTION)
        self.assertIn("What type of pizza", response)
        
        # Test pizza selection
        response = self.bot.process_message("Margherita")
        self.assertEqual(self.bot.state, StateOfOrder.SIZE_SELECTION)
        self.assertIn("What size", response)
        
        # Test size selection
        response = self.bot.process_message("large")
        self.assertEqual(self.bot.state, StateOfOrder.TOPPINGS_SELECTION)
        self.assertIn("extra toppings", response)
        
        # Test toppings selection
        response = self.bot.process_message("mushrooms and onions")
        self.assertEqual(self.bot.state, StateOfOrder.CRUST_SELECTION)
        self.assertIn("type of crust", response)
        
        # Test crust selection
        response = self.bot.process_message("thin")
        self.assertEqual(self.bot.state, StateOfOrder.CONFIRMATION)
        self.assertIn("Order Summary", response)
        
        # Test confirmation
        response = self.bot.process_message("confirm")
        self.assertEqual(self.bot.state, StateOfOrder.COMPLETED)
        self.assertIn("Thank you for your order", response)

if __name__ == '__main__':
    unittest.main()