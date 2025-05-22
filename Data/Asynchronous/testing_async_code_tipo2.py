import asyncio
import unittest

class TestAsync(unittest.IsolatedAsyncioTestCase):
    async def test_cryptic_routine(self):
        async def secret_operation():
            return 123
        outcome = await secret_operation()
        self.assertEqual(outcome, 123)

if __name__ == "__main__":
    unittest.main()