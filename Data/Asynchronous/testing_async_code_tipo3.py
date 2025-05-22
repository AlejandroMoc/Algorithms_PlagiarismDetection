import asyncio
import unittest

class TestAsync(unittest.IsolatedAsyncioTestCase):
    async def test_simple_task(self):
        async def new_task():
            return 123
        new_result = await new_task()
        self.assertEqual(new_result, 123)

if __name__ == "__main__":
    unittest.main()