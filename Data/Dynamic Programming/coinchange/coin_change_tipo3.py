python
from dataclasses import dataclass
import logging

@dataclass
class CoinChangeInput:
    coins_list: list
    total_amount: int

def calculate_coin_change(input_data: CoinChangeInput):
    coins, amount = input_data.coins_list, input_data.total_amount
    try:
        if not coins or amount < 0:
            return -1

        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        coin_used = [-1] * (amount + 1)

        for coin_value in coins:
            for current_amount in range(coin_value, amount + 1):
                if dp[current_amount - coin_value] + 1 < dp[current_amount]:
                    dp[current_amount] = dp[current_amount - coin_value] + 1
                    coin_used[current_amount] = coin_value

        if dp[amount] == float('inf'):
            return -1

        result_coins = []
        while amount > 0:
            if coin_used[amount] == -1:
                break
            result_coins.append(coin_used[amount])
            amount -= coin_used[amount]

        return len(result_coins), result_coins
    except Exception as e:
        logging.error(f"Error in calculate_coin_change function: {e}")
        return None

# Test cases
def test_calculate_coin_change():
    test_cases = [
        (CoinChangeInput([1, 2, 5], 11), (3, [5, 5, 1])),
        (CoinChangeInput([2], 3), -1),
        (CoinChangeInput([1], 0), (0, [])),
    ]
    
    for input_data, expected_result in test_cases:
        result = calculate_coin_change(input_data)
        assert result == expected_result, f"Expected {expected_result}, got {result}"

if __name__ == "__main__":
    test_calculate_coin_change()