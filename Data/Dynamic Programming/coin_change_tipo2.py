from dataclasses import dataclass
import logging

@dataclass
class CurrencyExchangeInput:
    denominations: list
    total_amount: int

def exchange_currency(input_data: CurrencyExchangeInput):
    denominations, total_amount = input_data.denominations, input_data.total_amount
    try:
        if not denominations or total_amount < 0:
            return -1

        dp = [float('inf')] * (total_amount + 1)
        dp[0] = 0
        used_denomination = [-1] * (total_amount + 1)

        for denomination in denominations:
            for x in range(denomination, total_amount + 1):
                if dp[x - denomination] + 1 < dp[x]:
                    dp[x] = dp[x - denomination] + 1
                    used_denomination[x] = denomination

        if dp[total_amount] == float('inf'):
            return -1

        result_denominations = []
        while total_amount > 0:
            if used_denomination[total_amount] == -1:
                break
            result_denominations.append(used_denomination[total_amount])
            total_amount -= used_denomination[total_amount]

        return len(result_denominations), result_denominations
    except Exception as e:
        logging.error(f"Error in exchange_currency function: {e}")
        return None

# Test cases
def validate_exchange_currency():
    test_cases = [
        (CurrencyExchangeInput([1, 2, 5], 11), (3, [5, 5, 1])),
        (CurrencyExchangeInput([2], 3), -1),
        (CurrencyExchangeInput([1], 0), (0, [])),
    ]
    
    for input_data, expected in test_cases:
        result = exchange_currency(input_data)
        assert result == expected, f"Expected {expected}, got {result}"

if __name__ == "__main__":
    validate_exchange_currency()