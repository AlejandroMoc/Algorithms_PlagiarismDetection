import random

def generate_substring(input_text: str, num_words: int = 3) -> str:
    words_list = input_text.split()
    start_index = random.randint(0, len(words_list) - num_words)
    return ' '.join(words_list[start_index:start_index+num_words])