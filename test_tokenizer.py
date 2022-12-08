import sys
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizers/saved_standard.json")

input_ex = "this is an interesting base test tokenization written by sahil!"
if len(sys.argv) > 1:
    input_ex = sys.argv[1]

print(
    tokenizer.encode(input_ex).tokens
)
