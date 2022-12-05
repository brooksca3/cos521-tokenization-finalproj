from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("./training_data/").glob("**/*.txt")]

vocab = [
    "a",
    "the",
    "sahil",
    "other",
    "interest",
    "ing"
]

special = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
]

all_tokens = vocab + special
size = len(all_tokens)

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize 'training'
tokenizer.train(files=paths, vocab_size=size, min_frequency=0, special_tokens=all_tokens)

# Save files to disk
tokenizer.save_model(".", "experiment_tokenizer")
