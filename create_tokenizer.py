from pathlib import Path

from tokenizers.trainers import WordPieceTrainer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


def tokenizer_from_vocab(vocab):
    # TODO: replace these files with just one containing all lowercase and 
    # uppercase characters, or our training corpus
    paths = [str(x) for x in Path("./tokenizer_training_data/").glob("**/*.txt")]
    special_tokens = [
    "[UNK]", 
    "[CLS]", 
    "[SEP]", 
    "[PAD]", 
    "[MASK]"
    ]

    # wordpiece is used by bert, with the following settings
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    # vocab_size is 0, so this really does no training and only includes the 
    # characters from our training txt files
    trainer = WordPieceTrainer(vocab_size=0, special_tokens=special_tokens)
    tokenizer.train(paths, trainer)

    # this is where we add our tokens
    tokenizer.add_tokens(vocab)
    return tokenizer

if __name__ == "__main__":
    vocab = [
        "a",
        "the",
        "other",
        "sahil", 
        "jain", 
        "is", 
        "interest", 
        "ing"
    ]

    # Save a basic tokenizer to disk
    tokenizer = tokenizer_from_vocab(vocab)
    tokenizer.save("tokenizers/saved_standard.json")
