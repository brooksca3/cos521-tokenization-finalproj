from pathlib import Path
from create_tokenizer import tokenizer_from_vocab
from tokenizers import Tokenizer
from torch.utils.data import Dataset


class MarathiDataset(Dataset):
    def __init__(self, tokenizer, evaluate: bool = False):

        self.examples = []

        src_files = Path("./training_data/").glob("*-eval.txt") if evaluate else Path("./training_data/").glob("*-train.txt")
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import PreTrainedTokenizerFast


slow_tokenizer = Tokenizer.from_file("tokenizers/saved_standard.json")
slow_tokenizer.mask_token = "[MASK]"
slow_tokenizer.enable_truncation(max_length=512)
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizers/saved_standard.json") 
tokenizer.mask_token = "[MASK]"


config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=514,
    num_attention_heads=2, # default was 12
    num_hidden_layers=1, # default was 6
    type_vocab_size=1,
)
model = BertForMaskedLM(config=config)
print(model.num_parameters()) # -> approx 8M


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
dataset = MarathiDataset(tokenizer=slow_tokenizer)

training_args = TrainingArguments(
    output_dir="./models/marathiBERT",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
