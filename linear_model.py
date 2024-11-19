import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer

import hf_token


class TextDataset(IterableDataset):
    def __init__(self, tokenizer, split="train", max_length=1024):
        # Load OpenWebText dataset
        self.tokenizer = tokenizer
        self.dataset = load_dataset("openwebtext", split=split, streaming=True)
        self.max_length = max_length

    def __iter__(self):
        for item in self.dataset:
            # Tokenize the text
            encodings = self.tokenizer(
                item["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # Create input and target tensors
            input_ids = encodings["input_ids"].squeeze(0)
            attention_mask = encodings["attention_mask"].squeeze(0)

            # Target is the next token
            target = input_ids[1:].clone()
            input_ids = input_ids[:-1]
            attention_mask = attention_mask[:-1]

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "target": target,
            }


class LinearLanguageModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=512, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask):
        # input_ids: [batch_size, seq_len]
        x = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        x = self.linear(x)  # [batch_size, seq_len, vocab_size]
        return x

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target = batch["target"]

        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target = batch["target"]

        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B", token=hf_token.token
    )
    print("Tokenizer vocabulary size:", tokenizer.vocab_size)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    train_dataset = TextDataset(tokenizer, split="train")

    # Create dataloaders
    batch_size = 1024
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    # Initialize model
    embedding_dim = 4096
    model = LinearLanguageModel(
        vocab_size=tokenizer.vocab_size + 2,
        embedding_dim=embedding_dim,
        learning_rate=3e-4,
    )

    # Setup training
    logger = TensorBoardLogger("lightning_logs", name="linear_language_model")
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="checkpoints",
        filename="linear-language-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=None,
    )

    # Train model
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
