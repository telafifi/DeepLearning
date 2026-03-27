import json
from pathlib import Path

from .base_llm import BaseLLM
from .data import DATA_DIR, Dataset, benchmark
from .sft import TokenizedDataset, test_model, tokenize


class RFTDataset:
    """Dataset that loads RFT-generated data (question, answer, reasoning)."""

    def __init__(self, json_path: str):
        with open(json_path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def rft_format_example(prompt: str, answer: float, reasoning: str) -> dict[str, str]:
    """Format an RFT example: question + chain-of-thought reasoning as the answer."""
    return {
        "question": prompt,
        "answer": reasoning,
    }


def load() -> BaseLLM:
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str = "homework/rft_model",
    **kwargs,
):
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    llm = BaseLLM()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()
    llm.model.print_trainable_parameters()

    rft_json = DATA_DIR / "rft.json"
    rft_data = RFTDataset(str(rft_json))
    train_dataset = TokenizedDataset(llm.tokenizer, rft_data, rft_format_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        learning_rate=5e-4,
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        fp16=False,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    # Save the final model
    output_path = Path(__file__).parent / "rft_model"
    trainer.save_model(str(output_path))

    test_model(str(output_path))


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
