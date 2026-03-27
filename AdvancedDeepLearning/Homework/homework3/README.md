# Homework 3 - Well reasoned unit conversion

In this homework, we will train language models to perform unit conversions (meters to yard to feet etc).
We will use SmolLM2 and the huggingface library (with some more dependencies than usual).

The homework consists of four parts:

1. Implement generation and batched generation in `base_llm.py`
2. Using in-context learning and chain of thought to perform basic unit conversions in `cot.py`
3. Fine-tune SmolLM2 (using LoRA) to learn to convert units better in `sft.py`
4. Implement a very basic RL algorithm RFT (Yuan etal. 2023, https://arxiv.org/abs/2308.01825) to fine-tune the model in `rft.py` and `dataset.py`

Familiarize yourself with the starter code. All data ships with the starter code.

We provide dataloaders for the text data in `data.py`.

## Grading Criteria

Each part is worth 25 pts with 5 pts of extra credit for an especially performant RFT model.

## Generation with SmolLM2 (25 pts)

We start by implementing the generation function of SmolLM2.
We will generate from scratch rather than using huggingface pipelines.

To warm up, implement a sequential version of `generate` in `base_llm.py`.
If you feel confident, skip this and move straight to `batched_generate`.
We already took care of loading the model and tokenizer.
You can find some simple examples on how to use SmolLM2 here: <https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B>

Test your code with

```bash
python -m homework.base_llm test
```

Next, we implement a batched version `batched_generate`.
Batching will make much better use of your GPU and likely work 10-20x faster than an unbatched version.
The core structure of batched generation is very similar to regular generation, with one exception: All sequences that go into the transformer need to be of the same length.
This is achieved through padding the shorter sequences in the left (aligning all sequences on the right, where generation starts).
The transformers library will take care of padding in the `self.tokenizer` call, simply pass in a `list[str]` of prompts and use `padding=True` and return a PyTorch tensor `return_tensors="pt"`.
Generation if almost the same between unbatched and batched versions with the only difference being that `self.model.generate` take both `input_ids` (the tokenized input) and `attention_mask` as input.
`attention_mask` is produced by the tokenizer indicating which inputs have been padded.
Finally, the `self.tokenizer` should decode the generated output using `batch_decode`.
This will produce a flat `list[str]` of generations of length `num_return_sequences * len(prompts)`.
Reshape this list if required (`num_return_sequences is not None`).

## In context learning (25 pts)

Implement the `format_prompt` function in `cot.py`.
Given a `question: str` you should create a chat dialogue that prompts the LLM to produce the correct answer.
A chat dialogue has the following structure

```python
messages: list[dict[str, str]] = [
    {"role": role, "content": content},
    ...
]
```

where `role` is a string literal (`"system"`, `"user"`, or `"assistant"`), and `content` is a free-form string.
You can use the chat dialogue to both instruct the model to perform a task in the system or user message, and provide in-context examples in a prior assistant message.
The LLM will do best if you give it:

- brief instructions
- tell it to `be concise`
- Give one good example how to solve the task

Use the `self.tokenizer.apply_chat_template` with `add_generation_prompt=True` and `tokenize=False` to convert the chat messages into a single string following the chat-template SmolLM2 expects (including all special tokens, and the beginning of the assistant output).
Feel free to print this output to familiarize yourself with how this works.

Test your model with

```bash
python -m homework.cot test
```

You should be able to reach 0.5 accuracy and 0.85 answer_rate without too much tuning, and a good in-context example.

## Supervised fine-tuning (25 pts)

We will now go and fine-tune SmolLM2 to answer questions directly.
You should NOT use the chat template here, instead simply ask the model to complete a question with `<answer>{answer}</answer>`, where answer is the ground truth `float` answer.

Due to file-size limitations you will not be able to submit a fully fine-tuned model, but will need to submit a LoRA adapter.
Use the `get_peft_model` function to convert the `BaseLLM.model` into a LoRA adapted version.
The function above takes a `LoraConfig` argument, most parameters are quite flexible.
Our recommendation is to use:

- `target_modules="all-linear"` this will add an adapter to all layers
- `bias="none"` and `task_type="CAUSAL_LM"`
- `r` rank such that the overall model size stays below 20MB
- `lora_alpha` about 4-5 times the rank

If you're using a GPU call `model.enable_input_require_grads()` after adding the LoRA adapter to avoid a bug with `gradient_checkpointing=True,` in the `TrainingArguments` below.

We will use the higgingface `Trainer` to fine-tune the model.
The trainer takes 3 arguments:

- Our LoRA model
- `TrainingArguments`
  - Use `gradient_checkpointing=True` to save GPU memory
  - Set a reasonable `learning_rate`
  - Use `output_dir=output_dir`, `logging_dir=output_dir`, `report_to="tensorboard"` to create a
    tensorboard log and checkpoints in `output_dir`
  - You shouldn't have to train for more than 5 `num_train_epochs` with a `per_device_train_batch_size=32`
- A `TokenizedDataset`. We provide significant part of the tokenization starter code here.

Finally, call `Trainer.train` to train the model.

Either write a script that moves the final checkpoint in the correct directory or call `Trainer.save` to write the model to the `homework/sft_model` directory.

Train your model with

```bash
python -m homework.sft train
```

and make sure it can be loaded by the grader

```bash
python -m homework.sft train
```

## Rejection sampling Fine-Tuning (25 pts)

Finally, we implement a very basic RL algorithm to improve the reasoning capabilities of our LLM.
The above SFT experiment produced straight up `<answer>...</answer>` outputs without first thinking about how to convert units.
RFT will combine strengths of both Chain-of-Thought reasoning and SFT.

RFT (Yuan etal. 2023, https://arxiv.org/abs/2308.01825) uses an offline procedure to create chain-of-thought-based answers.
They start with a pre-trained LLM and in-context learning to create a new dataset of correct question / reasoning / answer tuples.
We will implement this in `datagen.py`.
Specifically, implement `generate_dataset` to produce 10 - 20 different completions from your `CoTModel`, then select the one with the correct answer, and add it to a dataset.
If none of the answer is correct, ignore that data point.

You should use the `CoTModel.batched_generate` function with `num_return_sequences > 1` and `temperature > 0` to produce a number of diverse outputs.
Using the `HuggingFaceTB/SmolLM2-1.7B-Instruct` model should further help you obtain better rollouts.
In our experiments, we had a 90+% success rate in generating this dataset (success = > 1 / 10 samples answered correctly).
Store the output in a json file in `data/rft.json`.
Here is a sample entry

```json
  [
    "How many gram are there per 6 kg?",
    6000.0,
    "1 kg = 1000 grams. 6 * 1000 = <answer>6000</answer>"
  ],
```

Modify your SFT code to train on this new data of question + reasoning pairs.

This model will likely perform better than SFT, but might need a slightly larger LoRA adapter.
Feel free to increase the rank as long as your total submission size is below 50Mb.

## Submission

Once you finished the assignment, create a submission bundle using:

```bash
python3 bundle.py homework [YOUR UT ID]
```

Delete any old checkpoints from your homework directory to keep the model size below 50MB.

Submit the zip file on Canvas. Please note that the maximum file size our grader accepts is **50MB**. Please keep your solution compact.
Please double-check that your zip file was properly created, by grading it again:

```bash
python3 -m grader [YOUR UT ID].zip
```

## Online grader

We will use an automated grader through Canvas to grade all your submissions. There is a soft limit of **5** submissions per assignment. Please contact the course staff before going over this limit, otherwise your submission might be counted as invalid.

The online grading system will use a slightly modified version of Python and the grader:

- Please do not use the `exit` or `sys.exit` command, it will likely lead to a crash in the grader
- Please do not try to access, read, or write files outside the ones specified in the assignment. This again will lead to a crash. File writing is disabled.
- Network access is disabled. Please do not try to communicate with the outside world.
- Forking is not allowed!
- `print` or `sys.stdout.write` statements from your code are ignored and not returned.

Please do not try to break or hack the grader. Doing so will have negative consequences for your standing in this class and the program.

## Installation

We encourage using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to install the required packages.

```bash
conda create --name advances_in_deeplearning python=3.12 -y
conda activate advances_in_deeplearning
```

First, install [PyTorch](https://pytorch.org/get-started/locally/)

Then install additional dependencies:

```bash
pip install -r requirements.txt
```
