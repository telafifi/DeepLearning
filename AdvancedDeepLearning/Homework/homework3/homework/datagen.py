import json
from pathlib import Path

from .cot import CoTModel
from .data import DATA_DIR, Dataset, is_answer_valid


def generate_dataset(output_json: str = "data/rft.json", oversample: int = 10, temperature: float = 0.6):
    model = CoTModel()
    train_data = Dataset("train")

    questions = [train_data[i][0] for i in range(len(train_data))]
    correct_answers = [train_data[i][1] for i in range(len(train_data))]

    # Instead of num_return_sequences (slow), duplicate prompts and generate 1 each
    expanded_prompts = []
    expanded_indices = []
    for i, q in enumerate(questions):
        prompt = model.format_prompt(q)
        for _ in range(oversample):
            expanded_prompts.append(prompt)
            expanded_indices.append(i)

    # Generate all at once (batched with micro_batch_size=32 internally)
    all_generations = model.batched_generate(expanded_prompts, temperature=temperature)

    # Group by question index and find first correct answer
    from collections import defaultdict
    question_gens = defaultdict(list)
    for idx, gen in zip(expanded_indices, all_generations):
        question_gens[idx].append(gen)

    dataset = []
    for i, (question, correct_answer) in enumerate(zip(questions, correct_answers)):
        generations = question_gens[i]
        for gen in generations:
            try:
                parsed = model.parse_answer(gen)
                if parsed == parsed and is_answer_valid(parsed, correct_answer):
                    dataset.append([question, correct_answer, gen])
                    break
            except Exception:
                continue

    print(f"Generated {len(dataset)} / {len(train_data)} successful examples")

    output_path = Path(__file__).parent.parent / output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
