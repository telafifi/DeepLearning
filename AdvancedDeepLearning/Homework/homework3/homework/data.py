import json
from dataclasses import dataclass
from pathlib import Path

from .base_llm import BaseLLM

DATA_DIR = Path(__file__).parent.parent / "data"


class Dataset:
    def __init__(self, split: str):
        with (DATA_DIR / f"{split}.json").open() as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def is_answer_valid(answer: float, correct_answer: float, relative_tolerance: float = 0.05) -> bool:
    return abs(round(answer, 3) - round(correct_answer, 3)) < relative_tolerance * abs(round(correct_answer, 3))


@dataclass
class BenchmarkResult:
    @dataclass
    class Sample:
        question: str
        answer: float
        correct_answer: float
        is_correct: bool

    accuracy: float
    answer_rate: float  # How often was the answer not NaN
    samples: list[Sample]

    @classmethod
    def from_answers(cls, answers: list[float], dataset: Dataset, max_question: int) -> "BenchmarkResult":
        samples = [
            cls.Sample(
                question=item[0], answer=answer, correct_answer=item[1], is_correct=is_answer_valid(answer, item[1])
            )
            for item, answer in zip(dataset, answers[:max_question])
        ]
        n = min(len(dataset), max_question)
        return cls(
            accuracy=sum(sample.is_correct for sample in samples) / n,
            answer_rate=sum(sample.answer == sample.answer for sample in samples) / n,
            samples=samples,
        )


def benchmark(func: BaseLLM, dataset: Dataset, max_question: int) -> BenchmarkResult:
    idx = range(min(len(dataset), max_question))
    questions = [dataset[i][0] for i in idx]
    answers = func.answer(*questions)
    return BenchmarkResult.from_answers(answers, dataset, max_question)


if __name__ == "__main__":
    print(Dataset("train")[0])
