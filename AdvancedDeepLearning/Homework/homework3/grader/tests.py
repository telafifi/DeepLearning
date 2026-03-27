import numpy as np
import torch

from .grader import Case, Grader

CKPT_TEMPLATE = "*_{}.pth"

MAX_NUM_PARAMS = 380000000  # 380M

def model_size_check(model):
    num_params = sum(p.numel() for p in model.parameters())
    if num_params > MAX_NUM_PARAMS:
        raise ValueError(
            f"Model has {num_params} parameters, which is greater than the maximum allowed {MAX_NUM_PARAMS}"
        )

class GenerateGrader(Grader):
    """Model non-batched inference grader"""

    TEST_SAMPLE_SIZE = 32
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    LOSS_BOUND = 6.2, 8.0

    def load_model(self) -> torch.nn.Module:
        llm = self.module.BaseLLM()
        model_size_check(llm.model)
        llm.model.eval()
        return llm

    def generate(self, model, questions):
        import tqdm

        answers = []
        for i in tqdm.tqdm(range(len(questions))):
            answer = model.generate(questions[i])
            answers.append(answer)
        return answers

    def normalize_score(self, loss, min_loss, max_loss):
        """
        Returns a score based on model's loss normalized to [0, 1]

        If the loss is less than or equal to min_loss, you get 1.0 (full score)
        If the loss is greater than or equal to max_loss, you get 0.0 (no points)
        Otherwise, score is linearly interpolated between these extremes
        """
        # Normalize so that lower loss gives higher score
        score_normalized = 1.0 - (loss - min_loss) / (max_loss - min_loss)
        return np.clip(score_normalized, 0.0, 1.0)

    def compute_loss(self, model, full_texts):
        """
        Compute the loss of the model on the full texts.
        """
        with torch.no_grad():
            tokens = model.tokenizer(full_texts, return_tensors="pt", padding=True)
            answer_output = model.model(
                input_ids=tokens["input_ids"].to(self.device), attention_mask=tokens["attention_mask"].to(self.device)
            )
            logits = answer_output.logits
            logits = logits[..., :-1, :].contiguous()
            labels = tokens["input_ids"][..., 1:].contiguous().to(self.device)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss = loss * tokens["attention_mask"][..., 1:].contiguous().to(self.device)
            loss = loss.sum() / tokens["attention_mask"][..., 1:].sum()
            return loss.cpu().item()

    def check_generate_score(self):
        llm = self.load_model()
        dataset = self.module.data.Dataset("valid")
        questions = [dataset[i][0] for i in range(self.TEST_SAMPLE_SIZE)]
        answers = self.generate(llm, questions)
        full_texts = [questions[i] + answers[i] for i in range(len(questions))]
        return self.compute_loss(llm, full_texts)

    @Case(score=10, timeout=40000)
    def test_generate(self):
        """Test non-batched generate function"""
        return self.normalize_score(self.check_generate_score(), *self.LOSS_BOUND)


class BatchedGenerateGrader(GenerateGrader):
    """Model batched inference grader"""

    def generate(self, model, questions):
        return model.batched_generate(questions)

    @Case(score=15, timeout=15000)
    def test_generate(self):
        """Test batched generate function"""
        return self.normalize_score(self.check_generate_score(), *self.LOSS_BOUND)


class CoTGrader(Grader):
    """CoT Model Grader"""

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    VALIDATION_ACC_BOUND = 0.0, 0.4
    model_name = "cot"

    def load_model(self) -> torch.nn.Module:
        llm = getattr(self.module, f"load_{self.model_name}")()
        model_size_check(llm.model)
        llm.model.eval()
        return llm

    def normalize_score(self, score, min_score, max_score):
        """
        Returns a score based on model's score normalized to [0, 1]

        If the score is greater than or equal to max_score, you get 1.0 (full score)
        If the score is less than or equal to min_score, you get 0.0 (no points)
        Otherwise, score is linearly interpolated between these extremes
        """
        score_normalized = (score - min_score) / (max_score - min_score)
        return np.clip(score_normalized, 0.0, 1.0)

    @Case(score=25, timeout=60000)
    def test_validation_loss(self):
        """Test the answer accuracy"""
        dataset = self.module.data.Dataset("valid")
        model = self.load_model()
        benchmark_result = self.module.data.benchmark(model, dataset, 100)
        print(benchmark_result.accuracy)

        return self.normalize_score(benchmark_result.accuracy, *self.VALIDATION_ACC_BOUND)


class SFTGrader(CoTGrader):
    """SFT Model Grader"""

    VALIDATION_ACC_BOUND = 0.4, 0.6
    model_name = "sft"


class RFTGrader(CoTGrader):
    """RFT Model Grader"""

    VALIDATION_ACC_BOUND = 0.6, 0.7
    EXTRA_CREDIT_ACC_BOUND = 0.7, 0.8
    EXTRA_CREDIT_RATIO = 5.0 / 25.0
    model_name = "rft"

    def normalize_score(self, score, min_score, max_score):
        """
        If the score is greater than or equal to extra_score lower bound, you get extra credit
        """
        normal_score = super().normalize_score(score, min_score, max_score)

        extra_bounds = self.EXTRA_CREDIT_ACC_BOUND
        extra_score_normalized = (score - extra_bounds[0]) / (extra_bounds[1] - extra_bounds[0])
        extra_score = np.clip(extra_score_normalized, 0.0, 1.0) * self.EXTRA_CREDIT_RATIO

        return normal_score + extra_score
