"""
Do not modify unless you know what you are doing!
"""

import numpy as np
import torch

from .datasets import classification_dataset, road_dataset
from .grader import Case, Grader
from .metrics import AccuracyMetric, DetectionMetric

# A hidden test split will be used for grading
CLASSIFICATION_DATA_SPLIT = "classification_data/val"
ROAD_DATA_SPLIT = "road_data/val"


def normalized_score(val: float, low: float, high: float):
    """
    Normalizes and clips the value to the range [0, 1]
    """
    return np.clip((val - low) / (high - low), 0, 1)


class BaseGrader(Grader):
    """
    Helper for loading models and checking their correctness
    """

    KIND: str = None
    METRIC = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            raise RuntimeError("No hardware acceleration available")

        self.data = None

        self._model = None
        self._metrics_computed = False
        self._metric_computer = self.METRIC()

    @property
    def model(self):
        """
        Lazily loads the model
        """
        if self._model is None:
            self._model = self.module.load_model(self.KIND, with_weights=True)
            self.model.to(self.device)

        return self._model

    @property
    def metrics(self):
        """
        Runs the model on the data and computes metrics
        """
        if not self._metrics_computed:
            self.compute_metrics()
            self._metrics_computed = True

        return self._metric_computer.compute()

    @torch.inference_mode()
    def compute_metrics(self):
        """
        Implemented by subclasses depending on the model
        """
        raise NotImplementedError


class ClassifierGrader(BaseGrader):
    """Classifier"""

    KIND = "classifier"
    METRIC = AccuracyMetric
    RANGE = 0.6, 0.8, 0.9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = classification_dataset.load_data(
            CLASSIFICATION_DATA_SPLIT,
            num_workers=1,
            batch_size=64,
            shuffle=False,
            transform_pipeline="default",
        )

    @torch.inference_mode()
    def compute_metrics(self):
        self.model.eval()

        for img, label in self.data:
            img = img.to(self.device)
            pred = self.model.predict(img)

            self._metric_computer.add(pred, label)

    @Case(score=10, timeout=5000)
    def test_model(self):
        """Predict"""
        batch_size = 16
        dummy_data = torch.rand(batch_size, 3, 64, 64).to(self.device)
        model = self.module.load_model(self.KIND, with_weights=False).to(self.device)
        output = model.predict(dummy_data)

        assert output.shape == (batch_size,), f"Expected shape ({batch_size},), got {output.shape}"

    @Case(score=25, timeout=10000)
    def test_accuracy(self):
        """Accuracy"""
        key = "accuracy"
        val = self.metrics[key]
        score = normalized_score(val, self.RANGE[0], self.RANGE[1])

        return score, f"{key}: {val:.3f}"

    @Case(score=2, timeout=500, extra_credit=True)
    def test_accuracy_extra(self):
        """Accuracy: Extra Credit"""
        key = "accuracy"
        val = self.metrics[key]
        score = normalized_score(val, self.RANGE[1], self.RANGE[2])

        return score


class RoadDetectorGrader(BaseGrader):
    """Detector"""

    KIND = "detector"
    METRIC = DetectionMetric
    RANGE_IOU = 0.35, 0.55, 0.75
    RANGE_ACCURACY = 0.95, 0.96, 0.98

    # lower is better
    RANGE_DEPTH = 0.03, 0.05, 0.08
    RANGE_TP_DEPTH = 0.035, 0.05, 0.09

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = road_dataset.load_data(
            ROAD_DATA_SPLIT,
            num_workers=2,
            batch_size=16,
            shuffle=False,
            transform_pipeline="default",
        )

    @torch.inference_mode()
    def compute_metrics(self):
        self.model.eval()

        for batch in self.data:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            image = batch["image"]
            track = batch["track"]
            depth = batch["depth"]

            pred, pred_depth = self.model.predict(image)

            self._metric_computer.add(pred, track, depth, pred_depth)

    @Case(score=10, timeout=1000)
    def test_model(self):
        """Predict"""
        batch_size = 4
        dummy_data = torch.rand(batch_size, 3, 96, 128).to(self.device)
        model = self.module.load_model(self.KIND, with_weights=False).to(self.device)
        output = model.predict(dummy_data)

        assert len(output) == 2, f"Expected 2 outputs, got {len(output)}"

        pred, pred_depth = output

        assert pred.shape == (batch_size, 96, 128), f"Label shape: {pred.shape}"
        assert pred_depth.shape == (batch_size, 96, 128), f"Depth shape: {pred_depth.shape}"

    @Case(score=10, timeout=10000)
    def test_accuracy(self):
        """Segmentation Accuracy"""
        key = "accuracy"
        val = self.metrics[key]
        score = normalized_score(val, self.RANGE_ACCURACY[0], self.RANGE_ACCURACY[1])

        return score, f"{key}: {val:.3f}"

    @Case(score=25, timeout=500)
    def test_iou(self):
        """Segmentation IoU"""
        key = "iou"
        val = self.metrics[key]
        score = normalized_score(val, self.RANGE_IOU[0], self.RANGE_IOU[1])

        return score, f"{key}: {val:.3f}"

    @Case(score=2, timeout=500, extra_credit=True)
    def test_iou_extra(self):
        """Segmentation IoU: Extra Credit"""
        key = "iou"
        val = self.metrics[key]
        score = normalized_score(val, self.RANGE_IOU[1], self.RANGE_IOU[2])

        return score

    @Case(score=10, timeout=500)
    def test_abs_depth_error(self):
        """Depth Error"""
        key = "abs_depth_error"
        val = self.metrics[key]
        score = normalized_score(val, self.RANGE_DEPTH[1], self.RANGE_DEPTH[2])

        # lower is better
        score = 1 - score

        return score, f"{key}: {val:.3f}"

    @Case(score=2, timeout=500, extra_credit=True)
    def test_abs_depth_error_extra(self):
        """Depth Error: Extra Credit"""
        key = "abs_depth_error"
        val = self.metrics[key]
        score = normalized_score(val, self.RANGE_DEPTH[0], self.RANGE_DEPTH[1])

        # lower is better
        score = 1 - score

        return score

    @Case(score=10, timeout=500)
    def test_tp_depth_error(self):
        """True Positives Depth Error"""
        key = "tp_depth_error"
        val = self.metrics[key]
        score = normalized_score(val, self.RANGE_TP_DEPTH[1], self.RANGE_TP_DEPTH[2])

        # Boosting IoU will allow for true positives
        assert val > 1e-2, "Model does not detect any true positives"

        # lower is better
        score = 1 - score

        return score, f"{key}: {val:.3f}"
