import matplotlib.pyplot as plt
import torch

from .datasets.road_dataset import load_data
from .models import load_model


class Visualizer:
    def __init__(self):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(10, 4),
        )

        self.fig = fig
        self.axes = axes

    def process(
        self,
        image: torch.Tensor,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        waypoints: torch.Tensor,
        waypoints_mask: torch.Tensor,
        pred: torch.Tensor,
        batch_idx: int = 0,
    ):
        track_left = track_left[batch_idx].detach().cpu().numpy()
        track_right = track_right[batch_idx].detach().cpu().numpy()
        waypoints = waypoints[batch_idx].detach().cpu().numpy()
        waypoints_mask = waypoints_mask[batch_idx].detach().cpu().numpy()
        pred = pred[batch_idx].detach().cpu().numpy()

        _, axes = self.fig, self.axes

        for ax in axes:
            ax.clear()

        axes[0].imshow(image[batch_idx].detach().cpu().numpy().transpose(1, 2, 0))
        axes[0].axis("off")

        axes[1].plot(track_left[:, 0], track_left[:, 1], "ro-")
        axes[1].plot(track_right[:, 0], track_right[:, 1], "bo-")
        axes[1].plot(waypoints[:, 0], waypoints[:, 1], "g--o")
        # axes[1].plot(pred[:, 0], pred[:, 1], "c--o")

        axes[1].set_xlim(-10, 10)
        axes[1].set_ylim(-5, 15)
        axes[1].axis("equal")


def visualize(
    data_path: str = "drive_data/val",
    model_name: str = "mlp_planner",
    device_str: str = "cuda",
):
    torch.manual_seed(0)

    device = torch.device(device_str)
    data = load_data(
        data_path,
        num_workers=0,
        batch_size=1,
        shuffle=True,
    )

    model = load_model(model_name, with_weights=True).to(device)
    model.eval()

    viz = Visualizer()

    for i, batch in enumerate(data):
        batch = {k: v.to(device) for k, v in batch.items()}
        image = batch["image"]
        track_left = batch["track_left"]
        track_right = batch["track_right"]
        waypoints = batch["waypoints"]
        waypoints_mask = batch["waypoints_mask"]

        pred = model(**batch)

        viz.process(
            image,
            track_left,
            track_right,
            waypoints,
            waypoints_mask,
            pred,
            batch_idx=0,
        )

        plt.show()

        # just show one sample
        break


if __name__ == "__main__":
    visualize()
