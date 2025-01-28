import os

from torch.utils.tensorboard import SummaryWriter


def get_next_run_dir(log_dir):
    """
    Get the next available run directory in log_dir.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        return os.path.join(log_dir, "run_0")

    # Scan for existing run directories
    existing_runs = [
        d for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("run_")
    ]

    # Extract run numbers and find the next available number
    run_numbers = [
        int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()
    ]
    next_run = max(run_numbers) + 1 if run_numbers else 0

    return os.path.join(log_dir, f"run_{next_run}")


def build_logger(base_folder_path, *args, **kwargs):
    log_dir = get_next_run_dir(base_folder_path)

    return SummaryWriter(log_dir, *args, **kwargs)

