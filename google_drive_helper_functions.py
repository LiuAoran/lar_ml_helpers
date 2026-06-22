from pathlib import Path
import torch

DEFAULT_MODEL_PATH = Path("/content/drive/MyDrive/pytorch_models")
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, model_name, save_dir=None):
    """
    Save a PyTorch model.

    Args:
        model: PyTorch model.
        model_name: Model file name, e.g. "model.pth".
        save_dir: Directory to save the model. Defaults to Google Drive.

    Returns:
        Path to the saved model.
    """
    save_dir = Path(save_dir) if save_dir else DEFAULT_MODEL_PATH
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / model_name

    torch.save(model.state_dict(), save_path)

    print(f"✅ Model saved: {save_path}")

    return save_path


def load_model(model, model_name, save_dir=None, device=None):
    """
    Load a PyTorch model.

    Args:
        model: PyTorch model instance.
        model_name: Model file name.
        save_dir: Directory containing the model.
        device: Device to load the model onto.

    Returns:
        Loaded model.
    """
    save_dir = Path(save_dir) if save_dir else DEFAULT_MODEL_PATH

    load_path = save_dir / model_name

    device = device if device else DEFAULT_DEVICE

    model.load_state_dict(
        torch.load(load_path, map_location=device)
    )

    print(f"✅ Model loaded: {load_path}")

    return model


def list_models(save_dir=None, pattern="*.pth"):
    """
    List all model files in a directory.

    Args:
        save_dir: Directory containing models.
        pattern: File pattern to match.

    Returns:
        List of model files.
    """
    save_dir = Path(save_dir) if save_dir else DEFAULT_MODEL_PATH

    model_files = list(save_dir.glob(pattern))

    if not model_files:
        print("❌ No model files found.")
        return []

    print("-" * 50)

    for idx, file in enumerate(model_files, start=1):
        size_mb = file.stat().st_size / (1024 * 1024)

        print(
            f"{idx}. {file.name} ({size_mb:.2f} MB)"
        )

    print("-" * 50)

    return model_files