from pathlib import Path
import torch

DEFAULT_MODEL_PATH = Path("/content/drive/MyDrive/pytorch_models")


def save_model(model, model_name, save_dir=None):
    """
    保存模型

    Args:
        model: PyTorch模型
        model_name: 文件名，例如 model.pth
        save_dir: 保存目录，默认 Google Drive
    """
    save_dir = Path(save_dir) if save_dir else DEFAULT_MODEL_PATH
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / model_name

    torch.save(model.state_dict(), save_path)

    print(f"✅ Saved: {save_path}")

    return save_path


def load_model(model, model_name, save_dir=None, device="cpu"):
    """
    加载模型
    """
    save_dir = Path(save_dir) if save_dir else DEFAULT_MODEL_PATH

    load_path = save_dir / model_name

    model.load_state_dict(
        torch.load(load_path, map_location=device)
    )

    print(f"✅ Loaded: {load_path}")

    return model


def list_models(save_dir=None, pattern="*.pth"):
    """
    查看目录中的模型
    """
    save_dir = Path(save_dir) if save_dir else DEFAULT_MODEL_PATH

    model_files = list(save_dir.glob(pattern))

    if not model_files:
        print("❌ No model files found.")
        return []

    for idx, file in enumerate(model_files, start=1):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"{idx}. {file.name} ({size_mb:.2f} MB)")

    return model_files