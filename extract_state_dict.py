from src.utmos_pl import UTMOSScore
from pathlib import Path
import torch
import argparse
from huggingface_hub import create_repo, HfApi

DATA_DIR = Path(__file__).resolve().parent / "data" / "checkpoints"
STATE_DICT_NAME = "utmos_state_dict.pt"


def export_lightning_state_dict(args):
    """
    1. Loads original Lightning UTMOS model through UTMOSScore.
    2. Saves only its plain state_dict into DATA_DIR.
    3. Uploads weights on HuggingFace
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    original_utmos = UTMOSScore(device="cpu")
    lightning_model = original_utmos.model.eval().to("cpu")

    save_path = DATA_DIR / STATE_DICT_NAME
    torch.save(lightning_model.state_dict(), save_path)

    print(f"Saved Lightning state_dict to: {save_path}")

    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    api = HfApi()

    api.upload_file(
        path_or_fileobj=str(save_path),
        path_in_repo=STATE_DICT_NAME,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload UTMOS checkpoint",
    )

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract state dict from the original UTMOS"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Blinorot/UTMOS-PyTorch",
        help="Target Hugging Face Hub repo ID.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/push to a private Hub repository.",
    )
    args = parser.parse_args()
    export_lightning_state_dict(args)
