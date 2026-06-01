from pathlib import Path

from huggingface_hub import hf_hub_download

TARGET_SR = 16000


def download_utmos_ckpt() -> Path:
    return Path(
        hf_hub_download(
            repo_id="Blinorot/UTMOS-PyTorch",
            filename="utmos_state_dict.pt",
            repo_type="model",
        )
    )


def download_scipted_utmos_ckpt() -> Path:
    return Path(
        hf_hub_download(
            repo_id="Blinorot/UTMOS-PyTorch",
            filename="utmos_scripted.pt",
            repo_type="model",
        )
    )
