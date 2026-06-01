import argparse

import numpy as np
import torch
import torchaudio
import wget
from huggingface_hub import HfApi, create_repo

from src.utmos_pl import UTMOSScore
from src.utmos_pytorch import UTMOSScoreTorch
from utils import ASSET_DATA_DIR, SCRIPT_DATA_DIR, TARGET_SR

SCRIPTED_CHECKPOINT = "utmos_scripted.pt"


def create_scripted_model(args):
    asset_url = "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav"
    asset_path = ASSET_DATA_DIR / "LJ037-0171.wav"
    if not asset_path.exists():
        wget.download(asset_url, str(asset_path))

    wav, sr = torchaudio.load(asset_path)
    if wav.shape[0] != 1:
        wav = wav[0:1]
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=TARGET_SR)

    device = "cpu"
    orig_utmos = UTMOSScore(device=device)
    torch_utmos = UTMOSScoreTorch(device=device)

    # # create TorchScript version
    script_utmos = torch.jit.script(torch_utmos)

    orig_score = orig_utmos.score(wav)
    print(f"Orig. UTMOS: {orig_score}")

    torch_score = torch_utmos.score(wav)
    torch_score = torch_score.detach().cpu().numpy()
    print(f"Torch UTMOS: {torch_score}")

    # there is no .score for a scripted module
    # use plain forward
    script_score = script_utmos(wav)
    script_score = script_score.detach().cpu().numpy()
    print(f"Script UTMOS: {script_score}")

    np.testing.assert_allclose(orig_score, torch_score, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(orig_score, script_score, rtol=1e-7, atol=1e-7)

    save_path = SCRIPT_DATA_DIR / SCRIPTED_CHECKPOINT
    script_utmos.save(save_path)

    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    api = HfApi()

    api.upload_file(
        path_or_fileobj=str(save_path),
        path_in_repo=SCRIPTED_CHECKPOINT,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload scripted UTMOS checkpoint",
    )

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a scripted version of PyTorch UTMOS"
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
    create_scripted_model(args)
