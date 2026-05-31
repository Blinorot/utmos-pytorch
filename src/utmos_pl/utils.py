from pathlib import Path
from huggingface_hub import hf_hub_download

TARGET_SR = 16000

def download_utmos_ckpt() -> Path:
    return Path(
        hf_hub_download(
            repo_id="sarulab-speech/UTMOS-demo",
            filename="epoch=3-step=7459.ckpt",
            repo_type="space",
            revision="47212055c2ecfb02d40cec2395233b83295d3d30"
        )
    )

def download_w2v_ckpt() -> Path:
    return Path(
        hf_hub_download(
            repo_id="sarulab-speech/UTMOS-demo",
            filename="wav2vec_small.pt",
            repo_type="space",
            revision="47212055c2ecfb02d40cec2395233b83295d3d30"
        )
    )