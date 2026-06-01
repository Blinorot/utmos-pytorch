<h1 align="center">
UTMOS-PyTorch
</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#usage">Usage</a> •
  <a href="#how-to-reproduce">How To Reproduce</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
<a href="https://pypi.org/project/utmos-pytorch/">
  <img src="https://img.shields.io/pypi/v/utmos-pytorch.svg?logo=pypi&logoColor=white&cacheSeconds=300" alt="PyPI version">
</a>
<a href="https://pypi.org/project/utmos-pytorch/">
  <img src="https://img.shields.io/pypi/pyversions/utmos-pytorch.svg?logo=python&logoColor=white&color=blue&cacheSeconds=300" alt="Python versions">
</a>
<a href="https://huggingface.co/Blinorot/UTMOS-PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Model-yellow.svg?logo=huggingface&logoColor=white" alt="Hugging Face model">
</a>
<a href="https://github.com/Blinorot/UTMOS-PyTorch/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
</a>
<a href="https://arxiv.org/abs/2204.02152">
  <img src="https://img.shields.io/badge/Paper-arXiv%3A2204.02152-b31b1b.svg?logo=arxiv&logoColor=white" alt="UTMOS paper">
</a>
</p>

## About

This is an unofficial `fairseq`-free implementation of the UTMOS MOS Prediction system proposed in [UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022](https://arxiv.org/abs/2204.02152).

The [original implementation](https://github.com/sarulab-speech/UTMOS22) is based on [fairseq](https://github.com/facebookresearch/fairseq). However, `fairseq` is difficult to install with recent Python, PyTorch, and dependency versions, which makes UTMOS hard to use in modern environments. [Recent study from ICASSP 2026](https://arxiv.org/abs/2509.24457) highlights the high correlation of UTMOS with subjective listening scores for neural codecs. Therefore, modern neural audio codec and TTS research benefits from an easy-to-install UTMOS implementation.

We provide a `fairseq`-free implementation written in `PyTorch` that matches the [original system](https://github.com/sarulab-speech/UTMOS22) using converted weights and re-written modules.

We also provide a `TorchScript` variant that can be loaded with only PyTorch, without installing this package.

The PyTorch and TorchScript versions are validated against the original implementation and produce matching scores.

> [!NOTE]
> As in the original version, we recommend running UTMOS with batch size 1 to avoid metric shifts caused by padding.

## Usage

You can install the repo as a package:

```bash
pip install utmos-pytorch
```

Or from source:

```bash
git clone https://github.com/Blinorot/UTMOS-PyTorch.git
cd UTMOS-PyTorch
pip install -e .
```

The code requires:

| Package         | Version |
| --------------- | ------- |
| Python          | >=3.9   |
| PyTorch         | >=2.2.0 |
| HuggingFace Hub | >=0.20  |

The TorchScript checkpoint was scripted with `PyTorch 2.5.1`. We have tested that it works on `PyTorch 2.2.0`, however, `PyTorch >=2.5.1` is recommended for the
TorchScript variant.

Then, you can run the model as follows:

```python
import torchaudio
from utmos_pytorch import UTMOSScoreTorch

device = "cpu" # set to "cuda" to use on GPU
utmos = UTMOSScoreTorch(device=device) # already in eval mode

# load an audio file, e.g. using torchaudio
audio_path = ... # path to an audio file
wav, sr = torchaudio.load(audio_path)

# convert to MONO 16 kHz
TARGET_SR = 16000
if wav.shape[0] != 1:
    wav = wav[0:1]
if sr != TARGET_SR:
    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=TARGET_SR)

# put on device
wav = wav.to(device)

# calculate the score
# accepts T, 1xT, Bx1xT
utmos_score = utmos.score(wav) # tensor of shape (batch_size,)
```

You can replace `UTMOSScoreTorch` with `UTMOSScoreScripted` to use the `TorchScript` variant instead. On first use, the package downloads converted UTMOS weights from [Hugging Face Hub](https://huggingface.co/Blinorot/UTMOS-PyTorch) and caches them locally using the Hugging Face cache.

For `TorchScript`, you can avoid downloading the package and use the model directly:

```python
import torch
import torchaudio
import wget

# download scripted checkpoint, e.g. using wget
checkpoint_url = "https://huggingface.co/Blinorot/UTMOS-PyTorch/resolve/main/utmos_scripted.pt"
checkpoint_path = ... # path to saved checkpoint
wget.download(checkpoint_url, checkpoint_path)

# load directly with torch.jit
device = "cpu" # set to "cuda" to use on GPU
utmos = torch.jit.load(checkpoint_path, map_location=device)
utmos.eval()

# load an audio file, e.g. using torchaudio
audio_path = ... # path to an audio file
wav, sr = torchaudio.load(audio_path)

# convert to MONO 16 kHz
TARGET_SR = 16000
if wav.shape[0] != 1:
    wav = wav[0:1]
if sr != TARGET_SR:
    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=TARGET_SR)

# put on device
wav = wav.to(device)

# calculate the score
# accepts T, 1xT, Bx1xT
with torch.no_grad():
    utmos_score = utmos.score(wav) # tensor of shape (batch_size,)
```

### Notes

The model expects audio sampled at **16 kHz**.

Accepted tensor shapes:

| Shape       | Meaning                                     |
| ----------- | ------------------------------------------- |
| `(T,)`      | single mono waveform                        |
| `(1, T)`    | single mono waveform with channel dimension |
| `(B, 1, T)` | batch of mono waveforms                     |

The input should be a floating point PyTorch tensor. Stereo audio should be converted to mono before scoring. `utmos.score(wav)` returns a tensor of shape `(batch_size,)`, where each value is a predicted MOS score. Higher is better. **Batch size 1 is recommended to avoid padding-related score shifts.**

API classes:

| Class                | Description                                     |
| -------------------- | ----------------------------------------------- |
| `UTMOSScoreTorch`    | PyTorch implementation using converted weights. |
| `UTMOSScoreScripted` | Wrapper around the TorchScript checkpoint.      |

## How To Reproduce

To reproduce PyTorch and Scripted checkpoints and validate them against the original UTMOS module, follow the steps below.

First, install all required packages in a new environment:

```bash
# Optional
conda create -n utmos python=3.9.7
conda activate utmos

pip install pip==22.0
pip install -r requirements.txt
```

Then, you need to export weights from the original UTMOS checkpoint:

```bash
# add --private to save privately
python extract_state_dict.py --repo-id USERNAME/REPO_NAME_ON_HUGGINGFACE
```

This will upload the state dict extracted from the original PyTorch Lightning UTMOS checkpoint to Hugging Face. The same state dict is used to load our `fairseq`-free PyTorch-only module.

To create a scripted version of the PyTorch model that allows to load UTMOS without class definitions, run

```bash
# add --private to save privately
python create_scripted_model.py --repo-id USERNAME/REPO_NAME_ON_HUGGINGFACE
```

It will upload the scripted model to HuggingFace as well.

Finally, to test that all 3 variations (Original, PyTorch, Scripted) return the same scores, run

```bash
# set --device "cpu" to run on cpu
# set --batch-size to a value bigger than 1 to test batched version
python test.py --device "cuda" --batch-size 1
```

The models are tested on `test-clean` partition of [LibriSpeech](https://www.openslr.org/12).

| UTMOS Version | Score (LibriSpeech Test-Clean) |
| ------------- | -----------------------------: |
| Original      |              4.085875394599128 |
| Torch         |              4.085875394599128 |
| Scripted      |              4.085875394599128 |

## Credits

The code is based on the original [UTMOS](https://github.com/sarulab-speech/UTMOS22) and [fairseq](https://github.com/facebookresearch/fairseq) repositories.

## License

This project is released under the [MIT License](./LICENSE).

Parts of the implementation are adapted from the original UTMOS and fairseq repositories, which are also MIT licensed. See [LICENSES](./LICENSES) for third-party license texts.

Converted checkpoints are derived from the original UTMOS checkpoint. Original authors retain copyright over the original model and weights.

## Citation

If you use this package, please cite the original UTMOS paper:

```bibtex
@inproceedings{saeki22c_interspeech,
  title     = {{UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022}},
  author    = {Takaaki Saeki and Detai Xin and Wataru Nakata and Tomoki Koriyama and Shinnosuke Takamichi and Hiroshi Saruwatari},
  year      = {2022},
  booktitle = {{Interspeech 2022}},
  pages     = {4521--4525},
  doi       = {10.21437/Interspeech.2022-439},
  issn      = {2958-1796},
}
```
