from typing import Optional

import torch

from .utils import download_scipted_utmos_ckpt


def get_scripted_utmos(ckpt_path, device):
    """
    Get plain scripted UTMOS via torch.jit.load

    Args:
        ckpt_path (str | None): path to scripted UTMOS strong learner.
            If None, downloads weights from HuggingFace.
        device (str): device to load the model to.
    """
    if ckpt_path is None:
        ckpt_path = str(download_scipted_utmos_ckpt())
    model = torch.jit.load(ckpt_path, map_location=device)
    model.eval()
    return model


class UTMOSScoreScripted:
    """
    Fairseq-free Scripted-PyTorch version of UTMOS from https://arxiv.org/abs/2204.02152.
    Based on original code from https://github.com/sarulab-speech/UTMOS22

    Same scoring interface as original UTMOSScore, but uses scripted nn.Module.
    Accepts audio in 16kHz sampling rate. The audio should be resampled by the user
    before passing to the model.

    This class is a thin wrapper around get_scripted_utmos that makes the supported
    arguments and call semantics explicit.
    """

    def __init__(self, ckpt_path: Optional[str] = None, device: str = "cpu"):
        """
        Args:
            ckpt_path: path to pretrained state_dict of UTMOS strong learner.
                If None, downloads weights from HuggingFace.
        """
        super().__init__()
        self.device = device
        self.model = get_scripted_utmos(ckpt_path, device)
        self.model.to(device)

    def __call__(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wavs (torch.Tensor): audio in 16kHz. The audio must be resampled by
                the user before passing to the model.
        Returns:
            scores (torch.Tensor): tensor with B scores, one per element in batch.
        """
        return self.forward(wavs)

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wavs (torch.Tensor): audio in 16kHz. The audio must be resampled by
                the user before passing to the model.
        Returns:
            scores (torch.Tensor): tensor with B scores, one per element in batch.
        """
        return self.model(wavs)

    @torch.no_grad()
    def score(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        Calculate UTMOS Score.

        Args:
            wavs (torch.Tensor): audio in 16kHz. The audio must be resampled by
                the user before passing to the model.
        Returns:
            scores (torch.Tensor): tensor with B scores, one per element in batch.
        """
        return self.forward(wavs)
