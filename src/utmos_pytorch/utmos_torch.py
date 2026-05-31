import torch
import torch.nn as nn
from typing import Optional
import os

from .wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
from .utils import download_utmos_ckpt


class SSL_model(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim) -> None:
        super(SSL_model, self).__init__()
        self.ssl_model, self.ssl_out_dim = ssl_model, ssl_out_dim

    def forward(self, wav, domains, judge_id):
        wav = wav.squeeze(1)  # [batches, audio_len]
        res, padding_mask, unmasked_features = self.ssl_model(
            wav, mask=False, features_only=True
        )
        return res

    def get_output_dim(self):
        return self.ssl_out_dim


class DomainEmbedding(nn.Module):
    def __init__(self, n_domains, domain_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_domains, domain_dim)
        self.output_dim = domain_dim

    def forward(self, wav, domains, judge_id):
        return self.embedding(domains)

    def get_output_dim(self):
        return self.output_dim


class LDConditioner(nn.Module):
    """
    Conditions ssl output by listener embedding
    """

    def __init__(self, input_dim, judge_dim, num_judges=None):
        super().__init__()
        self.input_dim = input_dim
        self.judge_dim = judge_dim
        self.num_judges = num_judges
        assert num_judges != None
        self.judge_embedding = nn.Embedding(num_judges, self.judge_dim)
        # concat [self.output_layer, phoneme features]

        self.decoder_rnn = nn.LSTM(
            input_size=self.input_dim + self.judge_dim,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )  # linear?
        self.out_dim = self.decoder_rnn.hidden_size * 2

    def get_output_dim(self):
        return self.out_dim

    def forward(self, ssl_feature, domain_feature, wav, domains, judge_id):
        judge_ids = judge_id
        
        concatenated_feature = torch.cat(
            (ssl_feature, domain_feature.unsqueeze(1).expand(-1, ssl_feature.size(1), -1),),
            dim=2,
        )
        concatenated_feature = torch.cat(
            (
                concatenated_feature,
                self.judge_embedding(judge_ids).unsqueeze(1).expand(-1, concatenated_feature.size(1), -1),
            ),
            dim=2,
        )
        decoder_output, (h, c) = self.decoder_rnn(concatenated_feature)
        return decoder_output


class Projection(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, range_clipping=False):
        super(Projection, self).__init__()
        self.range_clipping = range_clipping
        output_dim = 1
        # if range_clipping:
        #     self.proj = nn.Tanh()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), activation, nn.Dropout(0.3), nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x, wav, domains, judge_id):
        output = self.net(x)

        # range clipping
        # if self.range_clipping:
        #     return self.proj(output) * 2.0 + 3
        # else:
        #     return output
        return output

    def get_output_dim(self):
        return self.output_dim


class UTMOSModel(nn.Module):
    """
    Plain nn.Module version of BaselineLightningModule.

    It intentionally uses the same submodule names:
        self.feature_extractors
        self.output_layers

    Therefore state_dict keys are compatible with:
        UTMOSScore's BaselineLightningModule.state_dict()

    Wav2Vec2 Module is replaced with a fairseq-free variant.
    """

    def __init__(self):
        super().__init__()

        ssl_out_dim = 768
        cfg = Wav2Vec2Config()
        wav2vec = Wav2Vec2Model(cfg)
        
        self.feature_extractors = nn.ModuleList(
            [
                SSL_model(wav2vec, ssl_out_dim),
                DomainEmbedding(3, 128),
            ]
        )

        output_dim = sum(
            feature_extractor.get_output_dim()
            for feature_extractor in self.feature_extractors
        )

        output_layers = [
            LDConditioner(
                judge_dim=128,
                num_judges=3000,
                input_dim=output_dim,
            )
        ]

        output_dim = output_layers[-1].get_output_dim()

        output_layers.append(
            Projection(
                hidden_dim=2048,
                activation=nn.ReLU(),
                range_clipping=False,
                input_dim=output_dim,
            )
        )

        self.output_layers = nn.ModuleList(output_layers)

    def forward(self, wav, domains, judge_id):
        ssl_feature = self.feature_extractors[0](wav, domains, judge_id)
        domain_feature = self.feature_extractors[1](wav, domains, judge_id)

        x = self.output_layers[0](ssl_feature, domain_feature, wav, domains, judge_id)
        x = self.output_layers[1](x, wav, domains, judge_id)

        return x


def remove_weight_norm_for_jit(model):
    for module in model.modules():
        try:
            nn.utils.remove_weight_norm(module, name="weight")
        except ValueError:
            pass
    return model


class UTMOSScoreTorch(nn.Module):
    """
    Fairseq-free PyTorch version of UTMOS from https://arxiv.org/abs/2204.02152.
    Based on original code from https://github.com/sarulab-speech/UTMOS22

    Same scoring interface as original UTMOSScore, but uses plain nn.Module.
    Accepts audio in 16kHz sampling rate. The audio should be resampled by the user
    before passing to the model.
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        device: str = "cpu"):
        """
        Args:
            ckpt_path: path to pretrained state_dict of UTMOS strong learner.
                If None, downloads weights from HuggingFace.
        """
        super().__init__()
        if ckpt_path is None:
            ckpt_path = str(download_utmos_ckpt())

        self.device = device
        self.model = UTMOSModel()
        self.model.to(device)
        self.load_utmos_weights(ckpt_path, device)

    def load_utmos_weights(self, ckpt_path, device):
        """
        Loads converted weights.
        """

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Missing {ckpt_path}. Run export_lightning_state_dict() first."
            )

        state_dict = torch.load(ckpt_path, map_location=device)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        print("Torch UTMOS Missing keys:")
        for k in missing_keys:
            print(k)

        print("Torch UTMOS Unexpected keys:")
        for k in unexpected_keys:
            print(k)

        self.model.eval()

        remove_weight_norm_for_jit(self.model)

        for module in self.model.modules():
            if isinstance(module, torch.nn.LSTM):
                module.flatten_parameters()
    
    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wavs (torch.Tensor): audio in 16kHz. The audio must be resampled by
                the user before passing to the model.
        Returns:
            scores (torch.Tensor): tensor with B scores, one per element in batch.
        """
        if len(wavs.shape) == 1:
            out_wavs = wavs.unsqueeze(0).unsqueeze(0)
        elif len(wavs.shape) == 2:
            out_wavs = wavs.unsqueeze(0)
        elif len(wavs.shape) == 3:
            out_wavs = wavs
        else:
            raise ValueError("Dimension of input tensor needs to be <= 3.")

        bs = out_wavs.shape[0]

        output = self.model(
            wav=out_wavs,
            domains=torch.zeros(bs, dtype=torch.int, device=out_wavs.device),
            judge_id=torch.ones(bs, dtype=torch.int, device=out_wavs.device) * 288,
        )

        return output.mean(dim=1).squeeze(1).cpu() * 2 + 3
    
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
