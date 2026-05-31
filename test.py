from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import argparse

from src.utmos_pl import UTMOSScore
from src.utmos_pytorch import UTMOSScoreTorch, UTMOSScoreScripted
from data import LibrispeechDataset


def test_scores_on_dataset(
    orig_utmos,
    torch_utmos,
    script_utmos,
    dataset
):
    final_orig_score = 0
    final_torch_score = 0
    final_script_score = 0
    for wav in tqdm(dataset):
        wav = wav.to(device)
        orig_score = orig_utmos.score(wav)
        torch_score = torch_utmos.score(wav)
        script_score = script_utmos.score(wav)

        torch_score = torch_score.numpy()
        script_score = script_score.numpy()

        np.testing.assert_allclose(
            torch_score,
            orig_score,
            rtol=1e-6,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            script_score,
            orig_score,
            rtol=1e-6,
            atol=1e-6,
        )

        final_orig_score += orig_score.item()
        final_torch_score += torch_score.item()
        final_script_score += script_score.item()

    final_orig_score = final_orig_score / len(dataset)
    final_torch_score = final_torch_score / len(dataset)
    final_script_score = final_script_score / len(dataset)

    print("Final Results:")
    print("Orig:", final_orig_score)
    print("Torch:", final_torch_score)
    print("Script:", final_script_score)


def collate_fn(wav_list):
    wav_list = [elem.transpose(-1, -2) for elem in wav_list] # T x 1
    wav = pad_sequence(wav_list, batch_first=True, padding_value=0) # B x T x 1
    return wav.transpose(-1, -2)


def test_scores_on_dataset_batched(
    orig_utmos,
    torch_utmos,
    script_utmos,
    dataset,
    batch_size=2,
):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                        shuffle=False, drop_last=False)
    final_orig_score = 0
    final_torch_score = 0
    final_script_score = 0
    for wav in tqdm(loader):
        wav = wav.to(device)
        orig_score = orig_utmos.score(wav)
        torch_score = torch_utmos.score(wav)
        script_score = script_utmos.score(wav)

        torch_score = torch_score.numpy()
        script_score = script_score.numpy()

        np.testing.assert_allclose(
            torch_score,
            orig_score,
            rtol=1e-6,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            script_score,
            orig_score,
            rtol=1e-6,
            atol=1e-6,
        )

        final_orig_score += orig_score.sum().item()
        final_torch_score += torch_score.sum().item()
        final_script_score += script_score.sum().item()

    final_orig_score = final_orig_score / len(dataset)
    final_torch_score = final_torch_score / len(dataset)
    final_script_score = final_script_score / len(dataset)

    print("Final Results:")
    print("Orig:", final_orig_score)
    print("Torch:", final_torch_score)
    print("Script:", final_script_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare UTMOS variants on LibriSpeech"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to put the models on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to use (default: 1).",
    )
    args = parser.parse_args()

    device = args.device
    orig_utmos = UTMOSScore(device=device)
    torch_utmos = UTMOSScoreTorch(device=device)
    script_utmos = UTMOSScoreScripted(device=device)

    dataset = LibrispeechDataset(part="test-clean")

    batch_size = args.batch_size

    if batch_size == 1:
        test_scores_on_dataset(
            orig_utmos=orig_utmos,
            torch_utmos=torch_utmos,
            script_utmos=script_utmos,
            dataset=dataset
        )
        print("Test passed successfully")
    else:
        test_scores_on_dataset_batched(
            orig_utmos=orig_utmos,
            torch_utmos=torch_utmos,
            script_utmos=script_utmos,
            dataset=dataset,
            batch_size=batch_size,
        )
        print("Batched test passed successfully")