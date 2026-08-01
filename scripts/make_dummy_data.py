"""
Builds a small fake dataset with the layout of ASVspoof2019 LA.

The real one is about 7 GB, too much to keep locally just to debug the
pipeline. These files are noise. They only check that indexing, padding,
cropping and the front-end run at all, before anything goes to Kaggle.

Spoof utterances get an extra high-frequency tone, so the one-batch test
has something it can overfit.
"""

import argparse
from pathlib import Path

import torch
import torchaudio

SAMPLE_RATE = 16000
PARTITIONS = {
    "train": ("LA_T", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev": ("LA_D", "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval": ("LA_E", "ASVspoof2019.LA.cm.eval.trl.txt"),
}
ATTACKS = ["A01", "A02", "A03"]


def make_utterance(is_spoof, generator):
    """
    Create a random waveform. Durations vary between 1 and 7 seconds to
    cover both the padding and the cropping branch of the dataset code.
    """
    n_samples = int(torch.randint(1, 8, (1,), generator=generator).item() * SAMPLE_RATE)
    waveform = torch.randn(1, n_samples, generator=generator) * 0.05

    if is_spoof:
        t = torch.arange(n_samples) / SAMPLE_RATE
        waveform = waveform + 0.05 * torch.sin(2 * torch.pi * 6000 * t)

    return waveform


def make_partition(root, partition, n_utterances, generator):
    prefix, protocol_name = PARTITIONS[partition]

    flac_dir = root / f"ASVspoof2019_LA_{partition}" / "flac"
    flac_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir = root / "ASVspoof2019_LA_cm_protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for i in range(n_utterances):
        is_spoof = i % 5 != 0
        utterance_id = f"{prefix}_{i:07d}"
        speaker_id = f"LA_{i % 4:04d}"

        waveform = make_utterance(is_spoof, generator)
        torchaudio.save(str(flac_dir / f"{utterance_id}.flac"), waveform, SAMPLE_RATE)

        if is_spoof:
            attack = ATTACKS[i % len(ATTACKS)]
            lines.append(f"{speaker_id} {utterance_id} - {attack} spoof")
        else:
            lines.append(f"{speaker_id} {utterance_id} - - bonafide")

    (protocol_dir / protocol_name).write_text("\n".join(lines) + "\n")
    print(f"{partition}: {n_utterances} utterances -> {flac_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/dummy")
    parser.add_argument("--n-utterances", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)

    root = Path(args.output_dir) / "LA" / "LA"
    for partition in PARTITIONS:
        make_partition(root, partition, args.n_utterances, generator)


if __name__ == "__main__":
    main()
