# Voice Anti-spoofing with LCNN

A countermeasure system that tells bonafide speech from spoofing attacks,
trained on the Logical Access partition of
[ASVspoof 2019](https://www.asvspoof.org/). Homework for the Deep Learning
summer mini-course at HSE CS.

The model is a Light CNN: architecture from Table 1 of the
[STC paper](https://arxiv.org/abs/1904.05576), Max-Feature-Map activation
from [LightCNN](https://arxiv.org/abs/1511.02683), training recipe and data
preparation from [Wang and Yamagishi](https://arxiv.org/abs/2103.11326).

**4.56% EER** on the evaluation partition. All runs are in the
[WandB report](https://api.wandb.ai/links/aistratov/8na4ethh).

## Installation

```bash
conda create -n antispoofing python=3.11 && conda activate antispoofing
pip install -r requirements.txt  # or: uv sync
```

`pyproject.toml` and `requirements.txt` list the same packages. The torch
ones are unpinned on purpose: Kaggle and Colab already ship a CUDA build,
and pinning makes the installer replace the whole stack.

## Data

Download the LA partition from
[Kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset). The
archive unpacks with a doubled `LA` directory, so `data_root` usually ends
in `.../asvpoof-2019-dataset/LA/LA`:

```
LA/LA
├── ASVspoof2019_LA_train/flac/LA_T_*.flac
├── ASVspoof2019_LA_dev/flac/LA_D_*.flac
├── ASVspoof2019_LA_eval/flac/LA_E_*.flac
└── ASVspoof2019_LA_cm_protocols/*.txt
```

If the protocol file names differ, override the paths in
`src/configs/datasets/asvspoof.yaml`.

`scripts/make_dummy_data.py` writes the same layout filled with noise, for
checking that the code runs.

## Usage

```bash
python3 train.py -cn=lcnn data_root=DATA_ROOT
python3 train.py -cn=onebatchtest data_root=DATA_ROOT   # overfit 8 utterances
python3 inference.py data_root=DATA_ROOT
```

Inference defaults to `saved/lcnn_fft/checkpoint-epoch20.pth`, override with
`inferencer.from_pretrained=`. There is no `model_best.pth`: training runs
with `monitor: "off"` because the development EER reaches zero and stops
ordering checkpoints. The report explains why.

Scores go to `data/saved/asvspoof/test/avistratov.csv` as
`utterance_id,score`, the format the course grading script expects. The name
comes from `inferencer.predictions_name` and has to be the university
username.

## Running on Kaggle

```bash
!git clone https://github.com/USERNAME/REPO_ID
%cd REPO_ID
!pip install -q hydra-core torchmetrics soundfile

!python3 train.py -cn=lcnn \
    data_root=/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA \
    writer.run_name=lcnn_fft
```

Only the packages Kaggle does not ship. `pip install -r requirements.txt`
there pulls a torchaudio that can replace the CUDA build of the session.

Pick T4, not P100: Kaggle's torch is built for compute capability 7.0 and
above, and every kernel launch on the 6.0 card fails with `no kernel image
is available for execution on the device`.

Put the `wandb` key in Kaggle secrets rather than in a cell.

## Changes to the template

- `src/datasets/asvspoof.py` - dataset, index built from a CM protocol file.
- `src/transforms/spectrogram.py` - log power spectrum front-end, applied to
  the batch on the GPU, so nothing has to be precomputed.
- `src/model/lcnn.py` - Max-Feature-Map and the LCNN.
- `src/metrics/eer.py` - EER.
- `src/trainer/trainer.py` - the evaluation epoch reports the EER. It is not
  an ordinary template metric: the EER of a partition is not the average of
  its batch EERs, the threshold is chosen over the whole partition at once.
- `src/loss/cross_entropy.py` - loss.

## Credits

Built on the [PyTorch Project
Template](https://github.com/Blinorot/pytorch_project_template) by
[Petr Grinberg](https://github.com/Blinorot), MIT. `compute_det_curve` and
`compute_eer` come from the course homework repository.
