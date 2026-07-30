# Voice Anti-spoofing with LCNN

Homework project for the Deep Learning summer mini-course at the HSE CS
Faculty. The task is a countermeasure (CM) system that decides whether an
utterance is bonafide speech or a spoofing attack, trained and evaluated on
the Logical Access partition of the
[ASVspoof 2019](https://www.asvspoof.org/) dataset.

The model is a Light CNN. The architecture follows Table 1 of the
[STC anti-spoofing paper](https://arxiv.org/abs/1904.05576), the
Max-Feature-Map activation comes from the original
[LightCNN paper](https://arxiv.org/abs/1511.02683), and the training recipe
and the data preparation follow the
[comparative study of Wang and Yamagishi](https://arxiv.org/abs/2103.11326).

The project is built on the
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

The final model reaches an equal error rate of **4.56%** on the evaluation
partition. Training logs for all runs are in the
[WandB report](https://api.wandb.ai/links/aistratov/8na4ethh).

## Installation

1. Create and activate an environment, for example with `conda`:

   ```bash
   conda create -n antispoofing python=3.11
   conda activate antispoofing
   ```

2. Install the requirements, either way:

   ```bash
   pip install -r requirements.txt
   # or, with uv
   uv sync
   ```

   `pyproject.toml` and `requirements.txt` list the same packages. The torch
   ones are deliberately unpinned: Kaggle and Colab already ship a CUDA
   build, and pinning a version makes the installer replace the whole stack.

3. Reading `flac` files needs a backend for `torchaudio`. `soundfile` is in
   the dependencies and is enough. On Kaggle it is already available.

## Data

Download the LA partition of ASVspoof 2019 from
[Kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset).
Working inside Kaggle and attaching the dataset as an input is faster and
does not use up the disk quota.

The code needs a directory that holds the three partitions and the protocol
files, and it is passed as `data_root`. The archive unpacks with a doubled
`LA` directory, so the path usually looks like
`.../asvpoof-2019-dataset/LA/LA`, with this structure inside:

```
LA/LA
├── ASVspoof2019_LA_train/flac/LA_T_*.flac
├── ASVspoof2019_LA_dev/flac/LA_D_*.flac
├── ASVspoof2019_LA_eval/flac/LA_E_*.flac
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

Check the layout before starting a long run, the protocol file names in
particular:

```bash
ls DATA_ROOT
ls DATA_ROOT/ASVspoof2019_LA_cm_protocols
```

If the names differ, override the paths in
`src/configs/datasets/asvspoof.yaml` instead of moving files around.

There is also a script that builds a small fake dataset with the same
layout. It is only useful for checking that the code runs, the files it
writes are noise:

```bash
python3 scripts/make_dummy_data.py
```

## How to use

Train the model:

```bash
python3 train.py -cn=lcnn data_root=DATA_ROOT
```

Before a long run it is worth doing the one-batch test. It trains on a
single fixed batch of eight utterances and evaluates on the same batch, so
the loss should drop close to zero and the EER should reach zero. It does
not prove that the pipeline is correct, but if it fails, something is
definitely wrong:

```bash
python3 train.py -cn=onebatchtest data_root=DATA_ROOT
```

Run the trained model on the evaluation partition:

```bash
python3 inference.py data_root=DATA_ROOT \
    inferencer.from_pretrained=saved/lcnn_fft/checkpoint-epoch20.pth
```

That path is already the default in `src/configs/inference.yaml`, so the
override is only needed for a different checkpoint. Note that there is no
`model_best.pth` to load: training runs with `monitor: "off"`, because the
development EER stops ordering the checkpoints once it reaches zero. The
report explains the choice.

This prints the EER and writes the scores to
`data/saved/asvspoof/test/avistratov.csv`, one line per trial in the
`utterance_id,score` format that the grading script of the course expects.
The file name comes from `inferencer.predictions_name` and has to be the
university username, otherwise the submission is not accepted.

## Running on Kaggle

The dataset is attached as an input and the code is cloned from GitHub:

```bash
!git clone https://github.com/USERNAME/REPO_ID
%cd REPO_ID
!pip install -q hydra-core torchmetrics soundfile

!python3 train.py -cn=lcnn \
    data_root=/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA \
    writer.run_name=lcnn_fft
```

Pick the T4 accelerator and not the P100. The torch that Kaggle ships is
built for compute capability 7.0 and above, while the P100 is a 6.0 card,
so every kernel launch fails with `no kernel image is available for
execution on the device`. The T4 is 7.5 and works.

Only the packages that Kaggle does not ship are installed above. Running
`pip install -r requirements.txt` is a bad idea there: torchaudio from PyPI
depends on one exact version of torch and can silently replace the build
that matches the GPU of the session.

`wandb` asks for an API key on the first run. Add it through the Kaggle
secrets rather than pasting it into a cell.

## What was changed compared to the template

- `src/datasets/asvspoof.py`: the dataset, the index is built from a CM
  protocol file.
- `src/transforms/spectrogram.py`: the log power spectrum front-end. It is
  applied to the batch, and therefore on the GPU, so the spectrograms do not
  have to be precomputed and stored.
- `src/model/lcnn.py`: the Max-Feature-Map activation and the LCNN.
- `src/metrics/eer.py`: the EER. The functions that compute the detection
  curve are the ones provided with the homework.
- `src/trainer/trainer.py`: the evaluation epoch also reports the EER. It is
  not an ordinary metric of the template, because the EER of a partition is
  not the average of the EERs of its batches: its threshold is chosen over
  the whole partition at once.
- `src/loss/cross_entropy.py`: the loss.

## Credits

The project template is by [Petr Grinberg](https://github.com/Blinorot).
`compute_det_curve` and `compute_eer` are taken from the homework repository
of the course.
