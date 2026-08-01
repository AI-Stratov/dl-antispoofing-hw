import logging
from collections import Counter

import torch
import torchaudio

from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


BONAFIDE = 1
SPOOF = 0


class ASVspoof2019LADataset(BaseDataset):
    """
    Logical Access partition of ASVspoof 2019.

    The index comes from a CM protocol file. Every line has five fields:

        SPEAKER_ID UTTERANCE_ID - ATTACK_ID LABEL

    ATTACK_ID is a dash for bonafide trials, LABEL is bonafide or spoof.

    The length is fixed here rather than in a transform: the front-end runs
    on an already collated batch and so needs equal lengths before it.
    """

    def __init__(
        self,
        protocol_path,
        audio_dir,
        max_length,
        random_crop=False,
        pad_mode="zero",
        max_per_class=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            protocol_path (str): path to the CM protocol file.
            audio_dir (str): directory with the flac files of this partition.
            max_length (int): length of the waveform in samples that the
                dataset returns.
            random_crop (bool): crop a longer utterance at a random position
                instead of taking its beginning. False for evaluation, so
                that the metrics do not depend on the seed.
            pad_mode (str): how a short utterance is extended, 'zero' or
                'repeat'. Zeros follow the recipe; repetition costs about
                eight points of evaluation EER, see the report.
            max_per_class (int | None): keep at most this many utterances of
                each class. The one-batch test needs it: the training
                protocol is only a tenth bonafide, so a small slice taken
                without looking at the labels can hold a single class.
        """
        assert pad_mode in ("zero", "repeat"), f"unknown pad_mode {pad_mode}"

        self.max_length = max_length
        self.random_crop = random_crop
        self.pad_mode = pad_mode

        index = self._create_index(protocol_path, audio_dir)
        if max_per_class is not None:
            index = self._limit_per_class(index, max_per_class)
        super().__init__(index, *args, **kwargs)

        self.class_counts = Counter(entry["label"] for entry in self._index)
        logger.info(
            "%s: %d bonafide, %d spoof",
            protocol_path.split("/")[-1],
            self.class_counts[BONAFIDE],
            self.class_counts[SPOOF],
        )

    def _create_index(self, protocol_path, audio_dir):
        index = []
        with open(protocol_path) as protocol:
            for line in protocol:
                fields = line.strip().split()
                if not fields:
                    continue
                utterance_id, label = fields[1], fields[4]
                index.append(
                    {
                        "path": f"{audio_dir}/{utterance_id}.flac",
                        "label": BONAFIDE if label == "bonafide" else SPOOF,
                        "key": utterance_id,
                    }
                )
        return index

    @staticmethod
    def _limit_per_class(index, max_per_class):
        """
        Keep the first max_per_class entries of every class, in protocol
        order, so that the subset is the same on every run.
        """
        counts = {}
        kept = []
        for entry in index:
            label = entry["label"]
            if counts.get(label, 0) < max_per_class:
                kept.append(entry)
                counts[label] = counts.get(label, 0) + 1
        return kept

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        waveform = self.load_object(data_dict["path"])
        waveform = self.adjust_length(waveform)

        instance_data = {
            "data_object": waveform,
            "labels": data_dict["label"],
            "keys": data_dict["key"],
        }
        return self.preprocess_data(instance_data)

    def load_object(self, path):
        waveform, _ = torchaudio.load(path)

        return waveform.squeeze(0)

    def adjust_length(self, waveform):
        """
        Pad or crop a waveform to self.max_length samples.
        """
        length = waveform.shape[0]

        if length < self.max_length:
            if self.pad_mode == "repeat":
                n_repeats = self.max_length // length + 1
                waveform = waveform.repeat(n_repeats)
            else:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.max_length - length)
                )

        if waveform.shape[0] > self.max_length:
            if self.random_crop:
                start = torch.randint(
                    0, waveform.shape[0] - self.max_length + 1, (1,)
                ).item()
            else:
                start = 0
            waveform = waveform[start : start + self.max_length]

        return waveform
