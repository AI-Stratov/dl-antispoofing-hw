import torch
from tqdm.auto import tqdm

from src.metrics import EER
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eer = EER()

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        else:
            self.eer.update(batch["logits"], batch["labels"])

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Same as the base version plus the EER, which the MetricTracker
        cannot hold: it averages over batches and the EER does not.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on.
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        self.eer.reset()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.epoch_len, part)

            self.writer.add_scalar("epoch", epoch)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(batch_idx, batch, part)

            logs = self.evaluation_metrics.result()
            logs["EER"] = self.eer.result()
            if self.writer is not None:
                self.writer.add_scalar("EER", logs["EER"])

        return logs

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """

        if mode == "train" and batch_idx == 0:
            self.writer.add_image(
                "spectrogram", self._spectrogram_to_image(batch["data_object"][0])
            )

    @staticmethod
    def _spectrogram_to_image(spectrogram):
        """
        Scale a spectrogram to 0-1 for logging as a grayscale image, low
        frequencies at the bottom.

        Args:
            spectrogram (Tensor): spectrogram of shape (1, freq, time).
        Returns:
            image (ndarray): image of shape (freq, time).
        """
        image = spectrogram.detach().cpu().squeeze(0).flipud()
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image.numpy()
