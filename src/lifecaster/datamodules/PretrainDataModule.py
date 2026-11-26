from lifecaster.collate_fn import AutoregressiveCollate
from lifecaster.datamodules.BaseDataModule import BaseDataModule


class PretrainDataModule(BaseDataModule):
    """Lightning Data Module for AR pretraining"""

    def unpad(self, batch):
        batch = super().unpad(batch)
        batch["target"] = batch["target"].flatten()[batch["indices"]]
        return batch

    def collate_fn(self):
        return AutoregressiveCollate(
            max_seqlen=self.max_seqlen,
            background_length=self.background_length,
        )
