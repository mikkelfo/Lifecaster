import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig, OmegaConf

from lifecaster.datamodules.PretrainDataModule import PretrainDataModule
from lifecaster.lightningmodules.PretrainModule import PretrainModule
from lifecaster.loggers import RetryOrSkipTensorBoardLogger
from lifecaster.paths import FPATH, get_local_drive, get_network_drive, get_wandb_runid

OmegaConf.register_new_resolver("run_id", get_wandb_runid, use_cache=True)
OmegaConf.register_new_resolver("network_drive", get_network_drive)
OmegaConf.register_new_resolver("local_drive", get_local_drive)


@hydra.main(config_path="../configs", config_name="pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"Experiment: {cfg.experiment_name} / {cfg.run_id}")
    seed_everything(73)
    torch.set_float32_matmul_precision(cfg.hardware.float32_matmul_precision)

    data_module = PretrainDataModule(
        output_dir=FPATH.DATA / cfg.paths.output_dir,
        source_dir=FPATH.NETWORK_DATA / cfg.paths.source_dir,
        sources=cfg.data.sources,
        background=cfg.data.background,
        cohorts=cfg.data.cohorts,
        num_workers=cfg.hardware.num_workers,
        n_tokens=cfg.training.n_tokens,
        max_seqlen=cfg.training.max_seqlen,
        pipeline_cfg=cfg.data.pipeline,
    )
    data_module.prepare_data()  # We need this for the vocab and train_dataset

    lightning_module = PretrainModule(
        # Model
        vocab_size=len(data_module.data_pipeline.vocab),
        max_seqlen=cfg.model.max_seqlen,
        hidden_size=cfg.model.hidden_size,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        bias=cfg.model.bias,
        causal=cfg.model.causal,
        # Optimizer
        learning_rate=cfg.training.learning_rate,
        beta1=cfg.training.beta1,
        beta2=cfg.training.beta2,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
    )

    # Trainer setup
    logger = RetryOrSkipTensorBoardLogger(
        save_dir=FPATH.LOGS,
        name=cfg.experiment_name,
        version=cfg.run_id,
    )

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=FPATH.CHECKPOINTS / cfg.experiment_name / cfg.run_id,
        filename="{epoch}",
        monitor="val/loss",
        save_top_k=-1,
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val/loss", min_delta=1e-3, patience=2)

    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[lr_monitor, checkpoint_callback, early_stop],
        logger=logger,
        deterministic=False,
        precision=cfg.hardware.precision,
        fast_dev_run=cfg.training.fast_dev_run,
    )

    # Train
    trainer.fit(
        lightning_module,
        datamodule=data_module,
        ckpt_path=(
            FPATH.CHECKPOINTS / cfg.paths.ckpt_path
            if cfg.paths.ckpt_path is not None
            else None
        ),
    )


if __name__ == "__main__":
    main()
