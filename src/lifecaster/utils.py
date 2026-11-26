from collections import Counter
from typing import List, Union

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
from lightning.pytorch.plugins.io import TorchCheckpointIO


class LegacyTorchCheckpointIO(TorchCheckpointIO):
    """Implements torch.load with a `weights_only` keyword"""

    def load_checkpoint(self, path, map_location=None, **kwargs):
        """Implements torch.load with a `weights_only` keyword"""
        return torch.load(path, map_location=map_location, weights_only=False)


def get_pnrs(sources: Union[ds.Dataset, List[ds.Dataset]]) -> pa.Array:
    """Gets unique pnrs from Dataset or List of Dataset"""
    if isinstance(sources, ds.Dataset):
        return pc.unique(sources.to_table(columns=["person_id"])["person_id"])
    if isinstance(sources, list):
        return pc.unique(
            pa.concat_arrays(
                pc.unique(source.to_table(columns=["person_id"])["person_id"])
                for source in sources
            )
        )
    else:
        raise TypeError(
            f"{type(sources)} is not supported, only ds.Dataset and List[ds.Dataset]"
        )


def get_background_length(background: pl.DataFrame):
    """Returns number of background cols (not accounting person_id and date)"""
    bg_cols = set(background.columns)
    return len(bg_cols - set(["person_id", "date_col"]))


def calculate_abspos(
    date_col: pl.Expr, origin_point=pl.datetime(2020, 1, 1, time_unit="ns")
):
    """Calculate absolute position (abspos) given datetime col and origin_point (default 1/1/2020)"""
    # return date_col.dt.epoch(time_unit="s") / (60 * 60)
    return (date_col - origin_point).dt.total_seconds() / 60 / 60  # hours


def create_weights(outcomes: list[int], op=lambda x: x, pprint=True) -> List[float]:
    """
    Creates weights for a outcome list to be used in a WeightedRandomSampler.

    Args:
        outcomes (list[int]): A list of integer outcomes.
        op (Optional: Callable): An optional operation to apply to the denominator of the weights
    Returns:
        list: A list of weights for each outcome.
    """
    # Count the frequency of each outcome
    counter = Counter(outcomes)

    # Calculate weight for each class with optional operation
    weights = {key: 1 / op(count) for key, count in counter.items()}
    if not create_weights.done:
        print("Sampling with weights: ", weights)
        create_weights.done = True

    # Assign weights to each element in the outcomes list
    sample_weights = [weights[outcome] for outcome in outcomes]

    return sample_weights


# Sets the `done` status to False so print only happens once
create_weights.done = False


def auto_ntokens(
    trainer,
    model,
    datamodule,
    init_val=200_000,
    max_trials=4,
    steps_per_trial=50,
):
    """Tunes the n_tokens of datamodule to fit in memory"""
    import lightning.pytorch.tuner.batch_size_scaling as bss
    from lightning.pytorch.tuner.tuning import Tuner
    from lightning.pytorch.utilities.parsing import lightning_getattr, lightning_setattr

    # Monkey-patch for n_tokens > len(dataset)
    def patch_adjust_batch_size(
        trainer,
        batch_arg_name="n_tokens",
        factor=1.0,
        value=None,
        desc=None,
        max_val=None,
    ):

        model = trainer.lightning_module
        batch_size = lightning_getattr(model, batch_arg_name)
        new_size = value if value is not None else int(batch_size * factor)

        if desc:
            print(f"Batch size {batch_size} {desc}, trying batch size {new_size}")
        changed = new_size != batch_size
        lightning_setattr(model, batch_arg_name, new_size)
        return new_size, changed

    bss._adjust_batch_size = patch_adjust_batch_size

    tuner = Tuner(trainer)
    final_tokens = tuner.scale_batch_size(
        model,
        datamodule=datamodule,
        mode="binsearch",
        batch_arg_name="n_tokens",
        init_val=init_val,
        steps_per_trial=steps_per_trial,
        max_trials=max_trials,
        # margin=0.2,
        # max_val=10_000_000,
    )
    datamodule.n_tokens = 0.8 * final_tokens
    return final_tokens


def get_n_steps_per_epoch(datamodule):
    """Gets number of steps per epoch"""
    # Adjust n_steps and warmup
    sampler = datamodule.get_sampler(datamodule.train_dataset)
    avg_batch_size = datamodule.n_tokens / (sum(sampler.lengths) / len(sampler.lengths))
    n_steps = int(len(datamodule.train_dataset) / avg_batch_size)
    print(f"Info: Steps ~{int(n_steps)} per epoch")
    return n_steps
