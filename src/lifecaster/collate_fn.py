"""Implements all collate functionality"""

from typing import Dict, List, Literal, Tuple

import torch
from lifecaster.collate_utils import calc_age_from_abspos, find_censor_idx


class Collate:
    """Implements the core collate functions"""

    def __init__(self, max_seqlen, background_length, padding_token_id=0):
        self.max_seqlen = max_seqlen
        self.truncate_length = max_seqlen - background_length
        self.background_length = background_length
        self.background_present = int(background_length > 0)
        self.padding_token_id = padding_token_id
        self.data_keys = ["event", "age", "abspos", "segment"]

    def __call__(self, batch: Dict) -> dict:
        """Main function that includes all processing"""
        batch = self.pre_main(batch)

        output = self.main(batch)
        output = self.post_main(output, batch)
        return output

    def pre_main(self, batch):
        """Pre-processes batch before going into main"""
        return batch

    def post_main(self, output, batch):
        """Post-processes output"""
        output["attn_mask"] = output["event"] != self.padding_token_id
        return output

    @staticmethod
    def gather_person(batch: Dict, idx: int):
        """Gathers a Dict[key, val[idx]]"""
        return {key: val[idx] for key, val in batch.items()}

    def main(self, batch: Dict[str, List[dict]]) -> dict:
        """The main processing function"""
        output = {key: [] for key in self.data_keys}

        for i in range(len(batch["data"])):
            person = self.gather_person(batch, i)
            indv = self.process_person(person)
            for key, v in indv.items():
                output[key].append(v)

        for key in self.data_keys:
            output[key] = self.pad(output[key], padding_value=self.padding_token_id)

        return output

    def process_person(self, person):
        """Handles all person-specific processsing"""
        output = {}

        trunc_idx = self.find_truncation_idx(
            person["event_lengths"], truncate_length=self.truncate_length
        )

        trunc_event_lengths = self.truncate(person["event_lengths"], trunc_idx)

        for key in self.data_keys:
            if key == "segment":
                continue  # Segment is created after
            trunc_seq = self.truncate(person["data"][key], trunc_idx)
            seq = self.flatten(trunc_seq)

            output[key] = seq
        output["segment"] = self.flatten(
            [
                [sid for _ in range(n_tokens)]
                for sid, n_tokens in enumerate(trunc_event_lengths)
            ]
        )
        return output

    @staticmethod
    def find_truncation_idx(event_lengths: List[int], truncate_length: int) -> int:
        """Find the trucation index"""
        total = 0
        i = len(event_lengths)
        for count in reversed(event_lengths):
            if total + count > truncate_length:
                break
            total += count
            i -= 1
        return i

    def truncate(self, seq: list, idx: int) -> list:
        """Truncates `seq` based on `idx`, keeping background if present"""
        if idx == 0:
            return seq
        return seq[: self.background_present] + seq[idx:]

    @staticmethod
    def flatten(seq: List[list]) -> list:
        """Flattens `seq`"""
        return [e for sublist in seq for e in sublist]

    @staticmethod
    def pad(sequence: list, dtype: torch.dtype = None, **pad_kwargs) -> torch.Tensor:
        """Pads the sequence (using pad_kwargs) converts to tensor"""
        # Convert to tensors and pad with dtype conversion
        sequences = [torch.as_tensor(seq, dtype=dtype) for seq in sequence]
        return torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            **pad_kwargs,
        ).to(dtype)


class AutoregressiveCollate(Collate):
    """Standard collate with Autoregressive shift"""

    def post_main(self, output, batch):
        output = self.adapt_to_AR(output)
        return super().post_main(output, batch)

    def adapt_to_AR(self, output):
        """Defines targets (next token) and removes last token in data keys"""
        output["target"] = output["event"][:, 1:]
        output["target"] = output["target"].masked_fill(output["target"] == 0, -100)
        for feature in self.data_keys:
            output[feature] = output[feature][:, :-1]
        return output


class OutcomeCollate(Collate):
    """Standard collate with added outcome_info"""

    def post_main(self, output, batch):
        output = self._add_outcome_info(output, batch["outcome_info"])
        return super().post_main(output, batch)

    def _add_outcome_info(self, output: dict, outcome_info: List[dict]) -> Dict:
        output["person_id"] = [out["person_id"] for out in outcome_info]
        output["outcome"] = torch.tensor([out["outcome"] for out in outcome_info])
        return output


class CensorCollate(OutcomeCollate):
    """OutcomeCollate with censoring"""

    def pre_main(self, batch):
        batch = self._censor_data(batch)
        return super().pre_main(batch)

    def _censor_data(self, batch):
        for i in range(len(batch["data"])):
            censor_idx = find_censor_idx(
                abspos=batch["data"][i]["abspos"],
                censor=batch["outcome_info"][i]["censor"],
            )
            batch["data"][i] = {
                key: value[:censor_idx] for key, value in batch["data"][i].items()
            }
            batch["event_lengths"][i] = batch["event_lengths"][i][:censor_idx]
        return batch


class PredictCollate(OutcomeCollate):
    """Standard collate with PREDICT tokens"""

    def __init__(
        self,
        max_seqlen,
        background_length,
        padding_token_id=0,
        # Finetuning specific
        prediction_windows: list = [1000],
        predict_token_id: int = 1,
    ):
        super().__init__(
            max_seqlen=max_seqlen,
            background_length=background_length,
            padding_token_id=padding_token_id,
        )
        self.predict_token_id = predict_token_id
        self.prediction_windows = torch.as_tensor(prediction_windows)

    def process_person(self, person):
        person = self.add_predict_tokens_to_person(person)
        output = super().process_person(person)
        output = self.sort_by_abspos(output)
        return output

    def post_main(self, output, batch=None):
        output["predict_mask"] = output["event"] == self.predict_token_id
        return super().post_main(output, batch)

    def add_predict_tokens_to_person(self, person):
        """Adds a [PREDICT] token (self.predict_token_id) to the person for every prediction"""
        prediction_abspos = person["outcome_info"]["predict"]
        person["data"] = self.add_predict_data_to_person(
            person["data"], prediction_abspos
        )
        person["event_lengths"].extend([1] * len(prediction_abspos))
        return person

    def add_predict_data_to_person(self, person, predictions):
        """Adds predict-related features to person"""
        person["event"].extend([[self.predict_token_id] for _ in predictions])
        person["abspos"].append(predictions)
        person["age"].append(
            [calc_age_from_abspos(person["abspos"][0][0], a) for a in predictions]
        )
        return person

    @staticmethod
    def sort_by_abspos(output: Dict[str, list]):
        """Sort all keys in output by the `abspos`"""
        sorted_idxs = sorted(
            range(len(output["abspos"])), key=lambda i: output["abspos"][i]
        )

        return {key: [value[i] for i in sorted_idxs] for key, value in output.items()}

    def _add_outcome_info(self, output: dict, outcome_info: List[dict]) -> Dict:
        output["target"] = self._create_targets(outcome_info)
        return super()._add_outcome_info(output, outcome_info)

    def _create_targets(self, outcome_info: Tuple[Dict]) -> torch.Tensor:
        """Creates targets (k=prediction window) per predict abspos"""
        targets = []
        for outcome in outcome_info:
            if outcome["outcome"] is None:
                outcome["outcome"] = torch.nan
            predict_abspos = torch.as_tensor(sorted(outcome["predict"]))
            target = (
                predict_abspos.unsqueeze(-1) + self.prediction_windows
            ) >= outcome["outcome"]
            targets.append(target)
        return self.pad(
            targets,
            dtype=torch.float32,
            padding_value=-100,
        )


class AutoregressiveCensorCollate(CensorCollate, AutoregressiveCollate):
    """Autoregressive collate with censoring"""


class PredictCensorCollate(CensorCollate, PredictCollate):
    """Combines Censor with Predict
    This is the standard finetuning collate
    """


class TargetPredictCensorCollate(PredictCensorCollate):
    """For pre-defined targets"""

    def _create_targets(self, outcome_info):
        return self.pad(
            [torch.tensor(outcome["target"]) for outcome in outcome_info],
            dtype=torch.float32,
            padding_value=-100,
        )
