import torch
import torch.nn as nn

from lifecaster.model.components.time2vec import Time2Vec


class LifecasterEmbeddings(nn.Module):
    """Implements the standard event, age, abspos, and segment embeddings.

    Event and segment uses nn.Embedding while temporal (age and abspos) uses Time2Vec
    """

    def __init__(
        self,
        vocab_size: int,
        max_seqlen: int,
        hidden_size: int,
        # Set to defaults
        clip_min=-100,
        clip_max=100,
        init_scale=1e-4,
        padding_idx=0,
    ):
        super().__init__()
        self.event_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=padding_idx
        )
        self.t2v_age = Time2Vec(
            output_dim=hidden_size,
            clip_min=clip_min,
            clip_max=clip_max,
            # init_scale=init_scale,
        )
        self.t2v_abspos = Time2Vec(
            output_dim=hidden_size,
            clip_min=clip_min,
            clip_max=clip_max,
            init_scale=init_scale,
        )
        self.segment_embeddings = nn.Embedding(
            max_seqlen, hidden_size, padding_idx=padding_idx
        )

    def forward(
        self,
        event: torch.Tensor,
        age: torch.Tensor,
        abspos: torch.Tensor,
        segment: torch.Tensor,
    ):
        """Embeds information from `event`, `age`, `abspos`, and `segment`"""
        x = self.event_embeddings(event)

        x += self.t2v_age(age)
        x += self.t2v_abspos(abspos)

        x += self.segment_embeddings(segment)

        return x
