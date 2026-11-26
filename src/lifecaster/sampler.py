import torch
import torch.distributed as dist


class UnpadSampler:
    """Sampler to gather people based on seq_lens.

    With shard=True under DDP, every rank builds the same batches (seeded by
    seed + epoch) and keeps every world_size'th one, cut to an equal count so
    no rank waits on another at the end of the epoch.
    """

    def __init__(
        self, lengths, n_tokens, max_seqlen, sampler=None, shard=False, seed=0
    ):
        self.lengths = lengths
        self.n_tokens = n_tokens
        self.max_seqlen = max_seqlen
        self._sampler = sampler  # None = shuffle all, or e.g. a WeightedRandomSampler
        self.shard = shard
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        """Called by Lightning each epoch, so the order changes between epochs"""
        self.epoch = epoch

    def _indices(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)  # same on every rank
        if self._sampler is None:
            return torch.randperm(len(self.lengths), generator=g).tolist()
        self._sampler.generator = g
        return list(self._sampler)

    def _batches(self, indices):
        current_batch = []
        n = 0

        for idx in indices:
            length = int(min(self.lengths[idx], self.max_seqlen))

            n += length
            if n > self.n_tokens:
                if current_batch:
                    yield current_batch
                n = length
                current_batch = []
            current_batch.append(idx)

        if current_batch:
            yield current_batch

    def __iter__(self):
        batches = list(self._batches(self._indices()))
        if self.shard and dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
            batches = batches[rank::world][: len(batches) // world]
        yield from batches
