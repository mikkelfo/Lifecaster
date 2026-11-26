import numpy as np


class Serializer:
    """Custom byte serialization that is 10x faster than msgpack"""

    FIELDS = [("event", np.int32), ("age", np.float32), ("abspos", np.float32)]

    @classmethod
    def encode_person(cls, person: dict) -> bytes:
        """Custom numpy byte encoding (10x faster than msgpack)"""
        parts = []
        for key, dtype in cls.FIELDS:
            flat = np.array([x for seq in person[key] for x in seq], dtype=dtype)
            offsets = np.cumsum([0] + [len(s) for s in person[key]], dtype=np.int32)
            header = np.array([len(offsets) - 1], dtype=np.int32)
            parts.extend([header.tobytes(), offsets.tobytes(), flat.tobytes()])
        return b"".join(parts)

    @classmethod
    def decode_person(cls, buf: bytes) -> dict:
        """Custom numpy byte decoding (10x faster than msgpack)"""
        result, pos = {}, 0
        for key, dtype in cls.FIELDS:
            n_seqs = int(np.frombuffer(buf, dtype=np.int32, count=1, offset=pos)[0])
            pos += 4
            offsets = np.frombuffer(
                buf, dtype=np.int32, count=n_seqs + 1, offset=pos
            ).tolist()
            pos += (n_seqs + 1) * 4
            flat = np.frombuffer(
                buf, dtype=dtype, count=offsets[-1], offset=pos
            ).tolist()
            pos += offsets[-1] * np.dtype(dtype).itemsize

            result[key] = [flat[a:b] for a, b in zip(offsets, offsets[1:])]

        return result

    @classmethod
    def encode_lengths(cls, lengths: list) -> bytes:
        """Custom encoding for event lengths using int8"""
        return cls.encode(lengths, dtype=np.int8)  # int8 since values < 127

    @classmethod
    def decode_lengths(cls, buf: bytes) -> list:
        """Custmo decoding for event lengths using int8"""
        return cls.decode(buf, dtype=np.int8)

    @staticmethod
    def encode(lst: list, dtype=np.int32) -> bytes:
        """Standard np.array(lst, dtype=dtype).tobytes()"""
        return np.array(lst, dtype=dtype).tobytes()

    @staticmethod
    def decode(buf, dtype=np.int32) -> list:
        """Standard np.frombuffer(buf, dtype=dtype).tolist()"""
        return np.frombuffer(buf, dtype=dtype).tolist()
