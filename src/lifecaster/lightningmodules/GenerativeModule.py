import torch
import torch.nn.functional as F
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import flash_attn_with_kvcache

from lifecaster.lightningmodules.PretrainModule import PretrainModule


class GenerativeNanoEncoder(PretrainModule):
    """KV-cached autoregressive generation on top of PretrainNanoEncoder.

    `generate()` runs a single prefill pass to populate per-layer K/V caches,
    then decodes one token at a time through flash_attn_with_kvcache. Sampling
    is raw multinomial with a temperature; no top-k / top-p.
    """

    def __init__(self, *args, max_generation_length: int = 1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_generation_length = max_generation_length

    def prepare_for_inference(
        self, device=None, dtype="auto", warmup: bool = True, pad_token_id: int = 0
    ):
        """Cast params + buffers to a single (device, dtype). Patch rotary caches.

        The KV-cache path requires consistent dtype across cached K/V, the
        per-step query, and the rotary cos/sin tables, so this step is not
        optional before calling generate().
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        if isinstance(dtype, torch.dtype):
            target_dtype = dtype
        elif isinstance(dtype, str):
            dmap = {
                "fp16": torch.float16,
                "float16": torch.float16,
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp32": torch.float32,
                "float32": torch.float32,
                "auto": None,
            }
            target_dtype = dmap.get(dtype, None)
        else:
            target_dtype = None

        if target_dtype is None:
            if (
                device.type == "cuda"
                and torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
            ):
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16

        self.to(device=device, dtype=target_dtype)

        # Rotary modules may have already cached cos/sin in their old dtype.
        for m in self.modules():
            rot = getattr(getattr(m, "attn", None), "rotary", None) or getattr(
                m, "rotary", None
            )
            if rot is not None:
                for name in (
                    "_cos_cached",
                    "_sin_cached",
                    "cos_cached",
                    "sin_cached",
                    "_freqs_cached",
                    "freqs_cached",
                    "inv_freq",
                ):
                    buf = getattr(rot, name, None)
                    if isinstance(buf, torch.Tensor):
                        setattr(rot, name, buf.to(device=device, dtype=target_dtype))

        # Tiny forward so rotary cos/sin caches materialize in the new dtype.
        if warmup:
            with torch.inference_mode():
                evt = torch.zeros(1, 1, dtype=torch.long, device=device)
                msk = torch.ones(1, 1, dtype=torch.bool, device=device)
                _, idx, cu, max_s = unpad_input(evt.unsqueeze(-1), msk)[:4]
                meta = {
                    "attn_mask": msk,
                    "indices": idx,
                    "cu_seqlens": cu,
                    "max_seqlen_in_batch": max_s,
                    "total": int(cu[-1].item()),
                }
                x = self.embedding(evt.flatten()[idx])
                for block in self.transformer.h:
                    x = block(x, meta)

        return self

    @torch.no_grad()
    def generate(
        self,
        batch,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        pad_token_id: int = 0,
        eos_token_id=None,
        n_samples_per_person: int = 1,
        start_after_n_tokens: int | None = None,
        return_attention_mask: bool = True,
    ):
        """Autoregressive generation with a per-layer KV cache.

        Returns a dict with keys: prompt, generated, full, generation_lengths,
        finished, prompt_length, prompt_length_per_row (+ generated_attn_mask
        if requested).
        """

        device = batch["event"].device

        # 1. Batch tiling & prompt setup.
        if "attn_mask" not in batch or batch["attn_mask"].shape != batch["event"].shape:
            batch["attn_mask"] = batch["event"] != pad_token_id
        else:
            batch["attn_mask"] = batch["attn_mask"].to(torch.bool)

        if batch["event"].ndim == 1:
            batch["event"] = batch["event"].unsqueeze(0)
            batch["attn_mask"] = batch["attn_mask"].unsqueeze(0)

        B, T = batch["event"].shape

        if n_samples_per_person and n_samples_per_person > 1:

            def _tile(x):
                return (
                    x.repeat_interleave(n_samples_per_person, dim=0)
                    if torch.is_tensor(x) and x.dim() > 0 and x.size(0) == B
                    else x
                )

            batch = {k: _tile(v) for k, v in batch.items()}
            B = B * n_samples_per_person

        actual_len = batch["attn_mask"].sum(1).long()
        if start_after_n_tokens is None:
            prompt_len_vec = actual_len
        else:
            limit = min(int(start_after_n_tokens), T)
            prompt_len_vec = torch.minimum(
                actual_len, torch.full_like(actual_len, limit)
            )

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        keep = pos < prompt_len_vec.unsqueeze(1)
        batch["attn_mask"] = keep
        batch["event"][~keep] = pad_token_id

        # 2. Prefill: run the prompt through the full stack and populate KV caches.
        L = int(prompt_len_vec.max().item())
        max_total_len = min(T, L + max_new_tokens)

        event_trim = batch["event"][:, :L]
        mask_trim = batch["attn_mask"][:, :L]

        ret = unpad_input(event_trim.unsqueeze(-1), mask_trim)
        _, indices_L, cu_seqlens, max_seqlen_in_batch = ret[:4]
        indices_L = indices_L.to(torch.long)

        unpacked = {
            "event": event_trim.flatten()[indices_L],
            "attn_mask": mask_trim,
            "indices": indices_L,
            "max_seqlen_in_batch": max_seqlen_in_batch,
            "cu_seqlens": cu_seqlens,
            "total": int(cu_seqlens[-1].item()),
        }

        x = self.embed_information(unpacked)
        x = self.transformer.drop(x)

        head_dim = self.transformer.h[0].attn.head_dim
        n_heads = self.transformer.h[0].attn.num_heads
        hidden_size = self.hparams.hidden_size

        # RoPE cos/sin for the full target length (prompt + new tokens). All
        # layers share head_dim and inv_freq, so build once and reuse.
        rot0 = self.transformer.h[0].attn.rotary
        rotary_dim = int(getattr(rot0, "rotary_dim", head_dim))
        rotary_interleaved = bool(getattr(rot0, "interleaved", True))
        half = rotary_dim // 2
        inv = getattr(rot0, "inv_freq", None)
        if inv is None:
            inv = 1.0 / (
                10000.0
                ** (torch.arange(0, half, device=x.device, dtype=x.dtype) / half)
            )
        else:
            inv = inv.to(device=x.device, dtype=x.dtype)
        freqs = torch.arange(max_total_len, device=x.device, dtype=x.dtype).unsqueeze(
            1
        ) * inv.unsqueeze(0)
        rotary_cos, rotary_sin = torch.cos(freqs), torch.sin(freqs)

        caches = []
        for block in self.transformer.h:
            # Prefill: standard varlen attention path, identical to forward().
            x_norm = block.ln_1(x)
            q, k, v = block.attn.Wqkv(x_norm).chunk(3, dim=-1)
            q = q.view(-1, n_heads, head_dim)
            k = k.view(-1, n_heads, head_dim)
            v = v.view(-1, n_heads, head_dim)

            q, k = block.attn.rotary(q, k, cu_seqlens, max_seqlen_in_batch)

            qkv = torch.stack((q, k, v), dim=1)
            y = block.attn.self_attn(
                qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch, causal=True
            )
            y = y.view(-1, hidden_size)
            y = block.attn.resid_dropout(block.attn.out_proj(y))
            x = x + y
            x = x + block.mlp(block.ln_2(x))

            # Allocate (B, max_total_len, n_heads, head_dim) and write prompt K/V at [0, L).
            k_cache = k.new_zeros(B, max_total_len, n_heads, head_dim)
            k_cache[:, :L] = pad_input(k, indices_L, B, L)
            v_cache = v.new_zeros(B, max_total_len, n_heads, head_dim)
            v_cache[:, :L] = pad_input(v, indices_L, B, L)

            caches.append(
                {
                    "k": k_cache,
                    "v": v_cache,
                    "seqlens": prompt_len_vec.clone().int(),
                    "ln_1": block.ln_1,
                    "ln_2": block.ln_2,
                    "mlp": block.mlp,
                    "Wqkv": block.attn.Wqkv,
                    "out_proj": block.attn.out_proj,
                    "drop": block.attn.resid_dropout,
                    "softmax_scale": getattr(block.attn, "softmax_scale", None),
                }
            )

        x = self.transformer.ln_f(x)
        last_idx_prompt = cu_seqlens[1:] - 1
        last_hidden_current = x[last_idx_prompt]

        # 3. Decode loop: one new token per step through flash_attn_with_kvcache.
        write_pos = prompt_len_vec.clone()
        finished = torch.zeros(B, dtype=torch.bool, device=x.device)
        gen_len_vec = torch.zeros(B, dtype=torch.long, device=x.device)

        prev_rows_hash = None
        rows_int32 = None

        for _ in range(max_new_tokens):
            can_write = (~finished) & (write_pos < T)
            rows = torch.nonzero(can_write, as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                break

            # int32 row index for cache_batch_idx — only rebuild when active rows change.
            rows_hash = rows.sum().item()
            if rows_hash != prev_rows_hash:
                rows_int32 = rows.to(torch.int32)
                prev_rows_hash = rows_hash

            n_active = rows.numel()

            # Sample next token: raw multinomial (temperature<=0 falls back to argmax).
            logits = self.decoder(last_hidden_current[rows]).float()
            if temperature is None or temperature <= 0:
                next_tokens = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Bookkeeping.
            wp = write_pos[rows]
            batch["event"][rows, wp] = next_tokens
            batch["attn_mask"][rows, wp] = True
            if eos_token_id is not None:
                finished[rows] |= next_tokens == eos_token_id
            gen_len_vec[rows] += 1
            write_pos[rows] += 1

            # Embed new token and run one decode-time pass per layer.
            x_step = self.embedding(next_tokens)

            for cache in caches:
                x_norm = cache["ln_1"](x_step)
                qkv_out = cache["Wqkv"](x_norm)
                q_, k_, v_ = qkv_out.chunk(3, dim=-1)

                q_view = q_.view(n_active, 1, n_heads, head_dim)
                k_view = k_.view(n_active, 1, n_heads, head_dim)
                v_view = v_.view(n_active, 1, n_heads, head_dim)

                o = flash_attn_with_kvcache(
                    q_view,
                    cache["k"],
                    cache["v"],
                    k=k_view,
                    v=v_view,
                    rotary_cos=rotary_cos,
                    rotary_sin=rotary_sin,
                    cache_seqlens=cache["seqlens"][rows],
                    cache_batch_idx=rows_int32,
                    causal=True,
                    rotary_interleaved=rotary_interleaved,
                    softmax_scale=cache["softmax_scale"],
                    num_splits=2,
                )

                o = o.view(n_active, -1)
                y = cache["drop"](cache["out_proj"](o))
                x_step = x_step + y
                x_step = x_step + cache["mlp"](cache["ln_2"](x_step))

                cache["seqlens"][rows] += 1

            x_step = self.transformer.ln_f(x_step)
            last_hidden_current[rows] = x_step

            if finished.all():
                break

        # 4. Output assembly.
        gen_max = int(gen_len_vec.max().item())
        final_max = int(write_pos.max().item())

        if gen_max > 0:
            offsets = torch.arange(gen_max, device=device).unsqueeze(0)
            gather_indices = prompt_len_vec.unsqueeze(1) + offsets
            gather_indices = gather_indices.clamp(max=T - 1)
            generated = batch["event"].gather(1, gather_indices)
            valid_mask = offsets < gen_len_vec.unsqueeze(1)
            generated = torch.where(
                valid_mask, generated, torch.full_like(generated, pad_token_id)
            )
        else:
            generated = torch.full(
                (B, 1), pad_token_id, dtype=batch["event"].dtype, device=device
            )

        out = {
            "prompt": batch["event"][:, :L],
            "generated": generated,
            "full": batch["event"][:, :final_max],
            "generation_lengths": gen_len_vec,
            "finished": finished,
            "prompt_length": L,
            "prompt_length_per_row": prompt_len_vec,
        }
        if return_attention_mask:
            out["generated_attn_mask"] = batch["attn_mask"][:, :final_max]
        return out
