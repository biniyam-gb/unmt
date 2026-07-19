"""
A single shared Transformer encoder-decoder used for BOTH languages and BOTH
directions of translation, exactly as in Lample et al. 2018 ("Unsupervised MT
Using Monolingual Corpora Only") / Artetxe et al. 2018 ("Unsupervised NMT").

Key idea: there is only ONE encoder and ONE decoder. Which language is being
read/written is signalled purely by a learned language embedding added to the
token+position embedding at every input position. This is what lets the model
transfer structure across languages and is what makes online back-translation
possible with a single set of weights (no separate en->fi / fi->en models).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, PAD_ID, BOS_ID, EOS_ID


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1), :]


class SharedTransformerNMT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.lang_emb = nn.Embedding(cfg.n_langs, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.max_len + 8)
        self.emb_scale = math.sqrt(cfg.d_model)  # standard Transformer embedding scaling
        self.emb_dropout = nn.Dropout(cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_enc_layers,
                                              norm=nn.LayerNorm(cfg.d_model),
                                              enable_nested_tensor=False)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.n_dec_layers,
                                              norm=nn.LayerNorm(cfg.d_model))

        # Weight tying: output projection shares weights with input token embedding
        # (Press & Wolf 2017). Saves ~16M params at vocab=32000, d_model=512 and
        # generally helps quality.
        self.output_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=self.cfg.d_model ** -0.5)
        with torch.no_grad():
            self.token_emb.weight[PAD_ID].zero_()
        nn.init.normal_(self.lang_emb.weight, mean=0.0, std=0.02)

    def _embed(self, ids: torch.Tensor, lang_id: int) -> torch.Tensor:
        # ids: (B, T) long
        lang_vec = self.lang_emb.weight[lang_id].view(1, 1, -1)  # (1,1,d_model)
        x = self.token_emb(ids) * self.emb_scale + lang_vec
        x = self.pos_enc(x)
        return self.emb_dropout(x)

    @staticmethod
    def _padding_mask(ids: torch.Tensor) -> torch.Tensor:
        return ids.eq(PAD_ID)  # True where PAD -> ignored by attention (verified convention)

    def encode(self, src_ids: torch.Tensor, src_lang_id: int):
        src_kpm = self._padding_mask(src_ids)
        x = self._embed(src_ids, src_lang_id)
        memory = self.encoder(x, src_key_padding_mask=src_kpm)
        return memory, src_kpm

    @staticmethod
    def _causal_mask(sz: int, device) -> torch.Tensor:
        # Boolean mask (True = blocked). Kept the same dtype (bool) as the padding
        # masks throughout -- mixing a float additive mask with a bool padding mask
        # is deprecated in recent PyTorch and triggers silent-fallback warnings.
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

    def decode_logits(self, tgt_in_ids: torch.Tensor, tgt_lang_id: int,
                       memory: torch.Tensor, src_kpm: torch.Tensor) -> torch.Tensor:
        tgt_kpm = self._padding_mask(tgt_in_ids)
        causal = self._causal_mask(tgt_in_ids.size(1), tgt_in_ids.device)
        y = self._embed(tgt_in_ids, tgt_lang_id)
        h = self.decoder(y, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_kpm,
                          memory_key_padding_mask=src_kpm)
        logits = F.linear(h, self.token_emb.weight, self.output_bias)
        return logits

    def forward(self, src_ids, src_lang_id, tgt_in_ids, tgt_lang_id):
        """Teacher-forced training forward pass. Returns logits (B, T, V)."""
        memory, src_kpm = self.encode(src_ids, src_lang_id)
        return self.decode_logits(tgt_in_ids, tgt_lang_id, memory, src_kpm)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_greedy(self, src_ids, src_lang_id, tgt_lang_id, max_len=None):
        """Fast greedy decode. Used inside the online back-translation loop,
        where we need to generate synthetic pairs every training step and
        cannot afford beam search there."""
        self.eval()
        max_len = max_len or self.cfg.max_len
        B = src_ids.size(0)
        device = src_ids.device
        memory, src_kpm = self.encode(src_ids, src_lang_id)

        ys = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            logits = self.decode_logits(ys, tgt_lang_id, memory, src_kpm)
            next_tok = logits[:, -1, :].argmax(-1)  # (B,)
            next_tok = torch.where(finished, torch.full_like(next_tok, PAD_ID), next_tok)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished = finished | next_tok.eq(EOS_ID)
            if finished.all():
                break
        return ys

    @torch.no_grad()
    def generate_greedy_cached(self, src_ids, src_lang_id, tgt_lang_id, max_len=None):
        """Numerically equivalent to generate_greedy (verified in
        test_kv_cache_equivalence.py) but asymptotically cheaper: caches each
        decoder layer's raw (pre-norm1) hidden state per position instead of
        recomputing self-attention over the whole growing sequence at every
        step. Without this, generating a length-T sequence costs O(T^3)
        attention compute (recompute all T positions at each of T steps);
        with it, O(T^2) (process only the 1 new position each step, attending
        over a cache that grows by one). This matters here specifically
        because generation runs on EVERY back-translation training step, not
        just at evaluation time -- see train_bt.py."""
        self.eval()
        max_len = max_len or self.cfg.max_len
        B = src_ids.size(0)
        device = src_ids.device
        memory, src_kpm = self.encode(src_ids, src_lang_id)

        layers = list(self.decoder.layers)
        cache = [torch.zeros(B, 0, self.d_model, device=device, dtype=memory.dtype) for _ in layers]
        lang_vec = self.lang_emb.weight[tgt_lang_id].view(1, 1, -1)

        ys = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        cur_token = ys

        for t in range(max_len - 1):
            x = self.token_emb(cur_token) * self.emb_scale + lang_vec  # (B,1,D)
            x = x + self.pos_enc.pe[:, t:t + 1, :]
            x = self.emb_dropout(x)  # no-op in eval(), kept for exact parity with generate_greedy's code path

            new_cache = []
            for layer, layer_cache in zip(layers, cache):
                full_raw = torch.cat([layer_cache, x], dim=1)  # (B, t+1, D): this layer's raw input, all positions so far
                q = layer.norm1(x)
                kv = layer.norm1(full_raw)
                attn_out = layer.self_attn(q, kv, kv, need_weights=False)[0]  # no mask needed: 1 query, causally sees all cached (=past+self) keys by construction
                x = x + layer.dropout1(attn_out)

                q2 = layer.norm2(x)
                cross_out = layer.multihead_attn(q2, memory, memory, key_padding_mask=src_kpm, need_weights=False)[0]
                x = x + layer.dropout2(cross_out)

                ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(x)))))
                x = x + layer.dropout3(ff)

                new_cache.append(full_raw)
            cache = new_cache

            x = self.decoder.norm(x)  # nn.TransformerDecoder applies its final norm once, after all layers
            logits = F.linear(x, self.token_emb.weight, self.output_bias)  # (B,1,V)
            next_tok = logits[:, -1, :].argmax(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, PAD_ID), next_tok)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished = finished | next_tok.eq(EOS_ID)
            cur_token = next_tok.unsqueeze(1)
            if finished.all():
                break
        return ys

    @torch.no_grad()
    def generate_beam(self, src_ids, src_lang_id, tgt_lang_id, beam_size=5,
                       max_len=None, length_penalty_alpha=0.6):
        """Batched beam search. Used only at evaluation time (infrequent, so the
        extra compute over greedy is a non-issue)."""
        self.eval()
        max_len = max_len or self.cfg.max_len
        device = src_ids.device
        B = src_ids.size(0)
        memory, src_kpm = self.encode(src_ids, src_lang_id)  # (B, S, D), (B, S)

        S = memory.size(1)
        D = memory.size(2)
        # expand to (B*beam, ...)
        memory = memory.unsqueeze(1).expand(B, beam_size, S, D).reshape(B * beam_size, S, D)
        src_kpm = src_kpm.unsqueeze(1).expand(B, beam_size, S).reshape(B * beam_size, S)

        ys = torch.full((B * beam_size, 1), BOS_ID, dtype=torch.long, device=device)
        # only the first beam of each batch item is "real" initially
        beam_scores = torch.full((B, beam_size), float("-inf"), device=device)
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.view(B * beam_size)
        done = torch.zeros(B * beam_size, dtype=torch.bool, device=device)

        best_seq = [None] * B
        best_score = [float("-inf")] * B

        vocab_size = self.cfg.vocab_size
        for step in range(max_len - 1):
            logits = self.decode_logits(ys, tgt_lang_id, memory, src_kpm)
            logprobs = F.log_softmax(logits[:, -1, :], dim=-1)  # (B*beam, V)
            # freeze finished beams: force EOS->PAD with zero additional cost
            logprobs = torch.where(
                done.unsqueeze(1),
                torch.full_like(logprobs, float("-inf")).scatter(
                    1, torch.full((logprobs.size(0), 1), PAD_ID, device=device), 0.0
                ),
                logprobs,
            )
            cand_scores = beam_scores.unsqueeze(1) + logprobs  # (B*beam, V)
            cand_scores = cand_scores.view(B, beam_size * vocab_size)
            topk_scores, topk_idx = cand_scores.topk(beam_size, dim=-1)  # (B, beam)
            beam_idx = topk_idx // vocab_size   # which beam each candidate came from
            tok_idx = topk_idx % vocab_size     # which token was chosen

            # reorder ys according to beam_idx (per batch item)
            flat_beam_idx = (beam_idx + torch.arange(B, device=device).unsqueeze(1) * beam_size).view(-1)
            ys = ys[flat_beam_idx]
            ys = torch.cat([ys, tok_idx.view(-1, 1)], dim=1)
            beam_scores = topk_scores.view(-1)
            done = done[flat_beam_idx] | tok_idx.view(-1).eq(EOS_ID)

            memory = memory[flat_beam_idx]
            src_kpm = src_kpm[flat_beam_idx]

            # harvest finished beams with length-normalized score
            ys_r = ys.view(B, beam_size, -1)
            scores_r = beam_scores.view(B, beam_size)
            done_r = done.view(B, beam_size)
            for b in range(B):
                for k in range(beam_size):
                    if done_r[b, k]:
                        length = ys_r.size(2)
                        lp = ((5 + length) / 6) ** length_penalty_alpha  # GNMT length penalty
                        norm_score = scores_r[b, k].item() / lp
                        if norm_score > best_score[b]:
                            best_score[b] = norm_score
                            best_seq[b] = ys_r[b, k].clone()

            if done.all():
                break

        # fallback for any batch item that never finished: take current best beam as-is
        for b in range(B):
            if best_seq[b] is None:
                best_seq[b] = ys.view(B, beam_size, -1)[b, 0].clone()

        max_out_len = max(s.size(0) for s in best_seq)
        out = torch.full((B, max_out_len), PAD_ID, dtype=torch.long, device=device)
        for b, s in enumerate(best_seq):
            out[b, : s.size(0)] = s
        return out
