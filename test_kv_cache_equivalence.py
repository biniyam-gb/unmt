"""
Proves generate_greedy_cached produces EXACTLY the same output as
generate_greedy across multiple model sizes, seeds, batch sizes, and padding
patterns. This is the correct validation methodology for a pure speed
optimization: it should change nothing about the result, only how much
compute it costs to get there. Re-run this any time model.py's decoder is
touched -- KV-cache bugs are silent (they don't crash, they just quietly
corrupt the back-translation training signal).
"""
import torch

from config import ModelConfig, PAD_ID
from model import SharedTransformerNMT


def _pad_to(t: torch.Tensor, length: int) -> torch.Tensor:
    if t.size(1) == length:
        return t
    pad = torch.full((t.size(0), length - t.size(1)), PAD_ID, dtype=t.dtype)
    return torch.cat([t, pad], dim=1)


def main():
    configs = [
        dict(vocab_size=200, d_model=32, n_heads=4, n_enc_layers=3, n_dec_layers=3, d_ff=64, dropout=0.1, max_len=15),
        dict(vocab_size=500, d_model=64, n_heads=8, n_enc_layers=2, n_dec_layers=4, d_ff=128, dropout=0.0, max_len=30),
        dict(vocab_size=50, d_model=8, n_heads=2, n_enc_layers=1, n_dec_layers=1, d_ff=16, dropout=0.2, max_len=10),
    ]
    n_checked = 0
    all_pass = True
    for cfg_i, cfg_kwargs in enumerate(configs):
        for seed in range(5):
            torch.manual_seed(seed * 17 + cfg_i)
            cfg = ModelConfig(n_langs=2, **cfg_kwargs)
            model = SharedTransformerNMT(cfg)
            model.eval()  # dropout must be off for the two paths to be comparable at all
            for B in (1, 2, 7):
                src_len = torch.randint(3, 10, (1,)).item()
                src = torch.randint(4, cfg.vocab_size, (B, src_len))
                for b in range(B):
                    if torch.rand(1).item() < 0.4 and src_len > 3:
                        cut = torch.randint(2, src_len, (1,)).item()
                        src[b, cut:] = PAD_ID

                out_uncached = model.generate_greedy(src, 0, 1, max_len=cfg.max_len)
                out_cached = model.generate_greedy_cached(src, 0, 1, max_len=cfg.max_len)
                L = max(out_uncached.size(1), out_cached.size(1))
                same = torch.equal(_pad_to(out_uncached, L), _pad_to(out_cached, L))
                n_checked += 1
                if not same:
                    all_pass = False
                    print(f"MISMATCH cfg={cfg_i} seed={seed} B={B} src_len={src_len}")
                    print("  uncached:", out_uncached.tolist())
                    print("  cached:  ", out_cached.tolist())

    print(f"Checked {n_checked} (config x seed x batch-size) combinations.")
    print("ALL PASS -- cached and uncached decoding are numerically identical" if all_pass
          else "*** MISMATCHES FOUND ABOVE -- do not use generate_greedy_cached until fixed ***")
    return all_pass


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
