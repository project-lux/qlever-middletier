"""Where an embedding machine's time goes: raw GPU speed, attention kernels, and the encoder itself.

    python files/encode_benchmark.py --cards cards-agent.lmdb --scope agent \\
        [--model microsoft/harrier-oss-v1-0.6b] [--cards-n 4096]

Prints, in order:
1. bf16 matrix-multiply throughput (TFLOPS): what the tensor cores can do;
2. which scaled-dot-product attention backends run on this GPU, and how fast;
3. cards/s encoding real cards: as embed does today (one call; sentence-transformers sorts by
   characters and batches by count), then length-sorted batches filled to a token budget (little
   padding), then batches of exactly one length (no padding, so no attention mask and the flash
   kernel can run). Agent cards are short (median 36 tokens) with a long tail, so batching by count
   pads heavily, and padding forces a masked attention path.

Run it with nothing else on the GPU. Needs torch, sentence-transformers and lmdb; reads only the
card store.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def matmul_tflops(device: str, n: int = 8192, reps: int = 20) -> float:
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    a @ a
    synchronize(device)
    started = time.perf_counter()
    for _ in range(reps):
        a @ a
    synchronize(device)
    return reps * 2 * n**3 / (time.perf_counter() - started) / 1e12


def attention_backends(device: str) -> None:
    if device != "cuda":
        print("   (attention backends are only probed on CUDA)")
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(64, 16, 256, 64, device=device, dtype=torch.bfloat16)
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH):
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, q, q)
                synchronize(device)
                started = time.perf_counter()
                for _ in range(20):
                    F.scaled_dot_product_attention(q, q, q)
                synchronize(device)
            print(f"   {backend.name:20} works, {(time.perf_counter() - started) / 20 * 1e3:6.2f} ms a call")
        except Exception as error:  # a backend this GPU or build does not support
            print(f"   {backend.name:20} unavailable: {str(error).splitlines()[0][:90]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cards", required=True, help="card store built by `lux_embed cards --out`")
    parser.add_argument("--scope", default="agent")
    parser.add_argument("--model", default="microsoft/harrier-oss-v1-0.6b")
    parser.add_argument("--cards-n", type=int, default=4096, help="real cards to encode per measurement")
    parser.add_argument("--batches", default="64,256")
    parser.add_argument("--budgets", default="8192,16384,32768",
                        help="token budgets for length-sorted batches (cards a batch x its longest card)")
    parser.add_argument("--max-batch", type=int, default=1024, help="most cards in one length-sorted batch")
    parser.add_argument("--attention", default="sdpa", help="implementations to try, e.g. sdpa,eager")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    from lux_embed.card_store import CardStore
    from lux_embed.encoder import PROFILES

    device = args.device
    print(f"torch {torch.__version__}, {device}"
          + (f": {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}" if device == "cuda" else ""))
    print(f"1. bf16 matmul: {matmul_tflops(device):.1f} TFLOPS")
    print("2. attention backends:")
    attention_backends(device)

    profile = PROFILES[args.model]
    cards = []
    for batch in CardStore(args.cards, args.scope).batches(None, 1024):
        cards += [profile["document"].format(title=title, text=text) for _, title, text in batch]
        if len(cards) >= args.cards_n:
            break
    cards = cards[: args.cards_n]

    print(f"3. encoding {len(cards):,} real cards ({args.model}, bf16, max 512 tokens):")
    for attention in args.attention.split(","):
        try:
            model = SentenceTransformer(args.model, device=device, config_kwargs=profile["config_kwargs"],
                                        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": attention})
        except Exception as error:
            print(f"   {attention:6} could not load: {str(error).splitlines()[0][:90]}")
            continue
        model.max_seq_length = 512
        lengths = [min(len(ids), 512) for ids in model.tokenizer(cards)["input_ids"]]
        model.encode(cards[:256], batch_size=64)  # warm up

        def timed(label, batches):
            synchronize(device)
            started = time.perf_counter()
            for batch in batches:
                model.encode([cards[i] for i in batch], batch_size=len(batch), convert_to_numpy=True)
            synchronize(device)
            print(f"   {attention:6} {label:24}: {len(cards) / (time.perf_counter() - started):7.0f} cards/s"
                  f"  ({len(batches)} batches; mean {sum(lengths) / len(lengths):.0f} tokens a card)", flush=True)

        for size in (int(b) for b in args.batches.split(",")):
            # what embed does today: one call, which sentence-transformers sorts by characters and batches
            synchronize(device)
            started = time.perf_counter()
            model.encode(cards, batch_size=size, convert_to_numpy=True)
            synchronize(device)
            print(f"   {attention:6} {'batch ' + str(size):24}: {len(cards) / (time.perf_counter() - started):7.0f} cards/s",
                  flush=True)
        order = sorted(range(len(cards)), key=lambda i: -lengths[i])
        for budget in (int(b) for b in args.budgets.split(",")):
            batches, i = [], 0  # longest first; each batch as many cards as fit the budget once padded
            while i < len(order):
                size = max(1, min(args.max_batch, budget // lengths[order[i]]))
                batches.append(order[i:i + size])
                i += size
            timed(f"token budget {budget}", batches)
            groups = {}  # cards of exactly one length: no padding, so no attention mask
            for i in order:
                groups.setdefault(lengths[i], []).append(i)
            same = [g[j:j + max(1, min(args.max_batch, budget // n))]
                    for n, g in groups.items() for j in range(0, len(g), max(1, min(args.max_batch, budget // n)))]
            timed(f"same length, budget {budget}", same)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
