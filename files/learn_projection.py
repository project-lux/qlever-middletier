"""Learn a linear map that shrinks vectors for the HNSW graph without losing anyone's neighbours.

The graph only has to put each search's true neighbours somewhere in its pool (1000), because the
pool is reranked with the full vectors. So this learns ``y = normalize((x - mean) @ W)`` such
that, in the reduced space, each anchor's exact full-width top 100 ranks above everything that is
not in its full-width top 1000. Ranks 101-1000 are left alone: they are harmless in the pool.

    ../vectors/.venv/bin/python files/learn_projection.py --store ../vectors/data/mds/vectors-harrier.lmdb \\
        --scope concept --dims 256 --fit-texts fit_labels.txt --out harrier256.npz
    ../vectors/.venv/bin/python files/graph_quantization.py ... --reductions learned:harrier256.npz

Anchors are encoded query texts (``--fit-texts``) and record vectors (more-like-this), mixed in
every batch - half each, or ``--record-share`` records - so neither kind is traded for the other.
Training starts from query-aware PCA on the training queries (``qpcaN@1`` in graph_quantization.py,
the best fixed projection measured), so any gain is the training's. The negatives are every
record outside the anchor's full-width top 1000, scored with the map as it is at that step (the
whole store is projected each step anyway),
and each positive is pushed above all of them: ``softplus(logsumexp(negatives) - positive)``,
scores over a temperature. Negatives mined once an epoch went stale within it: the map pushed
down the intruders it was shown and made new ones, and validation recall fell and sawtoothed.
Validation runs every ``--eval-every`` steps and the best map is saved, so a run that never beats
its start saves the start.

Kept out of training, so the benchmark measures held-out queries: the benchmark's 200 seed
records (``default_rng(0)`` over the whole scope) and its 30 built-in texts. Keep ``--fit-texts``
apart from any ``--texts`` it is evaluated on. A slice of the anchors is held back as validation
and reported every epoch as pool recall: the share of the full top 10 / top 100 that the reduced
space's exact top 1000 contains, which is the benchmark's ceiling.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from files.graph_quantization import DEFAULT_TEXTS, load_vectors

POSITIVES = 100  # the full-width top K the reduced space must keep in its pool
NEUTRAL = 1000  # full-width ranks below this are never negatives: the pool reranks them
POOL = 1000  # the pool the graph walks for, as in the benchmark


def encode_queries(model: str, texts: list[str], dims: int, device: str) -> np.ndarray:
    from lux_embed.encoder import Encoder

    encoder = Encoder(device=device, model=model)
    return encoder._finish(encoder.model.encode(  # encode_query's prompt, in batches
        texts, batch_size=64, prompt_name=encoder.profile["query_prompt_name"],
        truncate_dim=dims, normalize_embeddings=True, convert_to_numpy=True))


def top_ids(store: torch.Tensor, anchors: torch.Tensor, k: int, exclude: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Each anchor's ``k`` highest-scoring store rows by dot product, its own row (``exclude``,
    -1 for none) left out."""
    out = []
    for start in range(0, len(anchors), chunk):
        scores = anchors[start:start + chunk] @ store.T
        own = exclude[start:start + chunk]
        rows = (own >= 0).nonzero().squeeze(1)
        scores[rows, own[rows]] = -torch.inf
        out.append(scores.topk(k, dim=1).indices)
    return torch.cat(out)


def project(x: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return F.normalize((x - mean) @ basis, dim=1)


def query_pca(queries: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """``qpcaN@1``: the mean and top principal directions of the training queries."""
    mean = queries.mean(axis=0)
    centred = (queries - mean).astype(np.float64)
    values, vectors = np.linalg.eigh(centred.T @ centred / len(centred))
    return mean.astype(np.float32), vectors[:, np.argsort(values)[::-1][:dims]].astype(np.float32)


def pool_recall(store, anchors, exclude, full_top, mean, basis) -> tuple[float, float]:
    """Share of the full top 10 / top 100 inside the reduced space's exact top ``POOL``."""
    with torch.no_grad():
        pool = top_ids(project(store, mean, basis), project(anchors, mean, basis), POOL, exclude).cpu().numpy()
    truth = full_top.cpu().numpy()
    hits = [(len(set(p) & set(t[:10])) / 10, len(set(p) & set(t[:POSITIVES])) / POSITIVES) for p, t in zip(pool, truth)]
    return tuple(np.mean(hits, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", required=True)
    parser.add_argument("--scope", default="concept")
    parser.add_argument("--dims", type=int, default=256)
    parser.add_argument("--fit-texts", required=True, help="training query texts, one a line")
    parser.add_argument("--records", type=int, default=30000, help="record vectors used as anchors")
    parser.add_argument("--validation", type=int, default=500, help="queries, and as many records, held back")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=50, help="steps between validations; the best is saved")
    parser.add_argument("--record-share", type=float, default=0.5,
                        help="share of each batch that is record anchors (more-like-this) rather than queries")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="only the first N vectors, for a quick run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(2)

    vectors, meta = load_vectors(args.store, args.scope, args.limit)
    count, width = vectors.shape
    print(f"{args.scope}: {count:,} vectors, {width}d, model {meta.get('model')}; learning {width} -> {args.dims}")

    texts = [line.strip() for line in open(args.fit_texts) if line.strip()]
    clash = {t.casefold() for t in texts} & {t.casefold() for t in DEFAULT_TEXTS}
    if clash:
        raise SystemExit(f"--fit-texts include the benchmark's own queries, e.g. {next(iter(clash))!r}")
    started = time.perf_counter()
    queries = encode_queries(meta.get("model"), texts, width, args.device)
    print(f"{len(queries):,} queries encoded in {time.perf_counter() - started:,.0f} s")

    # Records the benchmark seeds from are never anchors (they may still be neighbours).
    held = set(np.random.default_rng(0).choice(count, 200, replace=False).tolist())
    pickable = np.array([i for i in range(count) if i not in held])
    records = rng.choice(pickable, args.records + args.validation, replace=False)
    order = rng.permutation(len(queries))
    val_q, train_q = order[:args.validation], order[args.validation:]
    val_r, train_r = records[:args.validation], records[args.validation:]

    def anchors_of(query_rows, record_rows):
        anchors = np.vstack([queries[query_rows], vectors[record_rows]])
        own = np.concatenate([np.full(len(query_rows), -1), record_rows])
        return torch.from_numpy(anchors).to(device), torch.from_numpy(own).to(device)

    store = torch.from_numpy(vectors).to(device)
    train_a, train_own = anchors_of(train_q, train_r)
    val_a, val_own = anchors_of(val_q, val_r)
    started = time.perf_counter()
    with torch.no_grad():
        train_top = top_ids(store, train_a, NEUTRAL, train_own)
        val_top = top_ids(store, val_a, POSITIVES, val_own)
    print(f"full-width neighbours of {len(train_a) + len(val_a):,} anchors in {time.perf_counter() - started:,.0f} s")

    init_mean, init_basis = query_pca(queries[train_q], args.dims)
    mean = torch.nn.Parameter(torch.from_numpy(init_mean).to(device))
    basis = torch.nn.Parameter(torch.from_numpy(init_basis).to(device))
    optimiser = torch.optim.Adam([mean, basis], lr=args.lr)
    n_q = len(train_q)

    def report(label):
        """Validation pool recall; returns (queries, records) at @100, what checkpoints are kept on."""
        vq = pool_recall(store, val_a[:len(val_q)], val_own[:len(val_q)], val_top[:len(val_q)], mean, basis)
        vr = pool_recall(store, val_a[len(val_q):], val_own[len(val_q):], val_top[len(val_q):], mean, basis)
        print(f"{label:>10}: validation pool recall@10/@100  queries {vq[0]:.3f}/{vq[1]:.3f}"
              f"  records {vr[0]:.3f}/{vr[1]:.3f}", flush=True)
        return vq[1], vr[1]

    best = (sum(report("start")), 0, init_mean, init_basis)
    per_r = round(args.batch * args.record_share)  # record anchors in a batch
    per_q = args.batch - per_r
    step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        # A fixed share of queries and of records in every batch, whatever their numbers; an
        # epoch is as long as a half-and-half one would be, so shares compare step for step.
        steps = max(n_q, len(train_a) - n_q) * 2 // args.batch
        q_rows = torch.from_numpy(rng.integers(0, n_q, steps * per_q)).to(device)
        r_rows = torch.from_numpy(rng.integers(n_q, len(train_a), steps * per_r)).to(device)
        total = 0.0
        for i in range(steps):
            rows = torch.cat([q_rows[i * per_q:(i + 1) * per_q], r_rows[i * per_r:(i + 1) * per_r]])
            # Every record is scored against the batch with the map as it is now, so the negatives
            # are exactly what would push a positive out of the pool at this step: all records
            # outside the anchor's full-width top 1000 (which holds the positives) and itself.
            scores = project(train_a[rows], mean, basis) @ project(store, mean, basis).T / args.temperature
            pos = scores.gather(1, train_top[rows, :POSITIVES])
            neutral = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, train_top[rows], True)
            own = train_own[rows]
            records = (own >= 0).nonzero().squeeze(1)
            neutral[records, own[records]] = True
            neg = torch.logsumexp(scores.masked_fill(neutral, -torch.inf), dim=1, keepdim=True)
            loss = F.softplus(neg - pos).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item()
            step += 1
            if step % args.eval_every == 0:
                score = sum(report(f"step {step}"))
                if score > best[0]:
                    best = (score, step, mean.detach().cpu().numpy().copy(), basis.detach().cpu().numpy().copy())
        print(f"epoch {epoch}: loss {total / steps:.4f}, {time.perf_counter() - started:,.0f} s so far", flush=True)

    _, at, best_mean, best_basis = best
    np.savez(args.out, mean=best_mean, basis=best_basis, model=meta.get("model"), scope=args.scope, step=at)
    print(f"saved {args.out}: best validation at step {at}" + (" (the starting map: training never beat it)" if at == 0 else ""))


if __name__ == "__main__":
    main()
