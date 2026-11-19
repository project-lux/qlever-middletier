# lux_embed — embedding builder for LUX record classes

This package builds per-class vector indexes (item, work, set, agent, place, concept, event) from a LUX record LMDB. They're stored in the LMDB vector store (`qleverlux/vectors/lmdb_vector_store.py`) and served in place by the patched USearch (`view_ram_buffer`).

| Step | Command | What it does |
| --- | --- | --- |
| Pass 1 | `labels` | One parallel scan writes UUID → (class, primary name). References in LUX records carry no usable labels, and `_label` is informative only. Names come from `identified_by`: English primary, then untagged primary, then any primary, then other names. The class column also serves as the index for enumerating one class. |
| Stage 1 | (inside `embed` / `cards`, with `--qlever`) | Incoming facets from QLever, in batched `VALUES` queries:<br>• **works:** the creators, materials, types, places and dates of the objects carrying them<br>• **agents:** places, types, mediums, subjects and active span of what they made, ranked by tf-idf<br>• **sets:** top member creators, types and subjects<br>Values on more than half of all objects are dropped. Global frequencies are cached in `facet_df.json` next to the label store. Facet patterns whose predicates the endpoint lacks are skipped. |
| Stage 1 offline | `facets` | The same facets, extracted once beside QLever into a portable LMDB (`--out facets.lmdb`). `cards` / `embed` then take `--facets facets.lmdb` instead of `--qlever`, so the embedding machine needs no endpoint. See below. |
| Stage 2 | `cards` (preview, or `--out` to build them all) | One card per record, using the per-class field table in `cards.py`. Hierarchies (concept `broader`, place `part_of`) are walked in the records themselves. Statements are English or untagged, HTML-stripped, boilerplate kinds removed, and capped at 300 words. Cards are built by a pool of worker processes (`--card-workers`, default: the CPUs the job may use, less two). With `--out cards.lmdb` every card of the scope goes into a card store, so it's a CPU-only job. |
| Stage 3 | `embed` (`--cards cards.lmdb` to read a card store) | EmbeddingGemma 2 (text-only, 271M parameters). Runs in bf16 on CUDA/MPS, else f32, never f16. Documents use `title: … \| text: …`. Vectors use Matryoshka `truncate_dim` and are re-normalized in f32. The run resumes after the last committed 4096-row chunk. |
| Graph | `index` (run by `embed` unless `--no-index`) | A bf16 HNSW graph over the stored vectors, published as the scope's next generation. Readers pick it up with `refresh()`. |

## Setup

This is a separate command from the middle tier, run from this repo's root with its own venv (`../vectors/.venv`; `requirements-vectors.txt` lists what it needs). That venv is self-contained. It holds sentence-transformers ≥ 6.1 (with the `image` extra, because the processor config needs Pillow and torchvision even for text), lmdb, orjson, and a copy of the patched `usearch` package. Don't layer it over `ENV`: mixing the two sets of compiled packages segfaults at interpreter exit.

```bash
cd ~/Development/lux/qlever-middletier                              # this repo's root
PY=../vectors/.venv/bin/python
DATA=../vectors/data/mds                                    # the built stores live outside the repo
R=~/Development/experiments/mds/middletier/datacache/nrich_lux.lmdb   # record LMDB (either layout)
$PY -m lux_embed labels --records $R --labels $DATA/labels.lmdb          # ~25 s for 14M
$PY -m lux_embed kinds  --records $R                                     # check SKIPPED_STATEMENTS
$PY -m lux_embed cards  --records $R --labels $DATA/labels.lmdb --scope agent --limit 5 --qlever http://localhost:7011
HF_HUB_OFFLINE=1 $PY -m lux_embed embed --records $R --labels $DATA/labels.lmdb \
    --store $DATA/vectors.lmdb --scope agent --qlever http://localhost:7011
$PY -m lux_embed query --store $DATA/vectors.lmdb --labels $DATA/labels.lmdb --scope agent "Edo period print designers"
```

`RecordStore` auto-detects both record LMDB layouts:
- the middle tier's datacache: 36-character UUID keys, values `{"json": record, ...}`;
- data-pipeline's distribution store: 16-byte keys, bare records.

**Defaults:**
- **Dimensions:** 512d for item and work, 768d for the other classes (`--dims` overrides). Any wider query vector is sliced and re-normalized to an index's dimension, so cross-class queries work.
- **Model:** EmbeddingGemma 2 unless `embed --model` names another model with a profile in `encoder.py`. The profile sets the document format, the query prompt, and whether the model can be truncated. Today the only alternative is `microsoft/harrier-oss-v1-0.6b`. It produces 1024d vectors and wasn't trained for truncation, so it is used at full width in every class.
  - The store records its model. `query` encodes with that model, and `embed` won't resume a store built with a different one.
  - Every class in a store must use the same model, because cross-class queries reuse one query vector.
  - The `sparql_service` sidecar encodes text queries with the model the store records.
- **Trial runs:** `--limit N` marks the class as a trial. Use `--restart` for the full build.
- **Vector storage:** `--vector-dtype float16` halves the space taken by the exact vectors (about 30 GB → 15 GB for MDS). That's safe for unit vectors; only inference must avoid f16.

## Throughput (M4 Max GPU (MPS), bf16, MDS data)

| Class | Cards/s | Note |
| --- | ---: | --- |
| item | 272 | longest cards, plus a 2-hop fetch of the depicted work |
| work | 549 | short cards, plus Stage 1 facets |
| agent | ~360 | Stage 1 facets |
| set | 83 | Stage 1 is bound by large memberships |

That's about 12 hours for all 14.3M MDS records. The full ~75M LUX instance is a CUDA job: the device is picked automatically, and each class can run on its own.

## Running on a CUDA machine (e.g. RTX 6000)

1. **Check out this repo.** The build needs `lux_embed/`, `qleverlux/vectors/` and `files/usearch-view-ram-buffer.patch`; nothing else has to be installed for it.
2. **USearch:**
   1. Check out USearch `840374af` (3.0.0-rc.1).
   2. Apply the patch from inside the checkout: `git apply <this repo>/files/usearch-view-ram-buffer.patch`.
   3. Install it with `pip install ./USearch`.

   Building graphs only needs stock v3. The patch matters only where graphs are served in place.
3. **Python environment:** a fresh venv on **Python 3.12 or later**. numpy ≥ 2.5 and USearch 3 both require it; `uv python install 3.12` provides one without root. The venv needs
   - a CUDA build of torch;
   - `'sentence-transformers[image]>=6.1'`, `lmdb`, `orjson` and `numpy`.

   Then make the model available locally. The venv has the `hf` command (from `huggingface_hub`, installed with sentence-transformers; older releases called it `huggingface-cli`):
   - `hf download google/embeddinggemma-2` (gated: accept its terms on the Hub and run `hf auth login` first);
   - or `hf download microsoft/harrier-oss-v1-0.6b`;
   - or copy the model's folder from another machine's HF cache (`~/.cache/huggingface/hub/models--<org>--<name>`).
4. **Inputs:**
   - Build the label store on that machine (`labels`), or copy it over.
   - Facets: either give `--qlever` an endpoint serving the same dataset, or extract them beside QLever
     and copy the result over (next section).
5. **Batch sizes:** a 48–96 GB card takes much larger batches than MPS. Start with `--encode-batch 256 --batch 8192`, and run one `embed` process per class, each with its own `--store` (LMDB allows one writer at a time).
   - bf16 is picked automatically.
   - Keep `--max-seq-length 512` unless cards are being truncated.
6. **Commands:** they're unchanged. Run `embed` per class in this order:
   1. `concept`, `place`, `event`, `set` first — small, and they validate the setup;
   2. then `agent`, `work`, `item`.

## Cards on CPUs, embeddings on GPUs

Building a card is pure Python, about 250 cards/s per core on full LUX. An H200 encodes about
900/s with Harrier and 1,500/s with Gemma. So on a cluster, build the cards first, as a CPU job:

```bash
python -m lux_embed cards --records $R --labels $L --facets facets.lmdb --scope work --out cards-work.lmdb
python -m lux_embed embed --cards cards-work.lmdb --store vectors-work.lmdb --scope work \
    --model microsoft/harrier-oss-v1-0.6b --encode-batch 256 --batch 8192 --vector-dtype float16 --no-index
```

- **What the GPU job needs:** only the card store, not the record LMDB, label store or facets. It never
  waits for cards, so a few CPUs are enough.
- **Reuse:** cards don't depend on the model (each encoder's document format is applied when it
  encodes). One build serves Harrier, Gemma, or a different `--max-seq-length`.
- **Resuming:** both steps resume after their last commit. `embed --cards` refuses a card store that is
  still being built, unless it's a `--limit` trial.
- **One writer per file:** give each scope that runs at the same time its own card store and vector store.

### Vectors as files, loaded afterwards

GPU jobs can write plain files instead of an LMDB store, and a big class can be split by key range
over several GPUs:

```bash
python -m lux_embed embed --cards cards-item.lmdb --scope item --out vec/item-3 --shard 3/8 \
    --model microsoft/harrier-oss-v1-0.6b --encode-batch 256 --batch 8192 --vector-dtype float16
python -m lux_embed load --store vectors.lmdb --scope item vec/item-*       # one CPU job, then the graph
```

- **What a shard holds:** a directory with `vectors.bin` (the rows), `iris.txt` and `meta.json`. It
  resumes after the last flushed batch, like `--store`.
- **What `load` does:** checks that the shards are finished, use the same model, width and dtype, and
  cover `0..N-1` once each. Then it writes them into the store in key order, byte for byte what
  `embed --store` would have written, and builds the graph (`--no-index` to skip).
- **Why bother:** no GPU job ever opens an LMDB store, so there's no writer contention, and every
  class can be loaded into the one `vectors.lmdb` that the sidecar reads. A store that already holds
  the scope needs `--restart`.

## Extracting facets where QLever runs

When the embedding machine can't reach QLever, extract Stage 1 on the QLever host. It needs only
this repo's `lux_embed/` and Python 3.10+ with `lmdb`: no torch, and no record LMDB.

```bash
python -m lux_embed facets --qlever http://localhost:7011 --out facets.lmdb             # work, agent, set
python -m lux_embed facets --qlever http://localhost:7011 --out facets.lmdb --scope set  # or one scope
```

- **What it does:** lists every target with an incoming join once (`targets-<scope>.txt`), then runs the
  `embed` batches over it with `--workers` concurrent queries (default 4).
- **What it stores:** the ranked, capped result per target, with values as UUIDs.
- **Resuming:** rerunning resumes after the last committed batch. A completed scope is skipped unless
  you pass `--restart`.
- **Document frequencies:** `facet_df.json` is cached inside the output directory. It's needed only here,
  because ranking happens at extraction time.

Copy the whole directory, then:

```bash
$PY -m lux_embed embed --records $R --labels $DATA/labels.lmdb --store $DATA/vectors.lmdb \
    --scope agent --facets facets.lmdb
```

The cards come out identical to `--qlever`. `--facets` on a scope that hasn't been extracted stops with an error.
A partly extracted scope warns and builds the remaining cards without facets. Scopes other than
work/agent/set ignore it.

## Known data notes (MDS)

- **Works:** these are all `VisualItem`. Their maker, place and date live on the carrying object, so work cards depend on Stage 1.
- **Concepts:** mostly local terms with no `broader`, so their cards are short.
- **Statement kinds:** only Description, Biography and Note occur. Full LUX has many more, mostly from library
  cataloguing (access, physical, bibliography, edition, language, location and provenance statements);
  `SKIPPED_STATEMENTS` in `linkedart.py` drops those. Re-run `kinds` on a new dataset to check it.
