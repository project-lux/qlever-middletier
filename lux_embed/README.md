# lux_embed — embedding builder for LUX record classes

This package builds per-class vector indexes (item, work, set, agent, place, concept, event) from a LUX record LMDB. They're stored in the LMDB vector store (`qleverlux/vectors/lmdb_vector_store.py`) and served in place by the patched USearch (`view_ram_buffer`).

| Step | Command | What it does |
| --- | --- | --- |
| Pass 1 | `labels` | One parallel scan writes UUID → (class, primary name). References in LUX records carry no usable labels, and `_label` is informative only. Names come from `identified_by`: English primary, then untagged primary, then any primary, then other names. The class column also serves as the index for enumerating one class. |
| Stage 1 | (inside `embed` / `cards`, with `--qlever`) | Incoming facets from QLever, in batched `VALUES` queries:<br>• **works:** the creators, materials, types, places and dates of the objects carrying them<br>• **agents:** places, types, mediums, subjects and active span of what they made, ranked by tf-idf<br>• **sets:** top member creators, types and subjects<br>Values on more than half of all objects are dropped. Global frequencies are cached in `facet_df.json` next to the label store. Facet patterns whose predicates the endpoint lacks are skipped. |
| Stage 2 | `cards` (preview) | One card per record, using the per-class field table in `cards.py`. Hierarchies (concept `broader`, place `part_of`) are walked in the records themselves. Statements are English or untagged, HTML-stripped, boilerplate kinds removed, and capped at 300 words. |
| Stage 3 | `embed` | EmbeddingGemma 2 (text-only, 271M parameters). Runs in bf16 on CUDA/MPS, else f32, never f16. Documents use `title: … \| text: …`. Vectors use Matryoshka `truncate_dim` and are re-normalized in f32. The run resumes after the last committed 4096-row chunk. |
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
3. **Python environment:** a fresh venv with
   - a CUDA build of torch;
   - `'sentence-transformers[image]>=6.1'`, `lmdb`, `orjson` and `numpy`.

   Then make the model available locally, either with `huggingface-cli download google/embeddinggemma-2` or by copying the HF cache.
4. **Inputs:**
   - Build the label store on that machine (`labels`), or copy it over together with `facet_df.json`.
   - Give `--qlever` the endpoint that serves the same dataset.
5. **Batch sizes:** a 48–96 GB card takes much larger batches than MPS. Start with `--encode-batch 256 --batch 8192`, and run one `embed` process per class.
   - bf16 is picked automatically.
   - Keep `--max-seq-length 512` unless cards are being truncated.
6. **Commands:** they're unchanged. Run `embed` per class in this order:
   1. `concept`, `place`, `event`, `set` first — small, and they validate the setup;
   2. then `agent`, `work`, `item`.

## Known data notes (MDS)

- **Works:** these are all `VisualItem`. Their maker, place and date live on the carrying object, so work cards depend on Stage 1.
- **Concepts:** mostly local terms with no `broader`, so their cards are short.
- **Statement kinds:** only Description, Biography and Note occur. Re-run `kinds` on full LUX and extend `SKIPPED_STATEMENTS` in `linkedart.py` as needed.
