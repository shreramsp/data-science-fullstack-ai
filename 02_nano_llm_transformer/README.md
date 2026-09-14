# 02 — NanoLlama Autoregressive SFT LLM

An independent, laptop-scale decoder-only transformer — built from modern
Llama-family primitives (RoPE, grouped-query attention, SwiGLU, RMSNorm) — pretrained
and instruction-tuned from scratch on a generated data-science corpus, with a
greedy hill-climbing architecture search and a Streamlit chatbot + admin
dashboard front end.

This implementation uses an original model pipeline, interface,
documentation, and reported results.

### Why not a real web-text corpus

Pretraining a ~4.6M-parameter model on even a small slice of real web text
does not produce a working chatbot — there isn't nearly enough capacity or
data to learn general language. To produce an honestly testable result at
this scale, the project generates its own **closed-domain**
corpus (`data/build_corpus.py`, seeded): 49 data-science concepts across 11
categories, 8 question intents, ~2,000 base (intent, key, template)
combinations, augmented to 6,408 training chat examples. The trade-off is
explicit and stated everywhere in this project: the model is fluent and
accurate *inside* that domain, and it says so when asked about anything
outside it. See `CRISP_DM.md` for the full reasoning and three data defects
that were found and fixed along the way.

## What was built

- `data/build_corpus.py` — seeded corpus generator: 49 concepts × 11
  categories × 8 intents (`define`, `use`, `category`, `family`, `compare`,
  `list`, `smalltalk`, `off_domain`), each with several paraphrase templates.
  Splits every intent 80/10/10 **before** augmentation, with a repair pass
  guaranteeing every concept/pair/category/small-talk group has at least one
  training example — then augments only the training split.
- `src/model.py` — `NanoLlama`: decoder-only transformer with RMSNorm
  pre-norm blocks, rotary position embeddings, grouped-query attention,
  SwiGLU feed-forward, weight-tied embedding/output head, no linear biases.
- `src/tokenizer.py` — word-level tokenizer fit on the training split only.
- `src/dataset.py` — chat formatting (`<bos><user>…<assistant>…<eos>`),
  response-only loss masking, and the raw-prose stream for pretraining.
- `src/train.py` — two-stage trainer: autoregressive pretraining (loss on
  every token) → supervised fine-tuning (loss on response tokens only),
  AdamW with cosine decay, gradient clipping, and best-validation-loss
  checkpoint selection. Reports masked cross-entropy, perplexity,
  teacher-forced token accuracy, and free-running exact match against a
  majority-response baseline.
- `src/autoresearch.py` — greedy coordinate-ascent hill climbing over
  `d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `sft_lr`, `dropout`.
  Every trial (accepted, rejected, or illegal) is logged to
  `artifacts/hillclimb.json`; the search never touches the test split.
- `src/infer.py` — shared generation code (temperature / top-k / top-p /
  greedy sampling) with a per-token trace (probability, entropy, top-5)
  used by both the CLI and the app.
- `app.py` — Streamlit front end with five tabs: **Chat** (with a
  generation-diagnostics panel), **Admin dashboard** (held-out metrics,
  training curves, per-intent accuracy, raw greedy decodes), **Autoresearch**
  (every hill-climbing trial), **Data** (corpus composition), **CRISP-DM**
  (this project's methodology document, rendered in-app).
- `CRISP_DM.md` — the six CRISP-DM phases mapped to this project's concrete
  decisions, including three data-quality bugs that were found from a bad
  held-out score and fixed at the generator, not patched at the model.

## Setup

```bash
cd 02_nano_llm_transformer
python3.13 -m venv .venv        # 3.13 recommended
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# 1. Generate the corpus (fast, seeded, deterministic)
python data/build_corpus.py

# 2. Train (two-stage: pretrain then SFT). ~3.5 min on Apple Silicon MPS,
#    longer on CPU-only.
python src/train.py --d-model 256 --n-layers 6 --n-heads 4 --n-kv-heads 2 \
    --sft-lr 3e-3 --dropout 0.0

# 3. (Optional) Re-run the architecture search that found those hyperparameters.
#    ~21 minutes on MPS — 29 short-budget trials.
python src/autoresearch.py

# 4. Launch the app
streamlit run app.py
```

```bash
# Quick CLI sanity check without the UI
python src/infer.py "what is k means ?" --greedy
```

## Approach

**Architecture** (`src/model.py`): pre-norm RMSNorm residual blocks, rotary
position embeddings, grouped-query attention (4 query heads sharing 2 KV
heads), a SwiGLU feed-forward block, and a weight-tied embedding/output head
— the same primitive family behind current Llama-class models, sized down to
run comfortably on a laptop GPU (Apple MPS) or CPU.

**Training** (`src/train.py`): stage 1 pretrains on ~3.2k tokens of
declarative domain prose with loss on every token, teaching vocabulary and
sentence shape; stage 2 fine-tunes the same weights on chat-formatted
(prompt, response) pairs with the loss masked to response tokens only,
teaching instruction-following. Both stages use AdamW with cosine LR decay
and warmup; the shipped checkpoint is the step with the best validation
loss.

**Autoresearch / hill climbing** (`src/autoresearch.py`): greedy
coordinate-ascent search over 6 hyperparameters, starting from
`d_model=128, n_layers=3, n_heads=4, n_kv_heads=2, sft_lr=1e-3, dropout=0.1`.
29 trials over two sweeps (~21 min on MPS) converged on **`d_model=256,
n_layers=6, n_heads=4, n_kv_heads=2, sft_lr=3e-3, dropout=0.0`**, cutting
validation loss from 0.0236 to 0.0040. That configuration is what ships.

**Data-quality fixes found via honest evaluation, not assumed correct**
(full detail in `CRISP_DM.md` §2–3):
1. `list` questions originally had 3 different sampled answers each — label
   noise capping that intent's exact match at ~33% however good the model
   was. Fixed by emitting one canonical answer per category.
2. Category answers read "backpropagation is a optimization method" —
   fixed with proper a/an agreement.
3. A global 80/10/10 shuffle-split could leave one `smalltalk` answer
   group's few phrasings entirely out of training by chance, so the model
   answered fluently but with a different group's reply. Fixed by
   stratifying per intent plus a repair pass guaranteeing every concept,
   comparison pair, category, and small-talk group has at least one
   training example.

## Honest results

From `models/metrics.json`, this exact shipped checkpoint (4,592,896
parameters, 648-word vocabulary, trained in 196.5s on Apple MPS):

| Metric | Validation | Test |
|---|---|---|
| Masked cross-entropy | 0.0007 | 0.0020 |
| Perplexity | 1.0007 | 1.0020 |
| Teacher-forced token accuracy | 99.98% | 99.96% |
| **Free-running exact match** | **99.5%** | **99.0%** |
| Majority-response baseline (exact match) | — | 3.0% |

Every intent is **100% exact match** on both splits except `smalltalk`
(1 correct out of 4 combined val+test examples). The model reliably tells
on-topic questions from off-topic ones — it never invents a fake
data-science definition for "hello" — but with only 6 small-talk answer
groups at 8–16 training examples each, it sometimes answers with the wrong
*one* of six equally fluent, memorized replies (e.g. answering "what can
you do?" with the self-introduction meant for "who are you?"). That is a
genuine, reported capacity/data-scale limit of a model this small, not a
hidden failure — see `CRISP_DM.md` §5 for the full analysis, including why
it wasn't chased further with more training cycles.

**What these numbers do and don't mean.** This measures compositional
generalization within a closed, templated grammar — held-out (concept,
phrasing) pairs the model never saw combined, but with every concept and
every phrasing style each seen separately during training. It is not a
test of open-domain language understanding, and the model correctly
refuses (100% exact match on `off_domain`) anything outside its trained
domain rather than guessing.

## Limitations

- Closed-domain by design: the model knows ~49 data-science concepts and
  nothing else. This is the explicit trade-off for training a multi-million
  parameter transformer from scratch on a laptop in minutes.
- Word-level tokenizer (not BPE): keeps sequences short and grammatical for
  this small a model and vocabulary, at the cost of not scaling to a larger,
  more varied vocabulary.
- `smalltalk` intent-selection accuracy is the one documented weak spot
  (see Honest results above).
- Synthetic, generated corpus — not a scraped or licensed dataset, and
  explicitly labeled as such throughout.
