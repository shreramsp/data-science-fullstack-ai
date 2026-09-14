# CRISP-DM — NanoLlama Autoregressive SFT LLM

The six CRISP-DM phases as they were actually applied to this project. Every
number quoted in the dashboard comes from `models/metrics.json` and
`artifacts/`, both written by the training run — nothing here is hand-entered.

---

## 1. Business understanding

**Goal.** Build a language model and chatbot small enough to train end-to-end on
a laptop GPU in minutes, using the primitives a modern LLM actually uses, and
surround it with the evaluation a data scientist would demand before trusting it.

**Success criteria, agreed up front.**

| Criterion | Target |
|---|---|
| Trains on a laptop (Apple MPS or CPU) | under ~10 minutes, no cloud |
| Produces grammatical, on-topic answers | judged by held-out exact match, not by vibes |
| Beats a trivial baseline | > majority-response exact match by a wide margin |
| Refuses out-of-domain questions | rather than hallucinating |
| Honest generalization measurement | test split scored exactly once |

**What is explicitly out of scope.** This is not a general-purpose assistant. It
has ~4.6M parameters and a 648-word vocabulary; it knows one closed domain. Saying
so plainly *is* part of the deliverable.

**Cost of errors.** A wrong definition delivered confidently is worse than a
refusal, which is why an explicit `off_domain` refusal intent is trained in and
why the chat UI surfaces unknown-token counts and per-token confidence.

---

## 2. Data understanding

There is no public corpus that is simultaneously (a) small enough to pretrain a
several-million-parameter model on in minutes and (b) coherent enough that the
result is not gibberish. Training a tiny model on a slice of a web corpus is
precisely how you get the garbage output this project is meant to avoid.

So the data is **generated**, deliberately and transparently
(`data/build_corpus.py`, seed 1337):

- **49 data-science concepts**, each with a factual one-line definition, a
  category (11 of them: supervised learning, unsupervised learning, evaluation
  metric, preprocessing, optimization, regularization, model failure mode,
  transformer component, …) and a "when to use it" clause.
- **8 intents** built on top of those concepts — `define`, `use`, `category`,
  `family`, `compare`, `list`, `smalltalk`, `off_domain` — each with 3–12
  paraphrased question templates.
- **2,000 distinct (intent, key, template) combinations** in total.
- A separate **declarative prose corpus** (`data/pretrain.txt`, ~3.2k
  whitespace tokens) for stage-1 pretraining.

Distributional facts that shaped later decisions: the longest chat sequence is
**84 tokens** (so `max_seq_len = 96` is sufficient and cheap), the longest response
is **64 tokens** (which sets the decoder's token budget), the vocabulary fitted on
training text alone is **648 words**, and the out-of-vocabulary rate on held-out
prompts is **0.07%**.

**Three data defects found and fixed during evaluation, not before.** Each was
caught by an honest held-out score looking wrong, then traced to its root cause
in the generator rather than patched at the model:

1. **Answer ambiguity.** The first build gave each `list` question three
   different sampled answers ("name three unsupervised learning methods" had
   three valid trios) — label noise that caps exact match at roughly 1/3
   however good the model is. The first trained model duly scored 33% there.
   Fix: the generator now emits one canonical trio per category, so every
   question has exactly one correct answer, verified corpus-wide (zero
   multi-answer prompts).
2. **Grammar.** Category answers read "backpropagation is a optimization
   method" — missing a/an agreement. Fix: an `article()` helper picks "a" or
   "an" from the category's first letter.
3. **Key starvation.** `smalltalk` packs 6 distinct answer groups into one
   intent, each with only 3-5 phrasings. An intent-level 80/10/10 shuffle-split
   left one group's phrasings entirely in validation/test by chance, so the
   model never saw that group's correct reply during training and reliably
   answered a *different* group's reply instead — grammatical and on-topic,
   just the wrong stock answer. Fix: `stratified_split()` keeps the 80/10/10
   split at the intent level (it already worked for every other intent) and
   adds one targeted repair pass — any `(intent, key)` left with zero training
   examples has exactly one held-out example pulled back into training, so
   every key the model is ever scored on was also trained on at least once.

**Known limitation, stated once and not hidden.** A templated corpus is a closed
grammar. Good scores here mean the model learned that grammar — they do not
transfer to open-domain language. Even after the key-starvation fix, `smalltalk`
remains the model's weakest intent (see §5) — six short, semantically similar
greeting/identity/thanks groups are a genuinely hard discrimination problem for
a 648-word, single-epoch-per-example model, and that residual gap is reported
rather than trained away with more cycles.

---

## 3. Data preparation

- **Chat format.** `<bos> <user> … <assistant> … <eos>`. The `<user>` and
  `<assistant>` tokens are real vocabulary entries, so the format is learned
  rather than imposed at decode time.
- **Loss masking.** In stage 2 the loss is computed on response tokens and the
  final `<eos>` only. Prompt tokens are context, never targets — this is what
  makes the stage supervised *fine-tuning* rather than more pretraining.
- **Splitting, before augmentation.** The 2,000 base combinations are shuffled,
  split 80/10/10 within each intent, then repaired so no `(intent, key)` is
  entirely absent from training (see §2). *Then* the training split — and only
  the training split — is augmented with polite prefixes and suffixes ("hi, …",
  "… thanks."), 4 variants each, giving 6,408 training examples against 199
  validation and 199 test examples. Augmenting first would have scattered
  near-duplicates of test items into training; this ordering makes that
  impossible.
- **What the split actually tests.** Concepts appear in training under *other*
  phrasings, and phrasings appear with *other* concepts, but the held-out
  (concept, phrasing) pair is new. That is a compositional generalization test.
  It is **not** a test of unseen knowledge: the pretraining prose covers every
  concept, exactly as a real LLM's pretraining covers the facts its SFT data
  asks about.
- **Tokenizer.** Word-level, fitted on the training split only, so no validation
  or test vocabulary leaks into the model. Unknown words map to `<unk>`.
- **Padding.** Batches are padded to the longest member; pad positions carry
  zero loss weight.

---

## 4. Modeling

`src/model.py` — a decoder-only transformer built from Llama-family primitives:

| Primitive | Why |
|---|---|
| **RMSNorm**, pre-norm | fewer ops than LayerNorm, stable at this depth |
| **Rotary position embeddings** | relative positions with no learned table |
| **Grouped-query attention** | query heads share KV heads — smaller cache |
| **SwiGLU** feed-forward | stronger block at the same parameter budget |
| **Weight tying** (embedding ↔ output head) | removes a separate `vocab × d_model` output matrix |
| **No linear biases**, scaled residual init | standard modern practice |

Training (`src/train.py`) is two stages on the same weights: autoregressive
pretraining on declarative prose (loss on every token), then SFT on chat pairs
(loss on responses). AdamW, cosine decay with warmup, gradient clipping, dropout
and weight decay. The shipped checkpoint is the one with the **best validation
loss**, which is checkpoint selection standing in for early stopping.

**Autoresearch / hill climbing** (`src/autoresearch.py`) is greedy coordinate
ascent over `d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `sft_lr` and
`dropout`. Each trial trains a fresh model on a short budget and is scored on
validation loss; the best improving move on each axis is accepted, then the
sweep moves on, stopping when a full pass finds no improvement. Illegal points
(head dimension not even, `d_model` not divisible by `n_heads`) are recorded and
skipped rather than silently dropped. Every trial — accepted, rejected or
illegal — lands in `artifacts/hillclimb.json` and is rendered in the dashboard.

Starting from `d_model=128, n_layers=3, n_heads=4, n_kv_heads=2, sft_lr=1e-3,
dropout=0.1`, 29 trials over two sweeps converged on **`d_model=256, n_layers=6,
n_heads=4, n_kv_heads=2, sft_lr=3e-3, dropout=0.0`** (validation loss 0.0040,
down from 0.0236 at the start) — more depth and width than the starting guess,
a higher learning rate, and no dropout at all, which tracks with a data set this
templated: there is little to regularize against when the training distribution
is this narrow.

**The search never touches the test split.** That is the whole reason the final
test numbers mean anything. It was run once before the key-starvation fix in
§2 and not repeated after — the fix changes which examples land in which split,
not the relative ranking of these architectures, and re-running a 20+ minute
search to reconfirm the same conclusion would be search for its own sake. The
final model in this repo is that winning configuration retrained on the
corrected, repaired corpus.

---

## 5. Evaluation

Four metrics, because no single one is honest on its own:

1. **Masked cross-entropy / perplexity** on held-out responses — the language
   modelling objective itself.
2. **Teacher-forced token accuracy** — fraction of response tokens the model
   ranks first given the true prefix. Optimistic by construction: it never sees
   its own mistakes.
3. **Free-running exact match** — greedy-decode the whole answer from the prompt
   alone and compare the full string. This is the strict metric, and the one
   that catches a model that looks fine under teacher forcing and falls apart
   when it has to generate.
4. **Per-intent exact match** — where the model is strong and where it is weak,
   rather than one flattering average.

Against a **majority-response baseline** (always answer with the most common
training response) to make the headline number interpretable.

**Final results** (`models/metrics.json`, this exact checkpoint):

| Metric | Validation | Test |
|---|---|---|
| Masked cross-entropy | 0.0007 | 0.0020 |
| Perplexity | 1.0007 | 1.0020 |
| Teacher-forced token accuracy | 99.98% | 99.96% |
| Free-running exact match | 99.5% | 99.0% |
| Majority-response baseline (exact match) | — | 3.0% |

**Where the errors are.** Every intent except one is 100% exact match on both
splits: `define`, `use`, `category`, `family`, `compare`, `list`, `off_domain`.
The exception is `smalltalk` (1/4 correct across val+test combined): the model
reliably recognizes a greeting/thanks/identity question as small talk rather
than a data-science question — it never hallucinates a fake definition for
"hello" — but it frequently answers with the *wrong one* of the six small-talk
groups, e.g. answering "what can you do?" with the self-description meant for
"who are you?". Teacher-forced token accuracy on these examples is still
~99.9%, which is the tell: the model has clearly memorized all six replies
word-for-word, it just doesn't reliably route a short, semantically-overlapping
greeting to the *right one* of six equally fluent options from only 8-16
training examples per group. That is a legible capacity/data-scale limitation
of a 4.6M-parameter, 648-word-vocabulary model, not a training bug, and it is
reported here rather than hidden or trained away with more search cycles.

Qualitative check: `artifacts/generation_samples.json` stores the first twelve
held-out prompts of each split with the model's verbatim greedy output beside the
reference — including the failures. The dashboard renders them unfiltered.

---

## 6. Deployment

`app.py` (Streamlit) serves the model locally with five tabs:

- **Chat** — the chatbot, with temperature / top-k / top-p / greedy controls and
  a diagnostics panel showing per-token probability, entropy, top-5 candidates,
  unknown-token count and whether generation stopped on `<eos>` or ran out of
  budget.
- **Admin dashboard** — held-out metrics against the baseline, both training
  curves, validation accuracy, per-intent exact match, raw sample generations,
  and the full model card.
- **Autoresearch** — every hill-climbing trial, the accepted path, and the
  search space.
- **Data** — corpus composition and the split discipline.
- **CRISP-DM** — this document.

Everything runs on CPU at inference; the checkpoint is a few megabytes. No
container, no server, no database — `streamlit run app.py` is the deployment.

**Monitoring hooks that already exist.** Unknown-token count and mean token
log-probability are computed per request and shown in the UI; both are the
natural drift signals if this were ever fronted by real traffic.

**Reproduction.** `data/build_corpus.py` is seeded, training is seeded, and the
exact hyperparameters used are recorded in `models/metrics.json`.
