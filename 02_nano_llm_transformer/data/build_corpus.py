"""Build the NanoLlama training corpus.

CRISP-DM "Data Understanding" + "Data Preparation" for this project.

There is no public dataset small enough to pretrain a laptop-scale LLM on and
still get coherent output, so this script *generates* a seeded, closed-domain
instruction corpus about data-science concepts. Two artifacts come out:

  1. data/pretrain.txt      -- declarative prose, used for stage-1 autoregressive
                               language-model pretraining (loss on every token).
  2. data/sft_{split}.jsonl -- (prompt, response) pairs for stage-2 supervised
                               fine-tuning (loss on response tokens only).

Split discipline: the (intent, concept, template) combinations are split
*before* any augmentation, and only the train split is augmented. Validation
and test therefore contain question phrasings the model never saw applied to
that concept -- a compositional generalization test, not a memorization test.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 1337
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Knowledge base. Every definition is a plain, factual one-liner; nothing here
# is invented performance data.
#   name, category, definition, use_case
# --------------------------------------------------------------------------
CONCEPTS: list[tuple[str, str, str, str]] = [
    ("linear regression", "supervised learning",
     "a model that fits a straight line through the data by minimizing squared error",
     "you need a fast interpretable baseline for a numeric target"),
    ("logistic regression", "supervised learning",
     "a linear model that turns a weighted sum of features into a class probability",
     "you want calibrated probabilities from a simple classifier"),
    ("decision tree", "supervised learning",
     "a model that splits the data with a sequence of threshold rules on single features",
     "you need a model a business reader can follow rule by rule"),
    ("random forest", "supervised learning",
     "an ensemble that averages many decision trees grown on bootstrap samples",
     "you want a strong tabular baseline that needs almost no tuning"),
    ("gradient boosting", "supervised learning",
     "an ensemble that adds trees one at a time so each tree fits the errors of the ones before it",
     "you want the best accuracy you can get on tabular data"),
    ("support vector machine", "supervised learning",
     "a classifier that looks for the widest margin between the classes",
     "you have a small clean data set with many features"),
    ("k nearest neighbors", "supervised learning",
     "a model that predicts from the labels of the closest training points",
     "the decision boundary is irregular and the data set is small"),
    ("naive bayes", "supervised learning",
     "a probabilistic classifier that applies bayes rule and assumes the features are independent",
     "you need a very fast text classifier"),
    ("neural network", "supervised learning",
     "a stack of linear layers and nonlinear activations trained by backpropagation",
     "the pattern is complex and you have enough data to fit it"),
    ("k means", "unsupervised learning",
     "an algorithm that partitions points into k clusters around their centroids",
     "you want compact round segments and you can pick k yourself"),
    ("dbscan", "unsupervised learning",
     "a clustering algorithm that grows clusters out of dense regions and leaves the rest as noise",
     "the clusters have odd shapes and the data contains noise"),
    ("hierarchical clustering", "unsupervised learning",
     "a method that builds a tree of nested clusters by repeatedly merging the closest groups",
     "you want to inspect segments at several levels of detail"),
    ("principal component analysis", "unsupervised learning",
     "a method that projects the data onto the directions that carry the most variance",
     "you need fewer correlated features before modeling"),
    ("isolation forest", "unsupervised learning",
     "an anomaly detector that scores a point by how few random splits it takes to isolate it",
     "you need unlabeled outlier detection on tabular data"),
    ("apriori", "unsupervised learning",
     "an algorithm that mines frequent item sets and turns them into association rules",
     "you want to know which products are bought together"),
    ("accuracy", "evaluation metric",
     "the fraction of predictions that are correct",
     "the classes are balanced and every error costs the same"),
    ("precision", "evaluation metric",
     "the fraction of predicted positives that really are positive",
     "a false alarm is expensive"),
    ("recall", "evaluation metric",
     "the fraction of the real positives that the model actually finds",
     "a missed positive is expensive"),
    ("f1 score", "evaluation metric",
     "the harmonic mean of precision and recall",
     "you need one number that balances false alarms and misses"),
    ("roc auc", "evaluation metric",
     "the chance that the model ranks a random positive above a random negative",
     "you care about ranking quality rather than one threshold"),
    ("mean squared error", "evaluation metric",
     "the average of the squared gaps between predictions and targets",
     "you are scoring a regression model and large errors should hurt more"),
    ("perplexity", "evaluation metric",
     "the exponential of the average cross entropy , so lower is better",
     "you are scoring a language model on held out text"),
    ("confusion matrix", "evaluation metric",
     "a table of true positives , false positives , true negatives and false negatives",
     "you want to see which kind of mistake the model makes"),
    ("cross validation", "evaluation practice",
     "a procedure that rotates the validation fold so every row gets scored exactly once",
     "the data set is small and a single split would be noisy"),
    ("train test split", "evaluation practice",
     "holding part of the data out of training so the score is measured on unseen rows",
     "you need an honest estimate before you ship a model"),
    ("holdout set", "evaluation practice",
     "a slice of data that is touched only once at the very end",
     "you have tuned many times and need an untouched final score"),
    ("one hot encoding", "preprocessing",
     "turning a categorical column into one binary column per category",
     "a model needs numbers but the column holds unordered labels"),
    ("standard scaling", "preprocessing",
     "rescaling a column to zero mean and unit variance",
     "the model is sensitive to the scale of the features"),
    ("missing value imputation", "preprocessing",
     "filling gaps in a column with a statistic or a learned estimate",
     "rows have holes and dropping them would throw away signal"),
    ("feature engineering", "preprocessing",
     "building new columns that expose the structure a model cannot find on its own",
     "the raw columns hide the relationship you care about"),
    ("tokenization", "preprocessing",
     "cutting raw text into the discrete units a language model actually sees",
     "you are preparing text for a language model"),
    ("gradient descent", "optimization",
     "an algorithm that walks the parameters downhill along the gradient of the loss",
     "you are fitting any model that has a differentiable loss"),
    ("adam", "optimization",
     "a gradient descent variant that keeps a running average of the gradient and its square",
     "you want a robust default optimizer for a neural network"),
    ("learning rate", "optimization",
     "the step size that decides how far the parameters move on each update",
     "training diverges or crawls and you need to fix the step size"),
    ("backpropagation", "optimization",
     "the chain rule applied backwards through a network to get every gradient in one pass",
     "you are training any neural network"),
    ("dropout", "regularization",
     "randomly zeroing activations during training so the network cannot lean on one path",
     "a neural network memorizes the training set"),
    ("weight decay", "regularization",
     "adding a penalty on the size of the weights to the loss",
     "the model is large relative to the data"),
    ("early stopping", "regularization",
     "halting training at the step where validation loss stops improving",
     "training loss keeps falling but validation loss has turned around"),
    ("overfitting", "model failure mode",
     "the model has learned the noise in the training set instead of the signal",
     "training loss is far below validation loss"),
    ("underfitting", "model failure mode",
     "the model is too simple to capture the pattern that is really there",
     "training loss and validation loss are both high"),
    ("data leakage", "model failure mode",
     "information from the target or from the future sneaking into the features",
     "an offline score looks too good to be true"),
    ("class imbalance", "model failure mode",
     "one class being so rare that accuracy rewards ignoring it",
     "the positive class is a small fraction of the rows"),
    ("attention", "transformer component",
     "a mechanism that lets every token read a weighted mixture of the other tokens",
     "the model has to relate tokens that sit far apart"),
    ("rotary position embedding", "transformer component",
     "rotating the query and key vectors by an angle that depends on position",
     "you want relative position information without a learned position table"),
    ("rms norm", "transformer component",
     "normalizing a vector by its root mean square and rescaling it",
     "you want layer normalization with fewer operations"),
    ("swiglu", "transformer component",
     "a feed forward block that gates one linear projection with a swish activation of another",
     "you want a stronger feed forward block at the same parameter budget"),
    ("grouped query attention", "transformer component",
     "letting several query heads share one key and value head",
     "you need to shrink the memory the attention cache uses"),
    ("supervised fine tuning", "training stage",
     "training a pretrained model on prompt and response pairs with the loss on the response only",
     "a base model predicts text well but does not follow instructions"),
    ("cross entropy loss", "objective",
     "the negative log probability the model assigned to the correct token",
     "you are training any classifier or language model"),
]

CONCEPT_BY_NAME = {c[0]: c for c in CONCEPTS}

DEFINE_TEMPLATES = [
    "what is {x} ?",
    "can you explain {x} ?",
    "define {x} .",
    "tell me about {x} .",
    "what does {x} mean ?",
    "explain {x} to me .",
    "give me a definition of {x} .",
    "i do not understand {x} .",
    "how would you describe {x} ?",
    "what exactly is {x} ?",
    "could you describe {x} ?",
    "help me understand {x} .",
]

USE_TEMPLATES = [
    "when should i use {x} ?",
    "when would i use {x} ?",
    "why would i use {x} ?",
    "what is {x} good for ?",
    "what problem does {x} solve ?",
    "when does {x} make sense ?",
]

CATEGORY_TEMPLATES = [
    "is {x} supervised or unsupervised ?",
    "what kind of method is {x} ?",
    "what category does {x} belong to ?",
    "where does {x} fit in ?",
    "how would you classify {x} ?",
]

FAMILY_TEMPLATES = [
    "what is related to {x} ?",
    "what else is like {x} ?",
    "what should i compare {x} with ?",
    "name something similar to {x} .",
]

COMPARE_TEMPLATES = [
    "what is the difference between {x} and {y} ?",
    "how does {x} differ from {y} ?",
    "compare {x} and {y} .",
    "{x} or {y} ?",
]

LIST_TEMPLATES = [
    "name three {c} methods .",
    "give me three examples of {c} .",
    "list some {c} techniques .",
    "what are some {c} methods ?",
]

SMALLTALK = [
    (["hello .", "hi .", "hey there .", "good morning .", "hi there ."],
     "hello . i am nanollama , a small language model that answers questions about data science ."),
    (["who are you ?", "what are you ?", "what is your name ?", "tell me about yourself ."],
     "i am nanollama , a tiny transformer trained from scratch on a data science instruction corpus ."),
    (["what can you do ?", "how can you help me ?", "what do you know ?", "what topics do you cover ?"],
     "i can define data science concepts , say when to use them , compare two of them , and list methods by category ."),
    (["thank you .", "thanks .", "that helps .", "thanks a lot ."],
     "you are welcome . ask me about another data science concept whenever you like ."),
    (["goodbye .", "bye .", "see you later .", "that is all ."],
     "goodbye . good luck with your data science project ."),
    (["how were you trained ?", "what are you trained on ?", "where does your knowledge come from ?",
      "what data were you trained on ?"],
     "i was pretrained on a small generated data science corpus and then fine tuned on prompt and response pairs ."),
]

OFF_DOMAIN = [
    "the weather tomorrow", "the football score", "a recipe for pasta", "the stock market",
    "who won the election", "how to fix my car", "the best movie this year", "my flight booking",
    "a poem about the sea", "the capital of france", "how to lose weight", "my horoscope",
    "the news today", "a joke", "my tax return", "the price of gold",
]
OFF_DOMAIN_TEMPLATES = [
    "tell me about {t} .",
    "what do you know about {t} ?",
    "can you help me with {t} ?",
    "i want to ask about {t} .",
]
OFF_DOMAIN_RESPONSE = (
    "i only answer questions about data science concepts , so i cannot help with that ."
)

# Polite wrappers used to augment the *training* split only.
PREFIXES = ["", "hi , ", "hey , ", "please , ", "quick question , ", "ok , "]
SUFFIXES = ["", " thanks .", " please .", " thank you ."]


def article(word: str) -> str:
    """a/an agreement -- every category in this corpus follows the vowel rule."""
    return "an" if word[0] in "aeiou" else "a"


def build_pairs() -> list[dict]:
    """Enumerate every (intent, key, template) combination exactly once."""
    rng = random.Random(SEED)
    pairs: list[dict] = []

    for name, category, definition, use_case in CONCEPTS:
        for i, tpl in enumerate(DEFINE_TEMPLATES):
            pairs.append(dict(intent="define", key=name, template=i,
                              prompt=tpl.format(x=name),
                              response=f"{name} is {definition} ."))
        for i, tpl in enumerate(USE_TEMPLATES):
            pairs.append(dict(intent="use", key=name, template=i,
                              prompt=tpl.format(x=name),
                              response=f"use {name} when {use_case} ."))
        for i, tpl in enumerate(CATEGORY_TEMPLATES):
            pairs.append(dict(intent="category", key=name, template=i,
                              prompt=tpl.format(x=name),
                              response=f"{name} is {article(category)} {category} method ."))

    # Same-family neighbours, taken deterministically from the category listing.
    by_cat: dict[str, list[str]] = {}
    for name, category, _, _ in CONCEPTS:
        by_cat.setdefault(category, []).append(name)
    for name, category, _, _ in CONCEPTS:
        siblings = [s for s in by_cat[category] if s != name]
        if len(siblings) < 2:
            continue
        a, b = siblings[0], siblings[1]
        for i, tpl in enumerate(FAMILY_TEMPLATES):
            pairs.append(dict(intent="family", key=name, template=i,
                              prompt=tpl.format(x=name),
                              response=f"{name} sits in the same family as {a} and {b} ."))

    # Comparisons: sample concept pairs, always store the pair in a fixed order.
    names = [c[0] for c in CONCEPTS]
    seen: set[tuple[str, str]] = set()
    while len(seen) < 140:
        x, y = rng.sample(names, 2)
        seen.add((x, y) if x < y else (y, x))
    for x, y in sorted(seen):
        cx, cy = CONCEPT_BY_NAME[x][1], CONCEPT_BY_NAME[y][1]
        dx, dy = CONCEPT_BY_NAME[x][2], CONCEPT_BY_NAME[y][2]
        if cx == cy:
            resp = (f"both are {cx} ideas . {x} is {dx} , while {y} is {dy} .")
        else:
            resp = (f"{x} is {article(cx)} {cx} idea and {y} is {article(cy)} {cy} idea . "
                    f"{x} is {dx} , while {y} is {dy} .")
        for i, tpl in enumerate(COMPARE_TEMPLATES):
            pairs.append(dict(intent="compare", key=f"{x}|{y}", template=i,
                              prompt=tpl.format(x=x, y=y), response=resp))

    # Category listings.
    # One canonical trio per category. Sampling several trios per category would
    # give the same question more than one valid answer -- label noise that caps
    # exact match at 1/n_trios for this intent no matter how good the model is.
    for category, members in sorted(by_cat.items()):
        if len(members) < 3:
            continue
        trio = members[:3]
        resp = (f"three {category} methods are {trio[0]} , {trio[1]} and {trio[2]} .")
        for i, tpl in enumerate(LIST_TEMPLATES):
            pairs.append(dict(intent="list", key=category, template=i,
                              prompt=tpl.format(c=category), response=resp))

    for group_id, (prompts, response) in enumerate(SMALLTALK):
        for i, p in enumerate(prompts):
            pairs.append(dict(intent="smalltalk", key=f"group{group_id}", template=i,
                              prompt=p, response=response))

    for topic in OFF_DOMAIN:
        for i, tpl in enumerate(OFF_DOMAIN_TEMPLATES):
            pairs.append(dict(intent="off_domain", key=topic, template=i,
                              prompt=tpl.format(t=topic), response=OFF_DOMAIN_RESPONSE))

    return pairs


def augment(pairs: list[dict], rng: random.Random, variants: int) -> list[dict]:
    """Wrap training prompts in polite prefixes/suffixes. Train split only."""
    out: list[dict] = []
    for p in pairs:
        combos = {("", "")}
        while len(combos) < variants:
            combos.add((rng.choice(PREFIXES), rng.choice(SUFFIXES)))
        for pre, suf in sorted(combos):
            q = (pre + p["prompt"] + suf).strip()
            out.append({**p, "prompt": q})
    rng.shuffle(out)
    return out


def build_pretrain_text(rng: random.Random) -> str:
    """Declarative prose for stage-1 pretraining (no chat formatting)."""
    blocks: list[str] = []
    for name, category, definition, use_case in CONCEPTS:
        blocks.append(
            f"{name} is {definition} . it is {article(category)} {category} idea . "
            f"use {name} when {use_case} . a data scientist reaches for {name} "
            f"as part of a {category} workflow ."
        )
    extra = [
        "a data science project moves through business understanding , data understanding , "
        "data preparation , modeling , evaluation and deployment .",
        "a model is only as honest as the split it was scored on .",
        "training loss measures fit , validation loss measures generalization .",
        "a language model predicts the next token given every token before it .",
        "lower cross entropy on held out text means the model is less surprised by it .",
        "the same preprocessing must be applied to training data and to live traffic .",
        "a baseline you can explain beats a black box you cannot check .",
        "every metric answers a different question , so pick the one that matches the cost .",
        "a transformer block mixes tokens with attention and mixes features with a feed forward layer .",
        "pretraining teaches the model language , fine tuning teaches it the task .",
    ]
    blocks.extend(extra * 4)
    rng.shuffle(blocks)
    return "\n".join(blocks) + "\n"


def stratified_split(pairs: list[dict], val_frac: float = 0.10,
                      test_frac: float = 0.10) -> tuple[list[dict], list[dict], list[dict]]:
    """Split each intent 80/10/10 independently, then repair any *key* left
    with zero training examples.

    `key` is the specific concept / comparison pair / category / small-talk
    group whose exact answer the model has to produce -- e.g. each of the 6
    small-talk answer groups is its own key. Intent-level stratification is
    enough for `define`/`use`/`category`/`family`/`compare`, where every key
    already has 4-12 phrasings, so a handful land in training by chance
    regardless of split granularity. It is not enough for `smalltalk`: with
    only 4-5 phrasings per group, one group's phrasings could all land in
    val/test purely by chance, and the model then answered fluently but with
    the wrong stock reply (it had never seen that group's correct answer).
    Stratifying at key granularity instead would fix that, but with most
    keys already this small, it inflates val/test to ~18%/18% and starves
    training for every intent, not just the one that needed it. So: keep the
    80/10/10 intent-level split, then run one targeted repair -- any key with
    zero examples in the training split gets exactly one of its held-out
    examples moved back into training (preferring to deplete validation
    before test, so the test measurement stays intact wherever possible).
    """
    by_intent: dict[str, list[dict]] = {}
    for p in pairs:
        by_intent.setdefault(p["intent"], []).append(p)

    train, val, test = [], [], []
    for rows in by_intent.values():          # `pairs` was pre-shuffled by the caller
        n = len(rows)
        n_test = max(1, round(test_frac * n))
        n_val = max(1, round(val_frac * n))
        if n >= 3 and n_test + n_val >= n:   # always leave at least one for training
            n_test, n_val = 1, 1
        test += rows[:n_test]
        val += rows[n_test:n_test + n_val]
        train += rows[n_test + n_val:]

    train_keys = {(p["intent"], p["key"]) for p in train}
    all_keys = {(p["intent"], p["key"]) for p in pairs}
    for key in sorted(all_keys - train_keys):
        for bucket in (val, test):           # deplete val first, protect test
            for i, p in enumerate(bucket):
                if (p["intent"], p["key"]) == key:
                    train.append(bucket.pop(i))
                    break
            else:
                continue
            break
    return train, val, test


def main() -> None:
    rng = random.Random(SEED)
    pairs = build_pairs()
    rng.shuffle(pairs)

    train_base, val, test = stratified_split(pairs)
    train = augment(train_base, rng, variants=4)

    for split, rows in (("train", train), ("val", val), ("test", test)):
        path = HERE / f"sft_{split}.jsonl"
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    pretrain = build_pretrain_text(rng)
    (HERE / "pretrain.txt").write_text(pretrain)

    intents: dict[str, int] = {}
    for r in pairs:
        intents[r["intent"]] = intents.get(r["intent"], 0) + 1
    intents_by_split: dict[str, dict[str, int]] = {}
    for split_name, rows in (("train_base", train_base), ("val", val), ("test", test)):
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["intent"]] = counts.get(r["intent"], 0) + 1
        intents_by_split[split_name] = dict(sorted(counts.items()))
    stats = {
        "seed": SEED,
        "n_concepts": len(CONCEPTS),
        "n_categories": len({c[1] for c in CONCEPTS}),
        "base_combinations": len(pairs),
        "train_examples": len(train),
        "train_base_combinations": len(train_base),
        "val_examples": len(val),
        "test_examples": len(test),
        "augmentation_variants_per_train_combination": 4,
        "split_method": ("stratified per intent (80/10/10 within each intent), "
                        "plus a repair pass guaranteeing every key has >=1 "
                        "training example"),
        "intent_counts": dict(sorted(intents.items())),
        "intent_counts_by_split": intents_by_split,
        "pretrain_characters": len(pretrain),
        "pretrain_whitespace_tokens": len(pretrain.split()),
    }
    (HERE / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
