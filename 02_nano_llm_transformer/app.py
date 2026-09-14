"""NanoLlama — chatbot front end plus a data-science admin dashboard.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import infer  # noqa: E402

MODELS, ARTIFACTS, DATA = ROOT / "models", ROOT / "artifacts", ROOT / "data"

# Categorical slots 1-3 of the validated reference palette. Assigned in fixed
# order; never cycled. Charts stay single-axis and every chart has a table.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"

st.set_page_config(page_title="NanoLlama", page_icon="🦙", layout="wide")


@st.cache_resource(show_spinner="Loading NanoLlama…")
def get_model():
    return infer.load(device="cpu")


@st.cache_data
def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


metrics = read_json(MODELS / "metrics.json")
train_log = read_json(ARTIFACTS / "train_log.json")
hillclimb = read_json(ARTIFACTS / "hillclimb.json")
samples = read_json(ARTIFACTS / "generation_samples.json")
data_stats = read_json(DATA / "dataset_stats.json")

st.title("🦙 NanoLlama")
_size = f"{metrics['parameters'] / 1e6:.1f}M-parameter" if metrics else "laptop-scale"
st.caption(f"A {_size} Llama-style transformer trained from scratch on a "
           "closed data-science instruction corpus — chatbot + CRISP-DM admin dashboard.")

if not infer.artifacts_ready():
    st.error("No trained checkpoint found.")
    st.code("python data/build_corpus.py\npython src/train.py", language="bash")
    st.stop()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Decoding")
    greedy = st.toggle("Greedy (deterministic)", value=False,
                       help="Always take the most likely token. Best for factual answers.")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.05, disabled=greedy)
    top_k = st.slider("Top-k", 1, 100, 40, 1, disabled=greedy)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05, disabled=greedy)
    max_new = st.slider("Max new tokens", 8, 96, 72, 2,
                        help="Longest reference answer in the corpus is 64 tokens.")
    st.divider()
    if metrics:
        st.metric("Parameters", f"{metrics['parameters']:,}")
        st.caption(
            f"{metrics['config']['n_layers']} layers · {metrics['config']['d_model']} d_model · "
            f"{metrics['config']['n_heads']}Q/{metrics['config']['n_kv_heads']}KV heads · "
            f"vocab {metrics['config']['vocab_size']}"
        )
        st.caption(f"Trained {metrics['trained_at']} on {metrics['device']} "
                   f"in {metrics['train_seconds']:.0f}s")
    if st.button("Clear chat", width="stretch"):
        st.session_state.history = []
        st.rerun()

model, tok = get_model()

tab_chat, tab_admin, tab_research, tab_data, tab_crisp = st.tabs(
    ["💬 Chat", "📊 Admin dashboard", "🔬 Autoresearch", "🗂 Data", "📚 CRISP-DM"]
)

# --------------------------------------------------------------------------- chat
with tab_chat:
    st.session_state.setdefault("history", [])

    st.info("NanoLlama only knows the closed data-science corpus it was trained on. "
            "Try: *what is k means?* · *when should i use recall?* · "
            "*what is the difference between dropout and weight decay?* · "
            "*name three unsupervised learning methods.*", icon="💡")

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    prompt = st.chat_input("Ask about a data-science concept…")
    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Generating…"):
                result = infer.chat(model, tok, prompt, max_new_tokens=max_new,
                                    temperature=temperature, top_k=top_k, top_p=top_p,
                                    greedy=greedy)
            st.markdown(result["text"] or "_(empty generation)_")
            st.session_state.history.append({"role": "assistant", "content": result["text"]})

            with st.expander("Generation diagnostics"):
                a, b, c, d = st.columns(4)
                a.metric("Prompt tokens", result["prompt_tokens"])
                b.metric("Generated tokens", result["generated_tokens"])
                c.metric("Generation perplexity", f"{result['generation_perplexity']:.2f}")
                d.metric("Unknown prompt tokens", result["unknown_prompt_tokens"],
                         help=f"Words outside the {tok.vocab_size}-token vocabulary, "
                              "mapped to <unk>.")
                if not result["stopped_on_eos"]:
                    st.warning("Hit the token limit before <eos> — the answer may be cut off.")
                if result["unknown_prompt_tokens"]:
                    st.warning("Part of your question is outside the training vocabulary.")

                trace = pd.DataFrame([
                    {"step": t["step"], "token": t["token"],
                     "probability": t["probability"], "entropy (nats)": t["entropy_nats"],
                     "top 5": ", ".join(f"{x['token']} {x['probability']:.2f}"
                                        for x in t["top_5"])}
                    for t in result["trace"]
                ])
                st.markdown("**Per-token confidence** — probability the model assigned "
                            "to the token it emitted.")
                st.line_chart(trace.set_index("step")[["probability"]],
                              color=[C1], height=200)
                st.dataframe(trace, width="stretch", hide_index=True)

# --------------------------------------------------------------------------- admin
with tab_admin:
    if not metrics:
        st.warning("Run `python src/train.py` to populate the dashboard.")
    else:
        st.subheader("Held-out performance")
        st.caption("All numbers below come from `models/metrics.json`, written by the "
                   "training run. The test split was scored exactly once, after training.")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Test loss (masked CE)", f"{metrics['sft_test']['loss']:.3f}")
        k2.metric("Test perplexity", f"{metrics['sft_test']['perplexity']:.2f}")
        k3.metric("Test token accuracy", f"{metrics['sft_test']['token_accuracy']:.1%}")
        k4.metric("Test exact match", f"{metrics['generation_test']['exact_match']:.1%}",
                  delta=f"{metrics['generation_test']['exact_match'] - metrics['majority_response_baseline_exact_match']:+.1%} vs baseline")
        st.caption(f"Majority-response baseline exact match: "
                   f"{metrics['majority_response_baseline_exact_match']:.1%} — "
                   "the score from always replying with the single most common "
                   "training response.")

        split_tbl = pd.DataFrame([
            {"split": "validation",
             "loss": metrics["sft_val"]["loss"],
             "perplexity": metrics["sft_val"]["perplexity"],
             "token accuracy": metrics["sft_val"]["token_accuracy"],
             "exact match": metrics["generation_val"]["exact_match"],
             "examples": metrics["dataset"]["val"]},
            {"split": "test",
             "loss": metrics["sft_test"]["loss"],
             "perplexity": metrics["sft_test"]["perplexity"],
             "token accuracy": metrics["sft_test"]["token_accuracy"],
             "exact match": metrics["generation_test"]["exact_match"],
             "examples": metrics["dataset"]["test"]},
        ])
        st.dataframe(split_tbl, width="stretch", hide_index=True)

        st.divider()
        st.subheader("Training curves")
        if train_log:
            log = pd.DataFrame(train_log)
            left, right = st.columns(2)
            with left:
                st.markdown("**Stage 1 — autoregressive pretraining** (loss on every token)")
                pre = log[log.stage == "pretrain"].set_index("step")
                st.line_chart(pre[["train_loss", "val_loss"]].rename(
                    columns={"train_loss": "train", "val_loss": "validation"}),
                    color=[C1, C2], height=260)
            with right:
                st.markdown("**Stage 2 — supervised fine-tuning** (loss on response tokens)")
                sft = log[log.stage == "sft"].set_index("step")
                st.line_chart(sft[["train_loss", "val_loss"]].rename(
                    columns={"train_loss": "train", "val_loss": "validation"}),
                    color=[C1, C2], height=260)
            st.markdown("**Validation response-token accuracy during fine-tuning**")
            st.line_chart(sft[["val_token_accuracy"]].rename(
                columns={"val_token_accuracy": "validation token accuracy"}),
                color=[C1], height=220)
            st.caption(
                f"Best validation loss was at step {metrics['best_checkpoint_step']}; "
                "that checkpoint is the one that ships (checkpoint selection stands "
                "in for early stopping). Stage 1 ended at validation loss "
                f"{metrics['pretrain_stage_end_val_loss']:.3f} on the prose stream; "
                f"the same model scores {metrics['pretrain_val_after_sft']['loss']:.3f} "
                "there after fine-tuning — SFT deliberately moves the model off the "
                "pretraining distribution and onto the chat one."
            )
            with st.expander("Raw training log"):
                st.dataframe(log, width="stretch", hide_index=True)
        else:
            st.info("No training log found.")

        st.divider()
        st.subheader("Exact match by intent (test split)")
        by_intent = metrics["generation_test"]["by_intent"]
        intent_df = pd.DataFrame([
            {"intent": k, "exact match": v["exact_match"], "examples": v["n"]}
            for k, v in by_intent.items()
        ]).sort_values("exact match")
        st.bar_chart(intent_df.set_index("intent")[["exact match"]],
                     color=[C1], horizontal=True, height=300)
        st.dataframe(intent_df, width="stretch", hide_index=True)

        if samples:
            st.divider()
            st.subheader("Greedy decodes on held-out prompts")
            st.caption("Verbatim model output next to the reference answer — every "
                       "failure is shown first, then the earliest successes fill the rest.")
            for split in ("test", "val"):
                n_fail = metrics[f"generation_{split}"].get("n_failures", 0)
                st.markdown(f"**{split} split** — {n_fail} failure"
                           f"{'s' if n_fail != 1 else ''} out of "
                           f"{metrics[f'generation_{split}']['n']}")
                st.dataframe(pd.DataFrame(samples[split]), width="stretch",
                             hide_index=True)

        st.divider()
        st.subheader("Model card")
        st.json({"architecture": metrics["config"],
                 "parameters": metrics["parameters"],
                 "parameters_excluding_embedding": metrics["parameters_non_embedding"],
                 "training": metrics["hyperparameters"],
                 "device": metrics["device"],
                 "train_seconds": metrics["train_seconds"]}, expanded=False)

# --------------------------------------------------------------------------- autoresearch
with tab_research:
    st.subheader("Hill-climbing architecture search")
    if not hillclimb:
        st.info("Run `python src/autoresearch.py` to populate this tab.")
    else:
        st.markdown(
            f"**Method:** {hillclimb['method']}  \n"
            f"**Objective:** {hillclimb['objective']}  \n"
            f"**Budget per trial:** {hillclimb['budget_per_trial']}  \n"
            f"**Trials evaluated:** {hillclimb['n_trials_evaluated']} in "
            f"{hillclimb['total_seconds']:.0f}s on {hillclimb['device']}"
        )
        a, b = st.columns(2)
        a.markdown("**Starting point**")
        a.json(hillclimb["start"])
        b.markdown("**Best point found**")
        b.json(hillclimb["best_point"])

        evaluated = [t for t in hillclimb["trials"] if t["decision"] != "illegal"]
        tdf = pd.DataFrame([{
            "trial": t["trial"], "axis": t["axis"], "decision": t["decision"],
            "val loss": t["val_loss"], "val token accuracy": t["val_token_accuracy"],
            "parameters": t["parameters"], "seconds": t["seconds"],
            **{f"{k}": v for k, v in t["point"].items()},
        } for t in evaluated])

        st.markdown("**Validation loss per trial** — colour marks whether the move was "
                    "accepted into the incumbent.")
        st.scatter_chart(tdf, x="trial", y="val loss", color="decision",
                         height=320)
        st.caption("Search touched the validation split only; the test split was held "
                   "back for the final run, so the reported test numbers are not "
                   "contaminated by the search.")
        st.dataframe(tdf, width="stretch", hide_index=True)

        with st.expander("Search space"):
            st.json(hillclimb["search_space"])

# --------------------------------------------------------------------------- data
with tab_data:
    st.subheader("Corpus")
    if not data_stats:
        st.info("Run `python data/build_corpus.py`.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Concepts", data_stats["n_concepts"])
        b.metric("Base combinations", f"{data_stats['base_combinations']:,}")
        c.metric("Train examples", f"{data_stats['train_examples']:,}")
        d.metric("Vocabulary", metrics["dataset"]["vocab_size"] if metrics else "—")
        st.markdown(
            "Splits are made over **(intent, concept, question-template) combinations "
            "before augmentation**, and only the training split is augmented. "
            "Validation and test therefore hold question phrasings the model never saw "
            "applied to that concept."
        )
        intent_df = pd.DataFrame(
            [{"intent": k, "base combinations": v}
             for k, v in data_stats["intent_counts"].items()]
        ).sort_values("base combinations")
        st.markdown("**Base combinations by intent**")
        st.bar_chart(intent_df.set_index("intent")[["base combinations"]],
                     color=[C1], horizontal=True, height=280)
        st.dataframe(intent_df, width="stretch", hide_index=True)
        st.json(data_stats, expanded=False)

# --------------------------------------------------------------------------- crisp-dm
with tab_crisp:
    doc = ROOT / "CRISP_DM.md"
    if doc.exists():
        st.markdown(doc.read_text())
    else:
        st.info("CRISP_DM.md not found.")
