"""Dataset acquisition for the Data Science Skills Mastery Lab.

Primary dataset: the "Telco Customer Churn" table (7,043 subscribers, 21
columns) — the same table published as a Kaggle competition dataset and, in
identical form, as an IBM sample-data file. The IBM mirror is used because it
needs no Kaggle account, which keeps `git clone && pip install && run`
reproducible.

If the download is unavailable (offline machine, blocked network), a seeded
synthetic table with the same schema is generated instead so every skill in
the lab still has something real to compute on. Which path was used is
recorded in ``data/source.txt`` and surfaced in the UI — the lab never
pretends synthetic rows are the real dataset.
"""
from __future__ import annotations

import io
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "telco_churn.csv"
SOURCE_PATH = DATA_DIR / "source.txt"
MIRROR_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
N_SYNTHETIC = 7043
SEED = 42


def _download() -> pd.DataFrame:
    with urllib.request.urlopen(MIRROR_URL, timeout=30) as resp:
        raw = resp.read()
    return pd.read_csv(io.BytesIO(raw))


def _synthesize(n: int = N_SYNTHETIC, seed: int = SEED) -> pd.DataFrame:
    """Seeded fallback with the same schema and roughly similar structure."""
    rng = np.random.default_rng(seed)

    def pick(values, probs=None):
        return rng.choice(values, size=n, p=probs)

    tenure = rng.integers(0, 73, size=n)
    contract = pick(
        ["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24]
    )
    internet = pick(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22])
    monthly = np.where(internet == "No", rng.normal(21, 5, n),
                       np.where(internet == "DSL", rng.normal(58, 15, n),
                                rng.normal(91, 16, n)))
    monthly = monthly.clip(18.25, 120.0).round(2)

    addon = ["Yes", "No", "No internet service"]
    def service(p_yes):
        out = pick(["Yes", "No"], [p_yes, 1 - p_yes]).astype(object)
        out[internet == "No"] = "No internet service"
        return out

    phone = pick(["Yes", "No"], [0.9, 0.1])
    multi = pick(["Yes", "No"], [0.45, 0.55]).astype(object)
    multi[phone == "No"] = "No phone service"

    # Churn propensity mirrors the real dataset's headline drivers: short
    # tenure, month-to-month contracts, fiber optic, electronic-check payment.
    payment = pick(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        [0.34, 0.23, 0.22, 0.21],
    )
    logit = (
        -1.1
        - 0.045 * tenure
        + 1.2 * (contract == "Month-to-month")
        - 0.5 * (contract == "Two year")
        + 0.7 * (internet == "Fiber optic")
        + 0.5 * (payment == "Electronic check")
        + 0.012 * (monthly - 65)
        + rng.normal(0, 0.6, n)
    )
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    total = (monthly * np.maximum(tenure, 0) * rng.normal(1.0, 0.03, n)).round(2)
    total_str = total.astype(object).astype(str)
    total_str[tenure == 0] = " "  # the real file stores blanks for new accounts

    return pd.DataFrame(
        {
            "customerID": [f"{i:04d}-SYNTH" for i in range(n)],
            "gender": pick(["Male", "Female"]),
            "SeniorCitizen": pick([0, 1], [0.84, 0.16]),
            "Partner": pick(["Yes", "No"], [0.48, 0.52]),
            "Dependents": pick(["Yes", "No"], [0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi,
            "InternetService": internet,
            "OnlineSecurity": service(0.4),
            "OnlineBackup": service(0.44),
            "DeviceProtection": service(0.44),
            "TechSupport": service(0.4),
            "StreamingTV": service(0.49),
            "StreamingMovies": service(0.49),
            "Contract": contract,
            "PaperlessBilling": pick(["Yes", "No"], [0.59, 0.41]),
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_str,
            "Churn": churn,
        }
    )


def ensure_dataset(force: bool = False) -> tuple[Path, str]:
    """Return (csv_path, source_label), downloading or synthesizing as needed."""
    if CSV_PATH.exists() and not force:
        source = SOURCE_PATH.read_text().strip() if SOURCE_PATH.exists() else "cached"
        return CSV_PATH, source

    try:
        df = _download()
        source = "real: IBM/Kaggle Telco Customer Churn mirror"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        df = _synthesize()
        source = f"synthetic fallback (download failed: {type(exc).__name__})"

    df.to_csv(CSV_PATH, index=False)
    SOURCE_PATH.write_text(source + "\n")
    return CSV_PATH, source


def load_raw(force: bool = False) -> pd.DataFrame:
    path, _ = ensure_dataset(force=force)
    return pd.read_csv(path)


def data_source() -> str:
    return SOURCE_PATH.read_text().strip() if SOURCE_PATH.exists() else "unknown"


if __name__ == "__main__":
    path, source = ensure_dataset(force=True)
    df = pd.read_csv(path)
    print(f"wrote {path}  rows={len(df)}  cols={df.shape[1]}")
    print(f"source: {source}")
