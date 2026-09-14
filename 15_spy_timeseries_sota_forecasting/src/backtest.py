"""Simple long/flat directional trading backtest driven by the horizon-1
median (P50) forecast, evaluated on the out-of-sample test period only.

Rule: go long (position = 1) when the predicted next-day log return is
above `threshold`; otherwise stay flat (position = 0). No shorting, no
leverage. A small slippage cost is charged only on days the position
changes, to approximate transaction cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
SLIPPAGE_BPS = 2.0  # basis points, charged only when the position flips


def run_backtest(actual_return: np.ndarray, pred_return: np.ndarray, threshold: float = 0.0) -> pd.DataFrame:
    position = (pred_return > threshold).astype(float)
    position_change = np.abs(np.diff(position, prepend=0.0))
    slippage = position_change * (SLIPPAGE_BPS / 10_000)
    strategy_return = position * actual_return - slippage
    return pd.DataFrame({
        "position": position,
        "actual_return": actual_return,
        "strategy_return": strategy_return,
        "buy_hold_return": actual_return,
    })


def summarize(bt: pd.DataFrame) -> dict:
    strat = bt["strategy_return"].to_numpy()
    bh = bt["buy_hold_return"].to_numpy()

    def _stats(r: np.ndarray) -> dict:
        cum_curve = np.cumprod(1 + r)
        total_return = float(cum_curve[-1] - 1)
        ann_return = float(cum_curve[-1] ** (TRADING_DAYS / len(r)) - 1) if len(r) else 0.0
        ann_vol = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        downside = r[r < 0]
        downside_vol = float(np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 0.0
        sortino = ann_return / downside_vol if downside_vol > 0 else 0.0
        running_max = np.maximum.accumulate(cum_curve)
        drawdown = cum_curve / running_max - 1
        max_drawdown = float(drawdown.min())
        return {
            "total_return": total_return, "annualized_return": ann_return,
            "annualized_vol": ann_vol, "sharpe_ratio": sharpe, "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
        }

    hit_rate = float(np.mean(np.sign(bt["position"] - 0.5) * np.sign(bt["actual_return"]) > 0))
    return {"strategy": _stats(strat), "buy_and_hold": _stats(bh), "directional_hit_rate": hit_rate}
