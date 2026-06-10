from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


ENTRY_REASON = "IndustryテーマEP後10MA押し目維持"
ENTRY_PRIORITY = {ENTRY_REASON: 100}


@dataclass(frozen=True)
class HybridSwingConfig:
    """industry_theme_ep_ex_biotech / pullback10 / stage2_or_atr8."""

    min_price: float = 1.0
    min_avg_dollar_volume20: float = 1_000_000.0
    ep_relvol20: float = 2.0
    ep_gap: float = 0.08
    ep_ret1: float = 0.12
    ep_breakout_ret1: float = 0.06
    ep_close_location: float = 0.55
    industry_theme_ret60: float = 0.10
    industry_residual_ret60: float = 0.15
    excluded_industry_keyword: str = "Biotechnology"
    ep_watch_days: int = 20
    signal_cooldown_days: int = 20
    pullback_sma10_buffer: float = 1.03
    max_pullback_drop_from_ep_close: float = 0.80
    atr_trailing_multiple: float = 8.0


def normalize_daily_bars(daily_bars: pd.DataFrame) -> pd.DataFrame:
    if daily_bars.empty:
        return pd.DataFrame()
    work = daily_bars.copy()
    work.columns = [str(col).lower() for col in work.columns]
    if "ts" in work.columns and "date" not in work.columns:
        work = work.rename(columns={"ts": "date"})
    required = ["symbol", "date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in work.columns]
    if missing:
        raise ValueError(f"daily_bars missing columns: {missing}")
    work["symbol"] = work["symbol"].astype(str).str.upper().str.strip()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "volume", "adj_close", "raw_close"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"])
    work["raw_close"] = pd.to_numeric(work.get("raw_close", work["close"]), errors="coerce").fillna(work["close"])
    return (
        work.sort_values(["symbol", "date"], kind="mergesort")
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )


def _true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _wilder_atr14(frame: pd.DataFrame) -> pd.Series:
    tr = _true_range(frame).to_numpy(float)
    out = np.full(len(tr), np.nan, dtype=float)
    period = 14
    atr = 0.0
    count = 0
    for i, value in enumerate(tr):
        if not np.isfinite(value):
            continue
        count += 1
        if count <= period:
            atr += value
            if count == period:
                atr /= period
                out[i] = atr
        else:
            atr = ((period - 1) * atr + value) / period
            out[i] = atr
    return pd.Series(out, index=frame.index)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _close_location(frame: pd.DataFrame) -> pd.Series:
    return (frame["close"] - frame["low"]) / (frame["high"] - frame["low"]).replace(0, np.nan)


def build_hybrid_features(
    daily_bars: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    *,
    cfg: HybridSwingConfig | None = None,
    spy_symbol: str = "SPY",
) -> pd.DataFrame:
    cfg = cfg or HybridSwingConfig()
    work = normalize_daily_bars(daily_bars)
    if work.empty:
        return work

    if universe is not None and not universe.empty:
        meta = universe.copy()
        meta.columns = [str(col).lower() for col in meta.columns]
        if "symbol" in meta.columns:
            meta["symbol"] = meta["symbol"].astype(str).str.upper().str.strip()
            keep = [col for col in ["symbol", "sector", "industry", "market_cap"] if col in meta.columns]
            work = work.merge(meta[keep].drop_duplicates("symbol"), on="symbol", how="left")
    for col in ["sector", "industry"]:
        if col not in work.columns:
            work[col] = "Unknown"
        work[col] = work[col].fillna("Unknown")

    frames: list[pd.DataFrame] = []
    for _, frame in work.groupby("symbol", sort=False):
        frame = frame.sort_values("date").copy()
        frame["row_in_symbol"] = np.arange(len(frame), dtype=np.int32)
        frame["prev_close"] = frame["close"].shift(1)
        frame["gap"] = frame["open"] / frame["prev_close"] - 1.0
        frame["ret1"] = frame["close"] / frame["prev_close"] - 1.0
        frame["ret20"] = frame["close"] / frame["close"].shift(20) - 1.0
        frame["ret60"] = frame["close"] / frame["close"].shift(60) - 1.0
        frame["dollar_volume"] = frame["raw_close"] * frame["volume"]
        frame["avg_volume20"] = frame["volume"].rolling(20, min_periods=5).mean()
        frame["avg_dollar_volume20"] = frame["dollar_volume"].rolling(20, min_periods=5).mean()
        frame["relvol20"] = _safe_div(frame["volume"], frame["avg_volume20"])
        for window, min_periods in [(10, 5), (20, 10), (50, 25), (100, 60), (150, 100), (200, 120)]:
            frame[f"sma{window}"] = frame["close"].rolling(window, min_periods=min_periods).mean()
        frame["sma50_slope20"] = frame["sma50"] / frame["sma50"].shift(20) - 1.0
        frame["sma200_slope20"] = frame["sma200"] / frame["sma200"].shift(20) - 1.0
        frame["high52"] = frame["high"].rolling(252, min_periods=60).max()
        frame["low52"] = frame["low"].rolling(252, min_periods=60).min()
        frame["prior20_high"] = frame["high"].shift(1).rolling(20, min_periods=10).max()
        frame["close_vs_52w_high"] = frame["close"] / frame["high52"] - 1.0
        frame["close_vs_52w_low"] = frame["close"] / frame["low52"] - 1.0
        frame["breakout20"] = frame["close"] >= frame["prior20_high"]
        frame["close_location"] = _close_location(frame).replace([np.inf, -np.inf], np.nan)
        frame["atr14"] = _wilder_atr14(frame)
        frame["sma20_slope5"] = frame["sma20"] / frame["sma20"].shift(5) - 1.0
        up_volume = frame["volume"].where(frame["ret1"] > 0, 0.0)
        down_volume = frame["volume"].where(frame["ret1"] < 0, 0.0)
        frame["upvol20"] = up_volume.rolling(20, min_periods=10).sum()
        frame["downvol20"] = down_volume.rolling(20, min_periods=10).sum()
        frame["upvol50"] = up_volume.rolling(50, min_periods=25).sum()
        frame["downvol50"] = down_volume.rolling(50, min_periods=25).sum()
        frame["ad_ratio20"] = _safe_div(frame["upvol20"], frame["downvol20"])
        frame["ad_ratio50"] = _safe_div(frame["upvol50"], frame["downvol50"])
        frame["stage2_proxy"] = (
            (frame["close"] > frame["sma50"])
            & (frame["sma50"] > frame["sma100"])
            & (frame["sma50_slope20"] > 0)
            & (frame["close_vs_52w_high"] >= -0.25)
            & (frame["close_vs_52w_low"] >= 0.30)
        )
        frame["minervini_proxy"] = (
            (frame["close"] > frame["sma50"])
            & (frame["close"] > frame["sma100"])
            & (frame["close"] > frame["sma150"])
            & (frame["sma50"] > frame["sma100"])
            & (frame["sma100"] > frame["sma150"])
            & (frame["close_vs_52w_high"] >= -0.25)
            & (frame["close_vs_52w_low"] >= 0.30)
        )
        frames.append(frame)
    work = pd.concat(frames, ignore_index=True)

    work["industry_ret60_median"] = work.groupby(["date", "industry"], dropna=False)["ret60"].transform("median")
    spy = (
        work.loc[work["symbol"].eq(spy_symbol.upper()), ["date", "ret60"]]
        .drop_duplicates("date")
        .rename(columns={"ret60": "spy_ret60"})
    )
    work = work.merge(spy, on="date", how="left")
    work["ret60_resid_spy"] = work["ret60"] - work["spy_ret60"]
    work["ret60_resid_industry"] = work["ret60"] - work["industry_ret60_median"]

    practical = (work["raw_close"] >= cfg.min_price) & (work["avg_dollar_volume20"] >= cfg.min_avg_dollar_volume20)
    clean_volume = (work["volume"] > 0) & (work["avg_volume20"] > 0) & work["relvol20"].replace([np.inf, -np.inf], np.nan).notna()
    no_biotech = ~work["industry"].fillna("").str.contains(cfg.excluded_industry_keyword, case=False, regex=False)
    work["ep_like"] = (
        practical
        & clean_volume
        & (work["relvol20"] >= cfg.ep_relvol20)
        & (
            (work["gap"] >= cfg.ep_gap)
            | (work["ret1"] >= cfg.ep_ret1)
            | ((work["ret1"] >= cfg.ep_breakout_ret1) & work["breakout20"])
        )
        & (work["close_location"] >= cfg.ep_close_location)
    )
    work["industry_theme_ep_ex_biotech"] = (
        work["ep_like"]
        & no_biotech
        & (work["industry_ret60_median"] >= cfg.industry_theme_ret60)
        & (work["ret60_resid_industry"] >= cfg.industry_residual_ret60)
    )
    return work.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def _selected_signal_positions(frame: pd.DataFrame, cfg: HybridSwingConfig) -> list[int]:
    signal_positions: list[int] = []
    last = -1_000_000
    for pos, is_signal in enumerate(frame["industry_theme_ep_ex_biotech"].fillna(False).to_numpy(bool)):
        if is_signal and pos - last > cfg.signal_cooldown_days:
            signal_positions.append(pos)
            last = pos
    return signal_positions


def _pullback10(decision: pd.Series, signal: pd.Series, cfg: HybridSwingConfig) -> bool:
    return bool(
        pd.notna(decision.get("sma10"))
        and decision["low"] <= decision["sma10"] * cfg.pullback_sma10_buffer
        and decision["close"] >= decision["sma10"]
        and decision["close"] > decision["open"]
        and decision["close"] >= signal["close"] * cfg.max_pullback_drop_from_ep_close
    )


def _next_business_day(value: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(value).normalize() + pd.offsets.BDay(1)).normalize()


def _entry_record(symbol: str, signal: pd.Series, decision: pd.Series, entry: pd.Series | None, cfg: HybridSwingConfig) -> dict:
    entry_date = pd.Timestamp(entry["date"]) if entry is not None else _next_business_day(pd.Timestamp(decision["date"]))
    entry_price = float(entry["open"]) if entry is not None and pd.notna(entry.get("open")) else np.nan
    return {
        "symbol": symbol,
        "entry_reason": ENTRY_REASON,
        "signal_date": pd.Timestamp(signal["date"]).date().isoformat(),
        "decision_date": pd.Timestamp(decision["date"]).date().isoformat(),
        "entry_date": entry_date.date().isoformat(),
        "entry_reference_price": float(decision["close"]),
        "entry_price": entry_price,
        "signal_low": float(signal["low"]),
        "signal_high": float(signal["high"]),
        "signal_close": float(signal["close"]),
        "ep_low_stop_used": True,
        "priority": ENTRY_PRIORITY[ENTRY_REASON],
        "priority_ep": 1,
        "priority_stage2": int(bool(signal.get("stage2_proxy", False) or signal.get("minervini_proxy", False))),
        "volume_ratio20": float(signal.get("relvol20", np.nan)),
        "industry_ret60_median": float(signal.get("industry_ret60_median", np.nan)),
        "ret60_resid_spy": float(signal.get("ret60_resid_spy", np.nan)),
        "ret60_resid_industry": float(signal.get("ret60_resid_industry", np.nan)),
        "avg_dollar_volume20": float(signal.get("avg_dollar_volume20", np.nan)),
        "industry": signal.get("industry", ""),
        "sector": signal.get("sector", ""),
        "ad_ratio20": float(decision.get("ad_ratio20", np.nan)),
        "ad_ratio50": float(decision.get("ad_ratio50", np.nan)),
        "close_vs_sma20": float(decision["close"] / decision["sma20"] - 1.0) if pd.notna(decision.get("sma20")) else np.nan,
        "sma20_slope5": float(decision.get("sma20_slope5", np.nan)),
    }


def build_historical_entry_candidates(
    features: pd.DataFrame,
    as_of: str | pd.Timestamp,
    cfg: HybridSwingConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or HybridSwingConfig()
    if features.empty:
        return pd.DataFrame()
    as_of_ts = pd.Timestamp(as_of).normalize()
    rows: list[dict] = []
    for symbol, frame in features.loc[features["symbol"].ne("SPY")].groupby("symbol", sort=False):
        frame = frame.sort_values("date").reset_index(drop=True)
        signals = _selected_signal_positions(frame, cfg)
        for sig_pos in signals:
            signal = frame.iloc[sig_pos]
            max_decision = min(len(frame) - 2, sig_pos + cfg.ep_watch_days)
            for dec_pos in range(sig_pos + 1, max_decision + 1):
                decision = frame.iloc[dec_pos]
                if pd.Timestamp(decision["date"]).normalize() > as_of_ts:
                    break
                if not _pullback10(decision, signal, cfg):
                    continue
                entry_pos = dec_pos + 1
                entry = frame.iloc[entry_pos]
                if pd.Timestamp(entry["date"]).normalize() > as_of_ts:
                    break
                record = _entry_record(symbol, signal, decision, entry, cfg)
                record["entry_row"] = int(entry_pos)
                record["decision_row"] = int(dec_pos)
                record["signal_row"] = int(sig_pos)
                rows.append(record)
                break
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    sort_cols = [
        "entry_date",
        "priority_ep",
        "priority_stage2",
        "volume_ratio20",
        "ret60_resid_spy",
        "ret60_resid_industry",
        "industry_ret60_median",
        "avg_dollar_volume20",
    ]
    return (
        out.sort_values(sort_cols, ascending=[True, False, False, False, False, False, False, False], kind="mergesort")
        .drop_duplicates(["symbol", "entry_date"], keep="first")
        .reset_index(drop=True)
    )


def detect_entry_candidates(features: pd.DataFrame, as_of: str | pd.Timestamp, cfg: HybridSwingConfig | None = None) -> pd.DataFrame:
    cfg = cfg or HybridSwingConfig()
    if features.empty:
        return pd.DataFrame()
    as_of_ts = pd.Timestamp(as_of).normalize()
    rows: list[dict] = []
    for symbol, frame in features.loc[features["symbol"].ne("SPY")].groupby("symbol", sort=False):
        frame = frame.sort_values("date").reset_index(drop=True)
        if not frame["date"].eq(as_of_ts).any():
            continue
        dec_pos = int(frame.index[frame["date"].eq(as_of_ts)][0])
        for sig_pos in _selected_signal_positions(frame, cfg):
            if not (0 < dec_pos - sig_pos <= cfg.ep_watch_days):
                continue
            signal = frame.iloc[sig_pos]
            decision = frame.iloc[dec_pos]
            if not _pullback10(decision, signal, cfg):
                continue
            already_triggered = any(_pullback10(frame.iloc[pos], signal, cfg) for pos in range(sig_pos + 1, dec_pos))
            if already_triggered:
                continue
            rows.append(_entry_record(symbol, signal, decision, None, cfg))
            break
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return (
        out.sort_values(
            ["priority", "volume_ratio20", "ret60_resid_spy", "ret60_resid_industry", "industry_ret60_median", "symbol"],
            ascending=[False, False, False, False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def evaluate_positions(
    features: pd.DataFrame,
    positions: pd.DataFrame,
    as_of: str | pd.Timestamp,
    cfg: HybridSwingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or HybridSwingConfig()
    if positions.empty:
        return pd.DataFrame(), pd.DataFrame()
    as_of_ts = pd.Timestamp(as_of).normalize()
    rows: list[dict] = []
    for pos in positions.to_dict("records"):
        symbol = str(pos["symbol"]).upper().strip()
        frame = features.loc[features["symbol"].eq(symbol)].sort_values("date").reset_index(drop=True)
        if frame.empty:
            continue
        entry_date = pd.Timestamp(pos["entry_date"]).normalize()
        signal_date = pd.Timestamp(pos.get("signal_date", entry_date)).normalize()
        frame = frame.loc[(frame["date"] >= entry_date) & (frame["date"] <= as_of_ts)].reset_index(drop=True)
        if frame.empty:
            continue
        entry_price = pos.get("entry_price")
        if pd.isna(entry_price) or entry_price in ("", None):
            entry_price = float(frame.iloc[0]["open"])
        entry_price = float(entry_price)
        signal_low = pos.get("signal_low", np.nan)
        if pd.isna(signal_low) or signal_low in ("", None):
            signal_frame = features.loc[(features["symbol"].eq(symbol)) & (features["date"].eq(signal_date))]
            signal_low = float(signal_frame.iloc[0]["low"]) if not signal_frame.empty else np.nan
        signal_low = float(signal_low) if pd.notna(signal_low) else np.nan
        high_watermark = -np.inf
        status = "継続"
        exit_reason = ""
        exit_signal_date = ""
        exit_date = ""
        exit_price = np.nan
        exit_idx = None

        for i, row in frame.iterrows():
            high_watermark = max(high_watermark, float(row["high"]))
            if pd.notna(signal_low) and float(row["low"]) <= signal_low:
                status = "Exit"
                exit_reason = "EP安値割れ"
                exit_signal_date = pd.Timestamp(row["date"]).date().isoformat()
                exit_date = exit_signal_date
                exit_price = min(float(row["open"]), signal_low)
                exit_idx = i
                break
            atr_stop = high_watermark - cfg.atr_trailing_multiple * float(row["atr14"]) if pd.notna(row["atr14"]) else np.nan
            atr_breach = pd.notna(atr_stop) and float(row["close"]) < atr_stop
            stage_fail = (
                pd.notna(row["sma50"])
                and pd.notna(row["sma150"])
                and float(row["close"]) < float(row["sma50"])
                and float(row["sma50"]) < float(row["sma150"])
            )
            if atr_breach or stage_fail:
                status = "Exit"
                exit_reason = "ATR8割れ" if atr_breach else "Stage2崩れ"
                exit_signal_date = pd.Timestamp(row["date"]).date().isoformat()
                if i + 1 < len(frame):
                    next_row = frame.iloc[i + 1]
                    exit_date = pd.Timestamp(next_row["date"]).date().isoformat()
                    exit_price = float(next_row["open"])
                    exit_idx = i + 1
                else:
                    exit_date = _next_business_day(pd.Timestamp(row["date"])).date().isoformat() + "予定"
                    exit_idx = i
                break

        ref = frame.iloc[exit_idx if exit_idx is not None else -1]
        mark_price = exit_price if pd.notna(exit_price) else float(ref["close"])
        result = mark_price / entry_price - 1.0
        held_frame = frame.loc[: exit_idx if exit_idx is not None else len(frame) - 1]
        held_high = held_frame["high"].max()
        held_high_date = held_frame.loc[held_frame["high"].idxmax(), "date"]
        rows.append(
            {
                "symbol": symbol,
                "entry_reason": pos.get("entry_reason", ENTRY_REASON),
                "signal_date": signal_date.date().isoformat(),
                "entry_date": entry_date.date().isoformat(),
                "entry_price": entry_price,
                "status": status,
                "exit_reason": exit_reason,
                "exit_signal_date": exit_signal_date,
                "exit_date_or_latest": exit_date or pd.Timestamp(ref["date"]).date().isoformat(),
                "exit_or_latest_price": mark_price,
                "return_pct": result,
                "held_high_date": pd.Timestamp(held_high_date).date().isoformat(),
                "held_max_return": float(held_high / entry_price - 1.0),
                "close_vs_sma20": float(ref["close"] / ref["sma20"] - 1.0) if pd.notna(ref["sma20"]) else np.nan,
                "sma20_slope5": float(ref["sma20_slope5"]) if pd.notna(ref["sma20_slope5"]) else np.nan,
                "ad_ratio20": float(ref["ad_ratio20"]) if pd.notna(ref["ad_ratio20"]) else np.nan,
                "ad_ratio50": float(ref["ad_ratio50"]) if pd.notna(ref["ad_ratio50"]) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(), pd.DataFrame()
    exits = out.loc[out["status"].eq("Exit")].reset_index(drop=True)
    holds = out.loc[out["status"].eq("継続")].reset_index(drop=True)
    return holds, exits


def format_percent(value: object) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value) * 100:+.1f}%"
    except Exception:
        return "-"


def _format_ratio(value: object) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.2f}x"
    except Exception:
        return "-"


def format_report_table(frame: pd.DataFrame, kind: str, *, as_of: str | pd.Timestamp) -> str:
    as_of_text = pd.Timestamp(as_of).date().isoformat()
    if frame.empty:
        return f"{kind}: 該当なし"
    lines: list[str] = []
    if kind == "Entry候補":
        lines.append("銘柄 | Entry根拠 | Signal | Entry予定 | 補足")
        lines.append("---|---|---:|---:|---")
        for row in frame.to_dict("records"):
            note = (
                f"Industry {format_percent(row.get('industry_ret60_median'))}, "
                f"個別残差 {format_percent(row.get('ret60_resid_industry'))}, "
                f"出来高 {_format_ratio(row.get('volume_ratio20'))}"
            )
            lines.append(
                f"{row['symbol']} | {row['entry_reason']} | {row['signal_date']} | {row['entry_date']} | {note}"
            )
    else:
        lines.append(kind)
        lines.append("銘柄 | Entry根拠 | Signal | Entry | Exit/継続 | 結果")
        lines.append("---|---|---:|---:|---:|---:")
        for row in frame.to_dict("records"):
            if kind == "Exit":
                status_text = f"{row['exit_date_or_latest']} {row['exit_reason']}"
            else:
                status_text = f"{as_of_text}時点で継続"
            lines.append(
                f"{row['symbol']} | {row.get('entry_reason', ENTRY_REASON)} | {row['signal_date']} | {row['entry_date']} | "
                f"{status_text} | {format_percent(row['return_pct'])}"
            )
    return "\n".join(lines)
