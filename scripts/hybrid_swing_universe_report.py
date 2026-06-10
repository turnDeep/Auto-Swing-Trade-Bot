from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.fmp import download_yfinance_bars
from signals.hybrid_swing_engine import (
    HybridSwingConfig,
    build_hybrid_features,
    build_historical_entry_candidates,
    detect_entry_candidates,
    format_percent,
    format_report_table,
)


LOGGER = logging.getLogger(__name__)
FMP_HISTORICAL_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
DISCORD_API_BASE = "https://discord.com/api/v10"
DEFAULT_BREAKOUT_CACHE = Path.home() / "BreakOut" / "analysis_outputs" / "ep_strength_confirmations_20260529"
DEFAULT_UNIVERSE_PATH = Path.home() / "BreakOut" / "analysis_outputs" / "russell3000_daily_10y_dataset" / "universe.parquet"


def _load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(Path.home() / "BreakOut" / ".env", override=False)


def _env_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    env_path = Path.home() / "BreakOut" / ".env"
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(rf"^{re.escape(name)}\s*=\s*(.+)$", text, re.M)
        if match:
            return match.group(1).strip().strip("\"'")
    return ""


def _fetch_fmp_delta(symbols: list[str], api_key: str, start: str, end: str) -> pd.DataFrame:
    session = requests.Session()
    rows: list[dict] = []
    chunk_size = 100
    chunks = [symbols[i : i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    for chunk_idx, chunk in enumerate(chunks, start=1):
        LOGGER.info("Fetching FMP historical delta chunk %s/%s", chunk_idx, len(chunks))
        joined = ",".join(chunk)
        for attempt in range(3):
            try:
                response = session.get(
                    FMP_HISTORICAL_URL.format(symbol=joined),
                    params={"from": start, "to": end, "apikey": api_key},
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                if "historicalStockList" in payload:
                    series = payload.get("historicalStockList", [])
                    for stock in series:
                        symbol = str(stock.get("symbol", "")).upper().strip()
                        for item in stock.get("historical", []):
                            rows.append(
                                {
                                    "symbol": symbol,
                                    "date": item.get("date"),
                                    "open": item.get("open"),
                                    "high": item.get("high"),
                                    "low": item.get("low"),
                                    "close": item.get("close"),
                                    "volume": item.get("volume"),
                                }
                            )
                else:
                    symbol = str(payload.get("symbol", joined)).upper().strip()
                    for item in payload.get("historical", []):
                        rows.append(
                            {
                                "symbol": symbol,
                                "date": item.get("date"),
                                "open": item.get("open"),
                                "high": item.get("high"),
                                "low": item.get("low"),
                                "close": item.get("close"),
                                "volume": item.get("volume"),
                            }
                        )
                break
            except Exception as exc:
                if attempt == 2:
                    LOGGER.warning("FMP delta fetch failed for chunk %s: %s", chunk_idx, exc)
                time.sleep(0.5 + attempt)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"])


def _fetch_yfinance_delta(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    # yfinance end date is exclusive, so add one calendar day.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    frame = download_yfinance_bars(symbols, interval="1d", start=start, end=end_exclusive, auto_adjust=False)
    if frame.empty:
        return frame
    frame = frame.rename(columns={"ts": "date"})
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    keep = ["symbol", "date", "open", "high", "low", "close", "volume"]
    return frame[keep].dropna(subset=keep)


def load_daily_panel(
    *,
    as_of: pd.Timestamp,
    cache_dir: Path,
    universe_path: Path,
    api_key: str,
    delta_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_path = cache_dir / "combined_daily_to_20260529.parquet"
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if not universe_path.exists():
        raise FileNotFoundError(universe_path)
    universe = pd.read_parquet(universe_path)
    universe["symbol"] = universe["symbol"].astype(str).str.upper().str.strip()
    universe = universe.head(3000).copy()

    base = pd.read_parquet(base_path)
    base["symbol"] = base["symbol"].astype(str).str.upper().str.strip()
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    base = base.loc[base["symbol"].isin(universe["symbol"])].copy()

    symbols = sorted(set(universe["symbol"].tolist() + ["SPY"]))
    delta_start = (base["date"].max() + pd.offsets.BDay(1)).date().isoformat()
    delta_end = as_of.date().isoformat()
    if delta_source == "fmp":
        delta = _fetch_fmp_delta(symbols, api_key, delta_start, delta_end)
    else:
        delta = _fetch_yfinance_delta(symbols, delta_start, delta_end)
        fetched_symbols = set(delta["symbol"]) if not delta.empty else set()
        missing = [symbol for symbol in symbols if symbol not in fetched_symbols]
        if missing:
            LOGGER.warning("yfinance delta missing %s symbols; falling back to FMP for missing preview=%s", len(missing), missing[:10])
            fallback = _fetch_fmp_delta(missing, api_key, delta_start, delta_end)
            delta = pd.concat([delta, fallback], ignore_index=True, sort=False)
    daily = pd.concat([base, delta], ignore_index=True, sort=False)
    daily = daily.loc[daily["date"].le(as_of)].copy()
    daily = (
        daily.sort_values(["symbol", "date"], kind="mergesort")
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )
    return daily, universe


def build_historical_candidate_frame(features: pd.DataFrame, as_of: pd.Timestamp, cfg: HybridSwingConfig) -> pd.DataFrame:
    return build_historical_entry_candidates(features, as_of, cfg)


def derive_position_state(features: pd.DataFrame, as_of: pd.Timestamp, cfg: HybridSwingConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active_rows: list[dict] = []
    today_exit_rows: list[dict] = []
    candidates = build_historical_candidate_frame(features, as_of, cfg)
    if candidates.empty:
        return candidates, pd.DataFrame(), pd.DataFrame()
    for symbol, raw_frame in features.loc[features["symbol"].ne("SPY")].groupby("symbol", sort=False):
        frame = raw_frame.sort_values("date").reset_index(drop=True)
        symbol_candidates = candidates.loc[candidates["symbol"].eq(symbol)].copy()
        if symbol_candidates.empty:
            continue
        symbol_candidates = symbol_candidates.sort_values(["entry_row", "priority"], ascending=[True, False], kind="mergesort")
        candidate_by_entry: dict[int, dict] = {}
        for entry_row, group in symbol_candidates.groupby("entry_row", sort=False):
            candidate_by_entry[int(entry_row)] = group.iloc[0].to_dict()
        position: dict | None = None
        high_water = -np.inf
        entry_i = -1
        entry_price = np.nan
        for i, row in frame.iterrows():
            row_date = pd.Timestamp(row["date"]).normalize()
            if row_date > as_of:
                break
            if position is None and i in candidate_by_entry:
                position = candidate_by_entry[i]
                entry_i = i
                entry_price = float(position["entry_price"])
                high_water = -np.inf
            if position is None:
                continue
            high_water = max(high_water, float(row["high"]))
            if position["ep_low_stop_used"] and float(row["low"]) <= float(position["signal_low"]):
                exit_price = min(float(row["open"]), float(position["signal_low"]))
                exit_record = {
                        **position,
                        "status": "Exit",
                        "exit_reason": "EP安値割れ",
                        "exit_signal_date": row_date.date().isoformat(),
                        "exit_date_or_latest": row_date.date().isoformat(),
                        "exit_or_latest_price": exit_price,
                        "return_pct": exit_price / entry_price - 1.0,
                        "held_max_return": float(frame.loc[entry_i:i, "high"].max() / entry_price - 1.0),
                }
                if row_date == as_of:
                    today_exit_rows.append(exit_record)
                position = None
                continue
            atr_stop = high_water - cfg.atr_trailing_multiple * float(row["atr14"]) if pd.notna(row["atr14"]) else np.nan
            atr_breach = pd.notna(atr_stop) and float(row["close"]) < atr_stop
            stage_fail = (
                pd.notna(row["sma50"])
                and pd.notna(row["sma150"])
                and float(row["close"]) < float(row["sma50"])
                and float(row["sma50"]) < float(row["sma150"])
            )
            if atr_breach or stage_fail:
                exit_reason = "ATR8割れ" if atr_breach else "Stage2崩れ"
                if i + 1 < len(frame) and pd.Timestamp(frame.loc[i + 1, "date"]).normalize() <= as_of:
                    exit_date = pd.Timestamp(frame.loc[i + 1, "date"]).normalize()
                    exit_price = float(frame.loc[i + 1, "open"])
                else:
                    exit_date = row_date + pd.offsets.BDay(1)
                    exit_price = np.nan
                exit_record = {
                        **position,
                        "status": "Exit",
                        "exit_reason": exit_reason,
                        "exit_signal_date": row_date.date().isoformat(),
                        "exit_date_or_latest": exit_date.date().isoformat(),
                        "exit_or_latest_price": exit_price,
                        "return_pct": exit_price / entry_price - 1.0 if pd.notna(exit_price) else np.nan,
                        "held_max_return": float(frame.loc[entry_i:i, "high"].max() / entry_price - 1.0),
                }
                if row_date == as_of:
                    today_exit_rows.append(exit_record)
                position = None
        if position is not None:
            latest = frame.loc[frame["date"].le(as_of)].iloc[-1]
            active_rows.append(
                {
                    **position,
                    "status": "継続",
                    "exit_reason": "",
                    "exit_signal_date": "",
                    "exit_date_or_latest": as_of.date().isoformat(),
                    "exit_or_latest_price": float(latest["close"]),
                    "return_pct": float(latest["close"]) / entry_price - 1.0,
                    "held_max_return": float(frame.loc[entry_i:, "high"].max() / entry_price - 1.0),
                    "close_vs_sma20": float(latest["close"] / latest["sma20"] - 1.0) if pd.notna(latest["sma20"]) else np.nan,
                    "sma20_slope5": float(latest["sma20_slope5"]) if pd.notna(latest["sma20_slope5"]) else np.nan,
                }
            )
    return candidates, pd.DataFrame(active_rows), pd.DataFrame(today_exit_rows)


def _markdown_table(frame: pd.DataFrame, kind: str, as_of: pd.Timestamp, limit: int = 80) -> str:
    if frame.empty:
        return f"{kind}: 該当なし"
    work = frame.copy()
    if kind == "Entry候補":
        return format_report_table(work.head(limit), "Entry候補", as_of=as_of)
    lines = [kind, "銘柄 | Entry根拠 | Signal | Entry | Exit/継続 | 結果", "---|---|---:|---:|---:|---:"]
    for row in work.head(limit).to_dict("records"):
        if kind == "Exit":
            status = f"{row['exit_date_or_latest']} {row['exit_reason']}"
        else:
            status = f"{as_of.date().isoformat()}時点で継続"
        lines.append(
            f"{row['symbol']} | {row['entry_reason']} | {row['signal_date']} | {row['entry_date']} | {status} | {format_percent(row['return_pct'])}"
        )
    if len(work) > limit:
        lines.append(f"...ほか {len(work) - limit} 件")
    return "\n".join(lines)


def _send_discord(content: str, token: str, channel_id: str) -> str | None:
    response = requests.post(
        f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
        headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
        data=json.dumps({"content": content}, ensure_ascii=False),
        timeout=30,
    )
    if response.status_code not in {200, 201}:
        return f"http_{response.status_code}:{response.text[:200]}"
    return None


def run(args: argparse.Namespace) -> dict:
    _load_env()
    api_key = _env_value("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY is required")
    as_of = pd.Timestamp(args.as_of).normalize()
    cfg = HybridSwingConfig()
    daily, universe = load_daily_panel(
        as_of=as_of,
        cache_dir=args.cache_dir,
        universe_path=args.universe_path,
        api_key=api_key,
        delta_source=args.delta_source,
    )
    LOGGER.info("Loaded daily panel rows=%s symbols=%s", len(daily), daily["symbol"].nunique())
    features = build_hybrid_features(daily, universe, cfg=cfg)
    LOGGER.info("Built hybrid features rows=%s symbols=%s", len(features), features["symbol"].nunique())
    entry = detect_entry_candidates(features, as_of, cfg)
    LOGGER.info("Detected same-day entry candidates=%s", len(entry))
    all_entries, holds, exits_today = derive_position_state(features, as_of, cfg)
    LOGGER.info("Derived state: historical_entries=%s holds=%s exits_today=%s", len(all_entries), len(holds), len(exits_today))
    active_symbols = set(holds["symbol"]) if not holds.empty else set()
    exit_symbols = set(exits_today["symbol"]) if not exits_today.empty else set()
    if not entry.empty:
        entry = entry.loc[~entry["symbol"].isin(active_symbols | exit_symbols)].copy()
    if not holds.empty:
        holds = holds.sort_values(["return_pct", "held_max_return"], ascending=[False, False], kind="mergesort")
    if not exits_today.empty:
        exits_today = exits_today.sort_values(["return_pct"], ascending=[False], kind="mergesort")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    entry_path = out_dir / f"universe_entry_candidates_{as_of.date()}.csv"
    hold_path = out_dir / f"universe_holds_{as_of.date()}.csv"
    exit_path = out_dir / f"universe_exits_{as_of.date()}.csv"
    all_entry_path = out_dir / f"universe_all_historical_entries_{as_of.date()}.csv"
    entry.to_csv(entry_path, index=False, encoding="utf-8-sig")
    holds.to_csv(hold_path, index=False, encoding="utf-8-sig")
    exits_today.to_csv(exit_path, index=False, encoding="utf-8-sig")
    all_entries.to_csv(all_entry_path, index=False, encoding="utf-8-sig")

    sections = [
        f"対象: {universe['symbol'].nunique():,}銘柄 / as of {as_of.date()}",
        f"Entry候補: {len(entry):,}件 / 継続: {len(holds):,}件 / Exit: {len(exits_today):,}件",
        "```md\n" + _markdown_table(entry, "Entry候補", as_of, limit=args.discord_limit) + "\n```",
        "```md\n" + _markdown_table(holds, "継続", as_of, limit=args.discord_limit) + "\n```",
        "```md\n" + _markdown_table(exits_today, "Exit", as_of, limit=args.discord_limit) + "\n```",
    ]
    preview = "\n\n".join(sections)
    preview_path = out_dir / f"universe_discord_preview_{as_of.date()}.md"
    preview_path.write_text(preview, encoding="utf-8")
    discord_error = None
    if args.send_discord:
        token = _env_value("DISCORD_BOT_TOKEN")
        channel_id = _env_value("DISCORD_CHANNEL_ID")
        if not token or not channel_id:
            discord_error = "discord_credentials_missing"
        else:
            # Keep under Discord's 2000-character hard limit.
            chunks = []
            current = ""
            for block in sections:
                addition = ("\n\n" if current else "") + block
                if len(current) + len(addition) > 1800:
                    chunks.append(current)
                    current = block
                else:
                    current += addition
            if current:
                chunks.append(current)
            for chunk in chunks:
                discord_error = _send_discord(chunk, token, channel_id)
                if discord_error:
                    break
    return {
        "entry_path": str(entry_path),
        "hold_path": str(hold_path),
        "exit_path": str(exit_path),
        "all_entry_path": str(all_entry_path),
        "preview_path": str(preview_path),
        "entry_count": len(entry),
        "hold_count": len(holds),
        "exit_count": len(exits_today),
        "discord_error": discord_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-universe industry-theme EP swing state report.")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_BREAKOUT_CACHE)
    parser.add_argument("--universe-path", type=Path, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "hybrid_swing_universe")
    parser.add_argument("--delta-source", choices=["yfinance", "fmp"], default="yfinance")
    parser.add_argument("--discord-limit", type=int, default=30)
    parser.add_argument("--send-discord", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
