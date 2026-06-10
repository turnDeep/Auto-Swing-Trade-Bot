from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import load_settings
from core.fmp import FMPClient
from signals.hybrid_swing_engine import (
    build_hybrid_features,
    detect_entry_candidates,
    evaluate_positions,
    format_report_table,
)


LOGGER = logging.getLogger(__name__)
FMP_HISTORICAL_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
FMP_PROFILE_URL = "https://financialmodelingprep.com/api/v3/profile/{symbol}"
DISCORD_API_BASE = "https://discord.com/api/v10"


CASE_STUDY_POSITIONS = [
    {
        "symbol": "AXTI",
        "entry_reason": "IndustryテーマEP後10MA押し目維持",
        "signal_date": "2025-12-09",
        "entry_date": "2025-12-22",
        "ep_low_stop_used": True,
    },
    {
        "symbol": "TTMI",
        "entry_reason": "IndustryテーマEP後10MA押し目維持",
        "signal_date": "2025-06-09",
        "entry_date": "2025-06-20",
        "ep_low_stop_used": True,
    },
    {
        "symbol": "LITE",
        "entry_reason": "IndustryテーマEP後10MA押し目維持",
        "signal_date": "2025-10-29",
        "entry_date": "2025-11-18",
        "ep_low_stop_used": True,
    },
    {
        "symbol": "FORM",
        "entry_reason": "IndustryテーマEP後10MA押し目維持",
        "signal_date": "2025-10-29",
        "entry_date": "2025-11-25",
        "ep_low_stop_used": True,
    },
    {
        "symbol": "AMR",
        "entry_reason": "IndustryテーマEP後10MA押し目維持",
        "signal_date": "2020-12-08",
        "entry_date": "2020-12-18",
        "ep_low_stop_used": True,
    },
]


def _load_env_files() -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(Path.cwd() / ".env")
    breakout_env = Path.home() / "BreakOut" / ".env"
    if breakout_env.exists():
        load_dotenv(breakout_env, override=False)


def _symbol_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().upper() for item in value.replace("\n", ",").split(",") if item.strip()]


def _case_positions() -> pd.DataFrame:
    frame = pd.DataFrame(CASE_STUDY_POSITIONS)
    frame["display_symbol"] = frame["symbol"]
    frame["symbol"] = frame.get("actual_symbol", frame["symbol"]).fillna(frame["symbol"])
    return frame


def _read_positions(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.columns = [str(col).strip() for col in frame.columns]
    if "display_symbol" not in frame.columns:
        frame["display_symbol"] = frame["symbol"]
    return frame


def _fetch_fmp_history(symbols: Iterable[str], api_key: str, start: str, end: str) -> pd.DataFrame:
    session = requests.Session()
    rows: list[dict] = []
    for symbol in list(dict.fromkeys(str(s).upper().strip() for s in symbols if str(s).strip())):
        for attempt in range(3):
            try:
                response = session.get(
                    FMP_HISTORICAL_URL.format(symbol=symbol),
                    params={"from": start, "to": end, "apikey": api_key},
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                for item in payload.get("historical", []):
                    rows.append(
                        {
                            "symbol": symbol,
                            "date": item.get("date"),
                            "open": item.get("open"),
                            "high": item.get("high"),
                            "low": item.get("low"),
                            "close": item.get("close"),
                            "adj_close": item.get("adjClose", item.get("close")),
                            "volume": item.get("volume"),
                        }
                    )
                break
            except Exception as exc:
                if attempt == 2:
                    LOGGER.warning("FMP historical fetch failed for %s: %s", symbol, exc)
                time.sleep(1 + attempt)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"])


def _fetch_profiles(symbols: Iterable[str], api_key: str) -> pd.DataFrame:
    session = requests.Session()
    rows: list[dict] = []
    for symbol in list(dict.fromkeys(str(s).upper().strip() for s in symbols if str(s).strip())):
        try:
            response = session.get(FMP_PROFILE_URL.format(symbol=symbol), params={"apikey": api_key}, timeout=20)
            response.raise_for_status()
            data = response.json()
            if data:
                item = data[0]
                rows.append(
                    {
                        "symbol": symbol,
                        "sector": item.get("sector") or "Unknown",
                        "industry": item.get("industry") or "Unknown",
                        "market_cap": item.get("mktCap") or item.get("marketCap"),
                    }
                )
        except Exception as exc:
            LOGGER.warning("FMP profile fetch failed for %s: %s", symbol, exc)
    return pd.DataFrame(rows)


def _chunk_messages(title: str, sections: list[str], max_len: int = 1800) -> list[str]:
    messages: list[str] = []
    current = title.strip()
    for section in sections:
        block = f"\n\n{section.strip()}"
        if len(current) + len(block) > max_len:
            messages.append(current)
            current = f"{title.strip()}\n\n{section.strip()}"
        else:
            current += block
    if current.strip():
        messages.append(current)
    return messages


def _send_discord(messages: list[str], token: str, channel_id: str) -> list[str]:
    session = requests.Session()
    errors: list[str] = []
    for content in messages:
        response = session.post(
            f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
            headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
            data=json.dumps({"content": content}, ensure_ascii=False),
            timeout=30,
        )
        if response.status_code not in {200, 201}:
            errors.append(f"http_{response.status_code}:{response.text[:160]}")
    return errors


def run_report(
    *,
    as_of: str,
    symbols: list[str],
    positions: pd.DataFrame,
    top_universe: int,
    send_discord: bool,
    output_dir: Path,
) -> dict[str, Path | int | list[str]]:
    settings = load_settings(REPO_ROOT)
    api_key = settings.credentials.fmp_api_key

    as_of_ts = pd.Timestamp(as_of).normalize()
    start_ts = as_of_ts - pd.Timedelta(days=900)
    if not positions.empty:
        date_values = []
        for col in ["signal_date", "entry_date"]:
            if col in positions.columns:
                date_values.append(pd.to_datetime(positions[col], errors="coerce"))
        if date_values:
            oldest = pd.concat(date_values).dropna().min()
            if pd.notna(oldest):
                start_ts = min(start_ts, oldest - pd.Timedelta(days=450))
    start = start_ts.date().isoformat()
    end = as_of_ts.date().isoformat()

    if positions.empty and not symbols:
        fmp = FMPClient(settings)
        universe = fmp.fetch_top_universe(top_universe)
        symbols = universe["symbol"].tolist()
    else:
        base_symbols = list(symbols)
        if not positions.empty:
            base_symbols.extend(positions["symbol"].astype(str).str.upper().tolist())
        symbols = sorted(set(base_symbols))
        universe = _fetch_profiles(symbols, api_key)

    needed_symbols = sorted(set(symbols + ["SPY"]))
    daily = _fetch_fmp_history(needed_symbols, api_key, start, end)
    features = build_hybrid_features(daily, universe)

    entry_candidates = detect_entry_candidates(features, as_of_ts)
    holds, exits = evaluate_positions(features, positions, as_of_ts)
    if not positions.empty and "display_symbol" in positions.columns:
        display_map = (
            positions.assign(
                signal_date_key=pd.to_datetime(positions["signal_date"], errors="coerce").dt.date.astype(str),
                entry_date_key=pd.to_datetime(positions["entry_date"], errors="coerce").dt.date.astype(str),
            )
            .set_index(["symbol", "signal_date_key", "entry_date_key"])["display_symbol"]
            .to_dict()
        )
        for frame in [holds, exits]:
            if not frame.empty:
                frame["symbol"] = frame.apply(
                    lambda row: display_map.get(
                        (
                            str(row["symbol"]).upper(),
                            str(pd.Timestamp(row["signal_date"]).date()),
                            str(pd.Timestamp(row["entry_date"]).date()),
                        ),
                        row["symbol"],
                    ),
                    axis=1,
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    entry_path = output_dir / f"hybrid_entry_candidates_{as_of_ts.date()}.csv"
    hold_path = output_dir / f"hybrid_holds_{as_of_ts.date()}.csv"
    exit_path = output_dir / f"hybrid_exits_{as_of_ts.date()}.csv"
    entry_candidates.to_csv(entry_path, index=False, encoding="utf-8-sig")
    holds.to_csv(hold_path, index=False, encoding="utf-8-sig")
    exits.to_csv(exit_path, index=False, encoding="utf-8-sig")

    sections = [
        "```md\n" + format_report_table(entry_candidates, "Entry候補", as_of=as_of_ts) + "\n```",
        "```md\n" + format_report_table(holds, "継続", as_of=as_of_ts) + "\n```",
        "```md\n" + format_report_table(exits, "Exit", as_of=as_of_ts) + "\n```",
    ]
    title = f"Industry Theme EP Swing 日次判定 {as_of_ts.date()}"
    messages = _chunk_messages(title, sections)
    errors: list[str] = []
    if send_discord:
        token = str(settings.credentials.discord_bot_token or "").strip()
        channel_id = str(settings.credentials.discord_channel_id or "").strip()
        if not token or not channel_id:
            errors.append("discord_credentials_missing")
        else:
            errors = _send_discord(messages, token, channel_id)

    preview_path = output_dir / f"hybrid_discord_preview_{as_of_ts.date()}.md"
    preview_path.write_text("\n\n---\n\n".join(messages), encoding="utf-8")
    return {
        "entry_path": entry_path,
        "hold_path": hold_path,
        "exit_path": exit_path,
        "preview_path": preview_path,
        "entry_count": int(len(entry_candidates)),
        "hold_count": int(len(holds)),
        "exit_count": int(len(exits)),
        "discord_errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily industry-theme EP swing report.")
    parser.add_argument("--as-of", required=True, help="Market close date, YYYY-MM-DD.")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols to scan for new entries.")
    parser.add_argument("--positions-csv", type=Path, help="Open/research positions CSV.")
    parser.add_argument("--case-study", action="store_true", help="Use built-in AAOI/PLTR/LITE/WDC/AEHR/VSH/BAND/FLEX case positions.")
    parser.add_argument("--top-universe", type=int, default=3000, help="FMP top universe size when --symbols is omitted.")
    parser.add_argument("--send-discord", action="store_true", help="Send the report to Discord.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "hybrid_swing")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_env_files()
    args = parse_args()
    positions = _read_positions(args.positions_csv)
    if args.case_study:
        positions = pd.concat([positions, _case_positions()], ignore_index=True, sort=False)
    symbols = _symbol_list(args.symbols)
    if args.case_study:
        symbols = sorted(set(symbols + _case_positions()["symbol"].tolist()))
    result = run_report(
        as_of=args.as_of,
        symbols=symbols,
        positions=positions,
        top_universe=args.top_universe,
        send_discord=args.send_discord,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
