# Stallion-System-Trade

English | [日本語](./README.ja.md)

Stallion-System-Trade is a two-stage intraday trading system for U.S. equities.

- Stage 1: nightly watchlist model selects the next-session top 100 symbols from the Russell-3000 universe using 7 daily features.
- Stage 2: intraday execution model scores the shortlist with 16 features on 5-minute data and submits up to 4 positions.
- Storage uses SQLite + Parquet.
- Live trading supports Webull Japan accounts and automatically falls back to demo mode when required live credentials are missing.
- Discord notifications report startup mode, pre-market status, order submission, fills, order failures, and market-close summary.

## Runtime Modes

Mode is determined automatically from `.env`.

- `LIVE`: enabled only when all of the following are present:
  - `WEBULL_APP_KEY`
  - `WEBULL_APP_SECRET`
  - `WEBULL_ACCOUNT_ID`
- `DEMO`: used whenever any required Webull credential is missing or incomplete.

Discord messages are prefixed with `[LIVE]` or `[DEMO]`.

## Trading Flow

### Stage 1: Nightly watchlist model

- Universe: top 3000 U.S. stocks by market cap
- Raw 1d and 5m history is retained for the full universe
- Tradeability filtering is applied after feature construction and before stage-1 / stage-2 learning and latest watchlist scoring
- Tradeability filter:
  - `min_price >= 5.0`
  - `min_daily_volume >= 1,000,000`
  - `min_dollar_volume >= 10,000,000`
- Input features at the latest completed close (`t`):
  1. `daily_buy_pressure_prev`
  2. `daily_rs_score_prev`
  3. `daily_rrs_prev`
  4. `prev_day_adr_pct`
  5. `industry_buy_pressure_prev`
  6. `sector_buy_pressure_prev`
  7. `industry_rs_prev`
- Model: `LogisticRegression`
- `daily_rs_score` uses `0.40 * ROC21 + 0.20 * ROC63 + 0.20 * ROC126 + 0.20 * ROC252`
- Output: top 100 symbols for the next session (`t+1`)

### Stage 2: Intraday execution

- Input symbols: stage-1 shortlist only
- Model: `HistGradientBoostingClassifier`
- Data:
  - 5-minute bars
  - 15-minute context derived from 5-minute bars
  - previous daily context
- Trading window: 5 to 90 minutes after the U.S. open
- Entry: next-bar open
- Exit: same-day flatten before close
- Max positions: 4

## Order and Slot Management

The live trader explicitly distinguishes:

- available slots
- pending buy order slots
- partially filled slots
- filled slots
- sell-pending slots
- reserved buying power

Important behavior:

- a submitted but unfilled order does **not** free the slot
- a slot is released only after:
  - cancel confirmed
  - rejected
  - expired
  - failed
  - fully sold and position cleared
- stale orders are monitored and can be canceled after the configured timeout
- partial fills keep the slot occupied

## Buying Power and Sizing

- The system uses **opening buying power**, not total equity
- Opening buying power is split into 4 equal slot budgets
- Each order uses integer-share sizing by default
- If the slot budget is below one share price:
  - the order is skipped
  - the reason is logged
  - a Discord notification is sent
- If a market order fails:
  - the trader retries with a configurable marketable limit order

## Discord Notifications

If both `DISCORD_BOT_TOKEN` and `DISCORD_CHANNEL_ID` are present, the bot sends:

- scheduler startup
- nightly pipeline start / finish / failure
- pre-market status (5 minutes before open)
- buy order submission
- fill updates
- cancel requests
- market-close summary

If only the bot token is present, token validity can still be checked but messages cannot be delivered until `DISCORD_CHANNEL_ID` is set.

## Environment Variables

Create `.env` from `.env.example`.

```env
FMP_API_KEY=
WEBULL_APP_KEY=
WEBULL_APP_SECRET=
WEBULL_ACCOUNT_ID=
DISCORD_BOT_TOKEN=
DISCORD_CHANNEL_ID=
```

## Main Commands

Nightly pipeline:

```bash
python ml_pipeline_60d.py
```

Live trader:

```bash
python webull_live_trader.py
```

Scheduler:

```bash
python master_scheduler.py
```

Docker:

```bash
docker compose up -d --build
```

On a fresh deployment, the first bootstrap can take a long time because the system must fetch and build the initial history. Later restarts reuse existing SQLite + Parquet artifacts.

## Docker Notes

- The container uses named volumes for `/app/data` and `/app/reports`
- Startup bootstrap runs only when required artifacts are missing
- The healthcheck uses `stallion.watchdog`

## Project Layout

| File | Role |
|---|---|
| `ml_pipeline_60d.py` | nightly pipeline entrypoint |
| `webull_live_trader.py` | live execution entrypoint |
| `master_scheduler.py` | scheduler and bootstrap |
| `stallion/config.py` | runtime configuration and mode selection |
| `stallion/storage.py` | SQLite + Parquet operational store |
| `stallion/watchlist_model.py` | stage-1 watchlist learning |
| `stallion/modeling.py` | stage-2 HistGBM learning and scoring |
| `stallion/live_trader.py` | live polling, scoring, slots, orders, close summary |
| `stallion/broker.py` | Webull JP broker wrapper and demo broker |
| `stallion/discord_notifier.py` | async Discord delivery |
| `stallion/slot_manager.py` | slot occupancy / reserved buying power |

## Current Scope

This repository is designed for automated operation, but you should still validate:

- your Webull JP account permissions
- Discord channel configuration
- time zone / market hours on your host
- initial bootstrap completion before relying on scheduled live trading

Use demo mode first if you are unsure.
