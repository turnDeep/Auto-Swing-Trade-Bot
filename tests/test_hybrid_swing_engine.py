from __future__ import annotations

import unittest

import pandas as pd

from signals.hybrid_swing_engine import (
    ENTRY_REASON,
    HybridSwingConfig,
    build_hybrid_features,
    build_historical_entry_candidates,
    detect_entry_candidates,
    evaluate_positions,
)


def _bars(symbol: str, dates: pd.DatetimeIndex, closes: list[float], *, volume: float = 10_000) -> list[dict]:
    rows: list[dict] = []
    for date, close in zip(dates, closes):
        rows.append(
            {
                "symbol": symbol,
                "date": date,
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": volume,
            }
        )
    return rows


class IndustryThemeEpEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.bdate_range("2025-01-02", periods=130)
        base = [10 + i * 0.03 for i in range(130)]
        test = base.copy()
        peer1 = [10 + i * 0.020 for i in range(130)]
        peer2 = [9 + i * 0.019 for i in range(130)]
        spy = [100 + i * 0.05 for i in range(130)]
        bio = base.copy()

        sig = 90
        test[sig] = test[sig - 1] * 1.14
        test[sig + 1 : sig + 5] = [test[sig] * x for x in [0.98, 0.965, 0.955, 0.96]]
        test[sig + 5] = test[sig] * 0.985
        bio[sig] = bio[sig - 1] * 1.14
        bio[sig + 1 : sig + 6] = test[sig + 1 : sig + 6]

        rows = []
        rows += _bars("TEST", self.dates, test)
        rows += _bars("PEER1", self.dates, peer1)
        rows += _bars("PEER2", self.dates, peer2)
        rows += _bars("BIOX", self.dates, bio)
        rows += _bars("SPY", self.dates, spy, volume=1_000_000)
        daily = pd.DataFrame(rows)
        for symbol in ["TEST", "BIOX"]:
            m = daily["symbol"].eq(symbol) & daily["date"].eq(self.dates[sig])
            close = float(daily.loc[m, "close"].iloc[0])
            daily.loc[m, "open"] = close * 0.94
            daily.loc[m, "high"] = close * 1.01
            daily.loc[m, "low"] = close * 0.92
            daily.loc[m, "volume"] = 80_000

        decision_date = self.dates[sig + 5]
        for symbol in ["TEST", "BIOX"]:
            m = daily["symbol"].eq(symbol) & daily["date"].eq(decision_date)
            close = float(daily.loc[m, "close"].iloc[0])
            daily.loc[m, "open"] = close * 0.985
            daily.loc[m, "high"] = close * 1.01
            daily.loc[m, "low"] = close * 0.94

        universe = pd.DataFrame(
            [
                {"symbol": "TEST", "sector": "Technology", "industry": "Semiconductors"},
                {"symbol": "PEER1", "sector": "Technology", "industry": "Semiconductors"},
                {"symbol": "PEER2", "sector": "Technology", "industry": "Semiconductors"},
                {"symbol": "BIOX", "sector": "Healthcare", "industry": "Biotechnology"},
                {"symbol": "SPY", "sector": "ETF", "industry": "ETF"},
            ]
        )
        self.cfg = HybridSwingConfig(min_avg_dollar_volume20=1_000)
        self.features = build_hybrid_features(daily, universe, cfg=self.cfg)
        self.signal_date = self.dates[sig]
        historical = build_historical_entry_candidates(self.features, self.dates[-1], self.cfg)
        first_test = historical.loc[historical["symbol"].eq("TEST")].iloc[0]
        self.decision_date = pd.Timestamp(first_test["decision_date"])

    def test_detects_exact_industry_theme_ep_pullback_and_excludes_biotech(self) -> None:
        candidates = detect_entry_candidates(self.features, self.decision_date, self.cfg)
        self.assertIn("TEST", set(candidates["symbol"]))
        self.assertNotIn("BIOX", set(candidates["symbol"]))
        row = candidates.loc[candidates["symbol"].eq("TEST")].iloc[0]
        self.assertEqual(row["entry_reason"], ENTRY_REASON)
        self.assertEqual(row["signal_date"], self.signal_date.date().isoformat())

    def test_historical_candidates_use_next_session_entry(self) -> None:
        candidates = build_historical_entry_candidates(self.features, self.decision_date + pd.offsets.BDay(1), self.cfg)
        row = candidates.loc[candidates["symbol"].eq("TEST")].iloc[0]
        self.assertEqual(row["decision_date"], self.decision_date.date().isoformat())
        self.assertEqual(row["entry_date"], (self.decision_date + pd.offsets.BDay(1)).date().isoformat())

    def test_ep_low_stop_exits_position(self) -> None:
        candidates = build_historical_entry_candidates(self.features, self.decision_date + pd.offsets.BDay(1), self.cfg)
        position = candidates.loc[candidates["symbol"].eq("TEST")].head(1).copy()
        signal_low = float(position.iloc[0]["signal_low"])
        entry_date = pd.Timestamp(position.iloc[0]["entry_date"])
        features = self.features.copy()
        stop_date = entry_date + pd.offsets.BDay(2)
        idx = features.index[features["symbol"].eq("TEST") & features["date"].eq(stop_date)]
        features.loc[idx, "low"] = signal_low * 0.99
        holds, exits = evaluate_positions(features, position, stop_date, self.cfg)
        self.assertTrue(holds.empty)
        self.assertEqual(exits.iloc[0]["exit_reason"], "EP安値割れ")


if __name__ == "__main__":
    unittest.main()
