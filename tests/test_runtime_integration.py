from __future__ import annotations

import tempfile
import unittest
import gc
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from stallion.buying_power_manager import compute_order_quantity
from stallion.config import load_settings
from stallion.discord_notifier import DiscordNotifier
from stallion.live_trader import (
    _build_close_summary_lines,
    _build_fill_lines,
    _build_order_submitted_lines,
    _build_pre_market_lines,
    _cancel_stale_orders,
    _compute_close_summary,
    _submit_order_with_fallback,
)
from stallion.order_state import PositionSlot, normalize_order_status
from stallion.slot_manager import SlotManager
from stallion.storage import SQLiteParquetStore


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeDiscordSession:
    def __init__(self) -> None:
        self.get_calls: list[str] = []
        self.post_calls: list[tuple[str, str]] = []

    def get(self, url: str, headers=None, timeout: int = 15):
        self.get_calls.append(url)
        return _FakeResponse(200, {"id": "bot-1", "username": "stallion"})

    def post(self, url: str, headers=None, data=None, timeout: int = 15):
        self.post_calls.append((url, data or ""))
        return _FakeResponse(200, {"id": "msg-1"})


class _MemoryStore:
    def __init__(self) -> None:
        self.notifications: list[dict] = []

    def append_discord_notification(self, **kwargs):
        self.notifications.append(kwargs)


class _CancelBroker:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.is_demo = False

    def cancel_order(self, *, client_order_id: str):
        self.cancelled.append(client_order_id)
        return {"client_order_id": client_order_id, "status_code": 200}


class _FallbackBroker:
    def __init__(self, market_status: int = 500, limit_status: int = 200) -> None:
        self.market_status = market_status
        self.limit_status = limit_status
        self.market_calls = 0
        self.limit_calls = 0

    def place_market_order(self, *, symbol: str, side: str, quantity: int):
        self.market_calls += 1
        return {"status_code": self.market_status, "client_order_id": "mkt-1", "order_type": "MARKET"}

    def place_marketable_limit_order(self, *, symbol: str, side: str, quantity: int, limit_price: float):
        self.limit_calls += 1
        return {
            "status_code": self.limit_status,
            "client_order_id": f"lim-{self.limit_calls}",
            "order_type": "LIMIT",
            "limit_price": limit_price,
        }


class RuntimeIntegrationTests(unittest.TestCase):
    def _make_settings(self, root: Path, *, live: bool, discord: bool) -> object:
        env = {"FMP_API_KEY": "test-fmp"}
        if live:
            env.update(
                {
                    "WEBULL_APP_KEY": "app",
                    "WEBULL_APP_SECRET": "secret",
                    "WEBULL_ACCOUNT_ID": "acct",
                }
            )
        if discord:
            env.update({"DISCORD_BOT_TOKEN": "token", "DISCORD_CHANNEL_ID": "channel"})
        with patch.dict("os.environ", env, clear=True):
            return load_settings(root)

    def test_missing_webull_credentials_enable_demo_mode(self):
        with tempfile.TemporaryDirectory() as tempdir:
            settings = self._make_settings(Path(tempdir), live=False, discord=False)
            self.assertTrue(settings.demo_mode)
            self.assertEqual(settings.trade_mode, "DEMO")

    def test_complete_webull_credentials_enable_live_mode(self):
        with tempfile.TemporaryDirectory() as tempdir:
            settings = self._make_settings(Path(tempdir), live=True, discord=False)
            self.assertFalse(settings.demo_mode)
            self.assertEqual(settings.trade_mode, "LIVE")

    def test_pre_market_notification_is_sent(self):
        with tempfile.TemporaryDirectory() as tempdir:
            settings = self._make_settings(Path(tempdir), live=False, discord=True)
            store = _MemoryStore()
            session = _FakeDiscordSession()
            notifier = DiscordNotifier(settings, store, session=session)
            self.addCleanup(notifier.close)
            notifier.notify("Pre-market status", _build_pre_market_lines(settings=settings, buying_power=12345.67, threshold=0.83))
            notifier.flush()
            self.assertEqual(len(store.notifications), 1)
            self.assertTrue(store.notifications[0]["delivered"])
            self.assertIn("Pre-market status", store.notifications[0]["payload"]["content"])

    def test_buy_order_notification_is_sent(self):
        with tempfile.TemporaryDirectory() as tempdir:
            settings = self._make_settings(Path(tempdir), live=False, discord=True)
            store = _MemoryStore()
            session = _FakeDiscordSession()
            notifier = DiscordNotifier(settings, store, session=session)
            self.addCleanup(notifier.close)
            notifier.notify(
                "BUY ORDER SUBMITTED",
                _build_order_submitted_lines(
                    symbol="NVDA",
                    quantity=3,
                    expected_price=912.5,
                    score=0.917,
                    threshold=0.83,
                    slot_id=3,
                    signal_reason="daily_buy_pressure_prev=0.9",
                ),
            )
            notifier.flush()
            self.assertEqual(len(store.notifications), 1)
            self.assertIn("NVDA", store.notifications[0]["payload"]["content"])
            self.assertIn("slot_id: 3", store.notifications[0]["payload"]["content"])

    def test_fill_notification_is_sent(self):
        with tempfile.TemporaryDirectory() as tempdir:
            settings = self._make_settings(Path(tempdir), live=False, discord=True)
            store = _MemoryStore()
            session = _FakeDiscordSession()
            notifier = DiscordNotifier(settings, store, session=session)
            self.addCleanup(notifier.close)
            notifier.notify(
                "BUY FILLED",
                _build_fill_lines(
                    symbol="AAPL",
                    qty_filled=10,
                    avg_fill_price=214.33,
                    filled_at="2026-03-19 09:37:15 ET",
                    partial_fill=False,
                    remaining_qty=0,
                    slot_id=1,
                ),
            )
            notifier.flush()
            self.assertEqual(len(store.notifications), 1)
            self.assertIn("AAPL", store.notifications[0]["payload"]["content"])
            self.assertIn("qty_filled: 10", store.notifications[0]["payload"]["content"])

    def test_market_close_notification_is_sent(self):
        with tempfile.TemporaryDirectory() as tempdir:
            settings = self._make_settings(Path(tempdir), live=False, discord=True)
            store = _MemoryStore()
            session = _FakeDiscordSession()
            notifier = DiscordNotifier(settings, store, session=session)
            self.addCleanup(notifier.close)
            summary = {
                "all_positions_closed": True,
                "remaining_positions": 0,
                "today_pnl": 423.18,
                "cumulative_pnl": 6214.90,
                "fills_today": 7,
                "wins_today": 4,
                "losses_today": 3,
                "canceled_orders_today": 2,
                "failed_orders_today": 0,
                "max_drawdown": -55.25,
            }
            notifier.notify("MARKET CLOSE SUMMARY", _build_close_summary_lines(summary))
            notifier.flush()
            self.assertEqual(len(store.notifications), 1)
            self.assertIn("cumulative_pnl_since_deploy", store.notifications[0]["payload"]["content"])

    def test_high_priced_symbol_is_skipped_when_budget_too_small(self):
        decision = compute_order_quantity(
            slot_budget=800.0,
            effective_buying_power=800.0,
            expected_price=1050.0,
            fractional_shares_enabled=False,
        )
        self.assertEqual(decision.quantity, 0)
        self.assertEqual(decision.reason, "per_slot_budget_below_share_price")

    def test_quantity_is_reduced_when_buying_power_is_tighter_than_slot_budget(self):
        decision = compute_order_quantity(
            slot_budget=2500.0,
            effective_buying_power=1100.0,
            expected_price=500.0,
            fractional_shares_enabled=False,
        )
        self.assertEqual(decision.quantity, 2)

    def test_stale_unfilled_order_is_cancel_requested_and_slot_not_released(self):
        with tempfile.TemporaryDirectory() as tempdir:
            settings = self._make_settings(Path(tempdir), live=False, discord=False)
            store = SQLiteParquetStore(settings)
            session_date = pd.Timestamp("2026-03-19")
            placed_at = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=settings.runtime.order_cancel_after_seconds + 10)).isoformat()
            store.upsert_live_order(
                {
                    "client_order_id": "ord-1",
                    "session_date": str(session_date.date()),
                    "symbol": "TSLA",
                    "side": "BUY",
                    "quantity": 5,
                    "filled_quantity": 0,
                    "requested_price": 250.0,
                    "status": "SUBMITTED",
                    "broker_order_id": None,
                    "placed_at": placed_at,
                    "updated_at": placed_at,
                    "payload_json": "{\"slot_id\": 1, \"reserved_buying_power\": 1250.0}",
                }
            )
            slot_manager = SlotManager([PositionSlot(slot_id=1, status="BUY_PENDING", symbol="TSLA", client_order_id="ord-1", requested_quantity=5, reserved_buying_power=1250.0)], max_positions=4)
            broker = _CancelBroker()
            notifier = DiscordNotifier(settings, _MemoryStore(), session=_FakeDiscordSession())
            self.addCleanup(notifier.close)
            _cancel_stale_orders(store, broker, session_date, settings, notifier)
            orders = store.load_live_orders(session_date=session_date)
            self.assertEqual(orders.iloc[0]["status"], "CANCEL_REQUESTED")
            slot_manager.sync_from_orders_and_positions(orders, pd.DataFrame(columns=["symbol"]))
            self.assertNotEqual(slot_manager.get_slot(1).status, "AVAILABLE")
            del store
            gc.collect()

    def test_partial_fill_keeps_slot_occupied(self):
        slot_manager = SlotManager([PositionSlot(slot_id=1, status="BUY_PENDING", symbol="AAPL", client_order_id="ord-1", requested_quantity=10, reserved_buying_power=2000.0)], max_positions=4)
        orders = pd.DataFrame(
            [
                {
                    "client_order_id": "ord-1",
                    "symbol": "AAPL",
                    "side": "BUY",
                    "quantity": 10,
                    "filled_quantity": 4,
                    "status": "PARTIALLY_FILLED",
                    "payload_json": "{\"slot_id\": 1}",
                }
            ]
        )
        slot_manager.sync_from_orders_and_positions(orders, pd.DataFrame(columns=["symbol"]))
        self.assertEqual(slot_manager.get_slot(1).status, "PARTIALLY_FILLED")

    def test_market_order_failure_falls_back_to_marketable_limit(self):
        with tempfile.TemporaryDirectory() as tempdir:
            settings = self._make_settings(Path(tempdir), live=False, discord=True)
            broker = _FallbackBroker(market_status=500, limit_status=200)
            notifier = DiscordNotifier(settings, _MemoryStore(), session=_FakeDiscordSession())
            self.addCleanup(notifier.close)
            result = _submit_order_with_fallback(
                broker,
                symbol="TSLA",
                side="BUY",
                quantity=2,
                expected_price=250.0,
                settings=settings,
                notifier=notifier,
            )
            self.assertEqual(result["order_type"], "LIMIT")
            self.assertEqual(broker.market_calls, 1)
            self.assertEqual(broker.limit_calls, 1)

    def test_pending_cancel_status_is_not_treated_as_free_slot(self):
        normalized = normalize_order_status("CANCEL_REQUESTED", quantity=10, filled_quantity=0)
        self.assertEqual(normalized, "CANCEL_REQUESTED")
        slot_manager = SlotManager([PositionSlot(slot_id=1, status="BUY_PENDING", symbol="MSFT", client_order_id="ord-2", requested_quantity=10, reserved_buying_power=1000.0)], max_positions=4)
        orders = pd.DataFrame(
            [
                {
                    "client_order_id": "ord-2",
                    "symbol": "MSFT",
                    "side": "BUY",
                    "quantity": 10,
                    "filled_quantity": 0,
                    "status": "CANCEL_REQUESTED",
                    "payload_json": "{\"slot_id\": 1}",
                }
            ]
        )
        slot_manager.sync_from_orders_and_positions(orders, pd.DataFrame(columns=["symbol"]))
        self.assertEqual(slot_manager.available_slots, 3)
        self.assertEqual(slot_manager.get_slot(1).status, "BUY_PENDING")

    def test_close_summary_contains_cumulative_pnl(self):
        with tempfile.TemporaryDirectory() as tempdir:
            settings = self._make_settings(Path(tempdir), live=False, discord=False)
            store = SQLiteParquetStore(settings)
            store.save_daily_trade_summary(
                session_date=pd.Timestamp("2026-03-18"),
                mode="DEMO",
                today_pnl=100.0,
                cumulative_pnl=100.0,
                fills_today=1,
                wins_today=1,
                losses_today=0,
                canceled_orders_today=0,
                failed_orders_today=0,
                remaining_positions=0,
                all_positions_closed=True,
                max_drawdown=0.0,
                payload={"test": True},
            )
            store.append_live_fill(
                {
                    "fill_id": "buy-1",
                    "session_date": "2026-03-19",
                    "symbol": "NVDA",
                    "side": "BUY",
                    "timestamp": "2026-03-19T14:35:00Z",
                    "quantity": 1,
                    "price": 100.0,
                    "payload_json": "{}",
                }
            )
            store.append_live_fill(
                {
                    "fill_id": "sell-1",
                    "session_date": "2026-03-19",
                    "symbol": "NVDA",
                    "side": "SELL",
                    "timestamp": "2026-03-19T19:55:00Z",
                    "quantity": 1,
                    "price": 110.0,
                    "payload_json": "{}",
                }
            )
            summary = _compute_close_summary(store, pd.Timestamp("2026-03-19"), settings)
            self.assertGreater(summary["cumulative_pnl"], 100.0)
            self.assertIn("cumulative_pnl_since_deploy", "\n".join(_build_close_summary_lines(summary)))
            del store
            gc.collect()


if __name__ == "__main__":
    unittest.main()
