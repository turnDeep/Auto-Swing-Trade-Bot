import os
import time
import datetime
import pytz
import logging
import json
import uuid
import requests
from webullsdkcore.client import ApiClient
from webullsdkcore.common.region import Region
from webullsdktrade.api import API
from webullsdktrade.common.currency import Currency

# Unified config and strategy modules
import config
from strategy import DynamicExitStrategy, check_entry_condition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_webull_trade_client():
    if not config.WEBULL_APP_KEY or not config.WEBULL_APP_SECRET or not config.WEBULL_ACCOUNT_ID:
        logger.error("Webull credentials missing!")
        return None
    api_client = ApiClient(config.WEBULL_APP_KEY, config.WEBULL_APP_SECRET, Region.JP.value)
    trade_api = API(api_client)
    return trade_api

def get_top_10_watchlist():
    try:
        with open('top_10_watchlist.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Could not load watchlist: {e}")
        return []

def place_order(api, symbol, side, qty, order_type="MARKET", limit_price=None):
    client_order_id = uuid.uuid4().hex
    
    new_orders = {
        "client_order_id": client_order_id,
        "symbol": symbol,
        "instrument_type": "EQUITY",
        "market": "US",
        "order_type": order_type,
        "quantity": str(qty),
        "support_trading_session": "N",
        "side": side,
        "time_in_force": "DAY",
        "entrust_type": "QTY",
        "account_tax_type": "SPECIFIC"
    }
    
    if limit_price:
        new_orders["limit_price"] = str(round(limit_price, 2))

    try:
        res = api.order_v2.place_order(account_id=config.WEBULL_ACCOUNT_ID, new_orders=new_orders)
        if res.status_code == 200:
            logger.info(f"Order Placed successfully: {new_orders}")
            return res.json()
        else:
            logger.error(f"Order failed: {res.status_code} - {res.text}")
    except Exception as e:
        logger.error(f"Order exception: {e}")
    return None

def get_buying_power(trade_api):
    try:
        res = trade_api.account.get_account_balance(config.WEBULL_ACCOUNT_ID, 'JPY')
        if res.status_code == 200:
            data = res.json()
            buying_power = 0.0
            if 'account_currency_assets' in data:
                for asset in data['account_currency_assets']:
                    if asset.get('currency') == 'USD':
                        bp = asset.get('buying_power') or asset.get('cash_balance') or asset.get('available_to_exchange', 0)
                        buying_power = float(bp)
                        break
            
            if buying_power > 0:
                logger.info(f"Account Buying Power (USD): ${buying_power:,.2f}")
                return buying_power
            else:
                logger.warning(f"No USD buying power found in response. Payload: {data}")
                return 0.0
        else:
            logger.error(f"Account balance API failed: {res.status_code} - {res.text}")
    except Exception as e:
        logger.error(f"Failed to fetch account balance: {e}")
    return None

def fetch_fmp_prices(symbols):
    """
    Fetches real-time price snapshot using FinancialModelingPrep API.
    """
    if not symbols:
        return {}
    
    symbols_str = ",".join(symbols)
    url = f"https://financialmodelingprep.com/api/v3/quote/{symbols_str}?apikey={config.FMP_API_KEY}"
    
    prices = {}
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            for item in data:
                if 'symbol' in item and 'price' in item:
                    prices[item['symbol']] = float(item['price'])
        else:
            logger.error(f"FMP Quote API returned HTTP {res.status_code}: {res.text}")
    except Exception as e:
        logger.error(f"FMP Quote API request failed: {e}")
        
    return prices

def fetch_fmp_opening_highs(symbols):
    """
    Fetches the highest price observed in the first 5 minutes (09:30-09:35) of today's session.
    Using 1min historical intervals.
    """
    opening_highs = {}
    for sym in symbols:
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{sym}?apikey={config.FMP_API_KEY}"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if not data:
                    continue
                    
                # The data is usually ordered newest to oldest, so find today's date
                today_str = data[0]['date'][:10] 
                
                # Filter for today's opening 5 bars (09:30:00 to 09:34:00)
                opening_bars = []
                for bar in data:
                    if bar['date'].startswith(today_str):
                        time_part = bar['date'][11:]
                        if "09:30:00" <= time_part <= "09:34:00":
                            opening_bars.append(bar['high'])
                            
                if opening_bars:
                    opening_highs[sym] = max(opening_bars)
        except Exception as e:
            logger.error(f"Failed to fetch FMP 1min history for {sym}: {e}")
            
    return opening_highs


def main():
    logger.info("Starting Daily Trading Execution (Webull execution + FMP quotes)")
    trade_api = init_webull_trade_client()
    if not trade_api: return

    watchlist = get_top_10_watchlist()
    if not watchlist:
        logger.error("Watchlist is empty. Aborting.")
        return
        
    logger.info(f"Monitoring TOP 10: {watchlist}")
    
    # Wait until 09:30 AM EST
    ny_tz = pytz.timezone('America/New_York')
    
    while True:
        now = datetime.datetime.now(ny_tz)
        target_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        if now < target_open:
            sleep_time = (target_open - now).total_seconds()
            logger.info(f"Market not open yet. Sleeping for {sleep_time:.0f} seconds.")
            time.sleep(sleep_time)
        else:
            break
            
    logger.info("Market Open! Waiting for 5 minutes to generate Opening Range candles...")
    time.sleep(305) # Wait until 9:35:05 AM
    
    # Fetch Opening Range Highs
    opening_highs = fetch_fmp_opening_highs(watchlist)
    if not opening_highs:
        logger.warning("Falling back to snapshot initial scan.")
        opening_highs = fetch_fmp_prices(watchlist)
        
    logger.info(f"Opening Highs established: {opening_highs}")
    
    active_trade = None
    trade_symbol = None
    trade_qty = 0
    active_strategy = None # Holds DynamicExitStrategy instance
    
    # Fetch real account balance from Webull
    buying_power = get_buying_power(trade_api)
    if buying_power is None or buying_power <= 0:
        logger.error("Could not retrieve buying power. Aborting.")
        return
    
    trade_capital = buying_power * 0.90  # Use 90% of actual balance
    logger.info(f"Trade Capital (90% of ${buying_power:,.2f}): ${trade_capital:,.2f}")
    
    logger.info("Beginning active breakout monitoring...")
    
    while True:
        now = datetime.datetime.now(ny_tz)
        current_prices = fetch_fmp_prices(watchlist)
            
        if not active_trade:
            # Check for breakout logic
            for sym, open_high in opening_highs.items():
                if sym in current_prices:
                    current_price = current_prices[sym]
                    if check_entry_condition(current_price, open_high, now):
                        logger.info(f"*** BREAKOUT DETECTED: {sym} at {current_price} > {open_high} ***")
                        
                        qty = int(trade_capital // current_price)
                        if qty <= 0:
                            logger.warning(f"Insufficient capital for {sym} at ${current_price}")
                            continue
                            
                        logger.info(f"Placing BUY order: {qty} shares of {sym} (${trade_capital:,.2f} / ${current_price:.2f})")
                        res = place_order(trade_api, sym, "BUY", qty, "MARKET")
                        
                        if res:
                            active_trade = res
                            trade_symbol = sym
                            trade_qty = qty
                            # Initialize the stateful shared strategy module
                            active_strategy = DynamicExitStrategy(entry_price=current_price)
                            
                        break # Only take ONE trade per day!
                        
            # If entry window has passed and no trade occurred
            if now.hour >= 10 and now.minute > 30 and not active_trade:
                logger.info("10:30 AM Entry Window Closed. No trades generated today.")
                break
                
        else:
            # Active Trade Management using the stateful strategy tracker
            if trade_symbol in current_prices:
                current_price = current_prices[trade_symbol]
                
                exit_triggered, exit_reason, execute_price = active_strategy.check_exit_condition(current_price, current_time=now)
                
                if exit_triggered:
                    logger.info(f"*** EXIT TRIGGERED: {exit_reason} for {trade_symbol} at ${execute_price:.2f} ***")
                    place_order(trade_api, trade_symbol, "SELL", trade_qty, "MARKET")
                    break
            
        time.sleep(2) # FMP limit allows relatively fast polling

if __name__ == "__main__":
    main()
