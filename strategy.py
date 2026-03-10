import numpy as np
import pandas as pd
from datetime import datetime, time

class DynamicExitStrategy:
    """
    A stateful strategy class used to track dynamic exit thresholds (Break-even, Trailing Stop) 
    for an active live trade, or for evaluating historical data.
    Defaults are based on the robust optimal holdout research:
    - break_even_r: 0.75
    - trail_start_r: 1.5
    - trail_offset_r: 0.75
    - initial_stop_pct: 0.03 (3%)
    """
    def __init__(self, entry_price, initial_stop_pct=0.03, break_even_r=0.75, trail_start_r=1.5, trail_offset_r=0.75):
        self.entry_price = float(entry_price)
        self.initial_stop_pct = initial_stop_pct
        
        self.r_val = entry_price * initial_stop_pct
        self.stop_loss = entry_price - self.r_val
        
        self.break_even_price = entry_price + (self.r_val * break_even_r)
        self.trail_start_price = entry_price + (self.r_val * trail_start_r)
        self.trail_offset_r = trail_offset_r
        
        self.highest_price = entry_price
        self.break_even_activated = False
        self.trailing_activated = False

    def check_exit_condition(self, current_price, current_time: datetime = None, eod_hour=15, eod_minute=55):
        """
        Evaluates the current price tick against the dynamic thresholds.
        Returns (True, "Reason", execute_price) if exit is triggered, else (False, None, None).
        current_time should be timezone-aware (US/Eastern) or localized properly.
        """
        self.highest_price = max(self.highest_price, current_price)
        
        # 1. Update Thresholds State
        if not self.trailing_activated and self.highest_price >= self.trail_start_price:
            self.trailing_activated = True
            
        if not self.trailing_activated and not self.break_even_activated and self.highest_price >= self.break_even_price:
            self.break_even_activated = True
            self.stop_loss = max(self.stop_loss, self.entry_price)

        if self.trailing_activated:
            current_trail_stop = self.highest_price - (self.r_val * self.trail_offset_r)
            self.stop_loss = max(self.stop_loss, current_trail_stop)
            
        # 2. Check Execution
        if current_price <= self.stop_loss:
            reason = "Stop Loss"
            if self.trailing_activated:
                reason = "Trailing Stop"
            elif self.break_even_activated and self.stop_loss >= self.entry_price:
                reason = "Break-Even Stop"
            return True, reason, current_price
            
        # 3. Time Stop (End of Day Liquidation)
        if current_time:
            if current_time.hour == eod_hour and current_time.minute >= eod_minute:
                return True, "EOD Close", current_price
            elif current_time.hour > eod_hour:
                return True, "EOD Close", current_price
                
        return False, None, None

def check_entry_condition(current_price, open_high, current_time: datetime):
    """
    Baseline Entry Rule:
    Buy if current price > opening 5-min high AND time is between 09:35 and 10:30.
    Returns True if entry condition is met.
    """
    if current_time.hour == 9 and current_time.minute < 35:
        return False
    if current_time.hour >= 10 and current_time.minute > 30:
        return False
    if current_time.hour >= 11:
        return False
        
    return current_price > open_high
