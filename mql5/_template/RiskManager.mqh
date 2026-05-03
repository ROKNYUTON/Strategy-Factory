//+------------------------------------------------------------------+
//|                                            RiskManager.mqh       |
//|                          StrategyFactory — DO NOT MODIFY BY AI   |
//|                                                                  |
//| Position sizing, daily loss check, drawdown circuit breaker,     |
//| concurrent position limits.                                      |
//|                                                                  |
//| Public API:                                                      |
//|   double RiskMgr_CalcLot(string symbol, double sl_distance_pts,  |
//|                          double risk_pct, double fixed_lot,      |
//|                          ENUM_SIZING_METHOD method);             |
//|   bool   RiskMgr_DailyLossOK(double max_daily_loss_pct);         |
//|   bool   RiskMgr_DrawdownOK(double max_dd_pct);                  |
//|   int    RiskMgr_OpenPositions(long magic, string symbol = "");  |
//+------------------------------------------------------------------+
#property strict

enum ENUM_SIZING_METHOD
  {
   SIZING_FIXED_LOT       = 0,
   SIZING_RISK_PCT        = 1,
   SIZING_KELLY_QUARTER   = 2
  };

//+------------------------------------------------------------------+
//| Calculate lot size based on sizing method                        |
//| sl_distance_pts: stop-loss distance in POINTS (not pips)         |
//+------------------------------------------------------------------+
double RiskMgr_CalcLot(string symbol,
                       double sl_distance_pts,
                       double risk_pct,
                       double fixed_lot,
                       ENUM_SIZING_METHOD method)
  {
   if(method == SIZING_FIXED_LOT)
      return(NormalizeLot(symbol, fixed_lot));

   if(sl_distance_pts <= 0.0)
     {
      Print("[RiskMgr] sl_distance_pts <= 0, falling back to fixed lot 0.01");
      return(NormalizeLot(symbol, 0.01));
     }

   double equity     = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_money = equity * (risk_pct / 100.0);

   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double point      = SymbolInfoDouble(symbol, SYMBOL_POINT);

   if(tick_size <= 0.0 || point <= 0.0)
     {
      Print("[RiskMgr] invalid symbol info for ", symbol);
      return(NormalizeLot(symbol, 0.01));
     }

   // Risk per 1 lot = sl_distance_pts * point / tick_size * tick_value
   double risk_per_lot = sl_distance_pts * point / tick_size * tick_value;
   if(risk_per_lot <= 0.0)
      return(NormalizeLot(symbol, 0.01));

   double lot = risk_money / risk_per_lot;

   // Quarter Kelly = halve again from raw Kelly. We treat SIZING_KELLY_QUARTER
   // as SIZING_RISK_PCT with the input risk_pct already pre-scaled to Quarter.
   // (No additional scaling here — keep upstream control.)

   return(NormalizeLot(symbol, lot));
  }

//+------------------------------------------------------------------+
//| Normalize lot to broker constraints                              |
//+------------------------------------------------------------------+
double NormalizeLot(string symbol, double lot)
  {
   double min_lot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_lot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double step_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   if(step_lot <= 0.0)
      step_lot = 0.01;

   lot = MathFloor(lot / step_lot) * step_lot;
   lot = MathMax(min_lot, MathMin(max_lot, lot));
   return(lot);
  }

//+------------------------------------------------------------------+
//| Check daily loss limit                                           |
//+------------------------------------------------------------------+
bool RiskMgr_DailyLossOK(double max_daily_loss_pct)
  {
   if(max_daily_loss_pct <= 0.0)
      return(true);

   datetime now    = TimeTradeServer();
   MqlDateTime dt;
   TimeToStruct(now, dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   datetime day_start = StructToTime(dt);

   double balance_today_start = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity              = AccountInfoDouble(ACCOUNT_EQUITY);

   // Sum closed P&L since day start
   double closed_pnl = 0.0;
   HistorySelect(day_start, now);
   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      datetime t = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
      if(t < day_start) continue;
      closed_pnl += HistoryDealGetDouble(ticket, DEAL_PROFIT)
                  + HistoryDealGetDouble(ticket, DEAL_SWAP)
                  + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
     }

   double balance_at_day_start = balance_today_start - closed_pnl;
   if(balance_at_day_start <= 0.0) return(true);

   double pct_loss = -closed_pnl / balance_at_day_start * 100.0;
   if(pct_loss >= max_daily_loss_pct)
     {
      PrintFormat("[RiskMgr] Daily loss %.2f%% >= limit %.2f%% — blocking new trades",
                  pct_loss, max_daily_loss_pct);
      return(false);
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Check overall drawdown circuit breaker                           |
//+------------------------------------------------------------------+
bool RiskMgr_DrawdownOK(double max_dd_pct)
  {
   if(max_dd_pct <= 0.0)
      return(true);

   double equity   = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance  = AccountInfoDouble(ACCOUNT_BALANCE);
   if(balance <= 0.0) return(true);

   double dd_pct = (balance - equity) / balance * 100.0;
   if(dd_pct >= max_dd_pct)
     {
      PrintFormat("[RiskMgr] Equity DD %.2f%% >= limit %.2f%% — blocking new trades",
                  dd_pct, max_dd_pct);
      return(false);
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Count open positions for a magic number (optionally per symbol)  |
//+------------------------------------------------------------------+
int RiskMgr_OpenPositions(long magic, string symbol = "")
  {
   int count = 0;
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != magic) continue;
      if(symbol != "" && PositionGetString(POSITION_SYMBOL) != symbol) continue;
      count++;
     }
   return(count);
  }
//+------------------------------------------------------------------+
