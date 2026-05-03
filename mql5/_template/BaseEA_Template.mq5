//+------------------------------------------------------------------+
//|                                          BaseEA_Template.mq5    |
//|                                StrategyFactory — TEMPLATE        |
//|                                                                  |
//| AI generation rules:                                             |
//|   1. Replace the placeholder block between the                   |
//|      "// === AI GENERATED LOGIC START ===" and                   |
//|      "// === AI GENERATED LOGIC END ===" markers ONLY.           |
//|   2. NEVER modify includes, OnInit/OnDeinit/OnTick scaffolding,  |
//|      or risk/logger/decomposer infrastructure.                   |
//|   3. Use ONLY built-in MQL5 indicators (iRSI, iBands, iATR,      |
//|      iMA, iMACD, iStochastic) unless the spec requires custom.   |
//|   4. Add inline comments referencing each YAML rule.             |
//|   5. Implement the "AI EDIT" function bodies marked with TODO.   |
//+------------------------------------------------------------------+
#property copyright "StrategyFactory"
#property version   "1.00"
#property strict

#include "RiskManager.mqh"
#include "Logger.mqh"
#include "PnLDecomposer.mqh"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//+------------------------------------------------------------------+
//| Strategy metadata (filled from YAML by ea_generator.py)          |
//+------------------------------------------------------------------+
#define STRATEGY_ID    "{{STRATEGY_ID}}"
#define STRATEGY_NAME  "{{STRATEGY_NAME}}"
#define STRATEGY_MAGIC {{STRATEGY_MAGIC}}

//+------------------------------------------------------------------+
//| Inputs — AI may add strategy-specific inputs at the end          |
//+------------------------------------------------------------------+
input group "=== Strategy Identity ==="
input string  InpStrategyId         = STRATEGY_ID;
input long    InpMagic              = STRATEGY_MAGIC;
input ENUM_LOG_LEVEL InpLogLevel    = LOG_INFO;

input group "=== Risk Management ==="
input ENUM_SIZING_METHOD InpSizingMethod = SIZING_RISK_PCT;
input double  InpFixedLot           = 0.01;
input double  InpRiskPerTradePct    = 0.5;
input int     InpMaxConcurrent      = 2;
input int     InpMaxConcurrentPerSym= 1;
input double  InpMaxDailyLossPct    = 2.0;
input double  InpMaxDDPct           = 10.0;
input int     InpMaxSlippagePts     = 20;

input group "=== Session ==="
input string  InpSessionStartUTC    = "22:00";
input string  InpSessionEndUTC      = "06:00";
input bool    InpSessionCrossesMidnight = true;

input group "=== Exit ==="
input double  InpStopLossATRMult    = 1.5;
input double  InpTakeProfitATRMult  = 2.0;
input int     InpATRPeriod          = 14;
input bool    InpTimeExitEnabled    = true;
input int     InpMaxHoldingBars     = 16;

input group "=== Visuals ==="
input bool    InpDrawLevels         = true;

// === AI GENERATED LOGIC START ===
// AI EDIT: Add strategy-specific inputs here (e.g., RSI period, BB deviations).
// Example:
//   input int     InpRSIPeriod         = 14;
//   input double  InpRSIOverbought     = 75.0;
//   input double  InpRSIOversold       = 25.0;
//   input int     InpBBPeriod          = 20;
//   input double  InpBBDeviations      = 2.0;
// === AI GENERATED LOGIC END ===

//+------------------------------------------------------------------+
//| Globals                                                          |
//+------------------------------------------------------------------+
CTrade        Trade;
CPositionInfo PosInfo;

datetime g_last_bar_time = 0;
int      g_atr_handle    = INVALID_HANDLE;

// Tracking position open times for time-based exit
struct OpenPosTracker
  {
   ulong    ticket;
   datetime open_time;
   int      open_bar_count;
  };
OpenPosTracker g_pos_trackers[];

// === AI GENERATED LOGIC START ===
// AI EDIT: Indicator handles (init in OnInit, release in OnDeinit).
// Example:
//   int g_rsi_handle = INVALID_HANDLE;
//   int g_bb_handle  = INVALID_HANDLE;
// === AI GENERATED LOGIC END ===


//+------------------------------------------------------------------+
//| OnInit                                                           |
//+------------------------------------------------------------------+
int OnInit()
  {
   Logger_Init(InpStrategyId, InpLogLevel);
   PnL_Reset();

   Trade.SetExpertMagicNumber(InpMagic);
   Trade.SetDeviationInPoints(InpMaxSlippagePts);
   Trade.SetTypeFillingBySymbol(_Symbol);

   g_atr_handle = iATR(_Symbol, _Period, InpATRPeriod);
   if(g_atr_handle == INVALID_HANDLE)
     {
      Logger_Msg(LOG_ERROR, "Failed to create ATR handle");
      return(INIT_FAILED);
     }

   // === AI GENERATED LOGIC START ===
   // AI EDIT: Initialize strategy indicator handles. Return INIT_FAILED on error.
   // Example:
   //   g_rsi_handle = iRSI(_Symbol, _Period, InpRSIPeriod, PRICE_CLOSE);
   //   if(g_rsi_handle == INVALID_HANDLE) return(INIT_FAILED);
   //   g_bb_handle  = iBands(_Symbol, _Period, InpBBPeriod, 0, InpBBDeviations, PRICE_CLOSE);
   //   if(g_bb_handle == INVALID_HANDLE) return(INIT_FAILED);
   // === AI GENERATED LOGIC END ===

   Logger_Msg(LOG_INFO, "Initialized successfully");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| OnDeinit                                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(g_atr_handle != INVALID_HANDLE) IndicatorRelease(g_atr_handle);

   // === AI GENERATED LOGIC START ===
   // AI EDIT: Release strategy indicator handles.
   // Example:
   //   if(g_rsi_handle != INVALID_HANDLE) IndicatorRelease(g_rsi_handle);
   //   if(g_bb_handle  != INVALID_HANDLE) IndicatorRelease(g_bb_handle);
   // === AI GENERATED LOGIC END ===

   PnL_PrintAggregate();
   Logger_Close();
  }

//+------------------------------------------------------------------+
//| OnTick — main loop                                               |
//+------------------------------------------------------------------+
void OnTick()
  {
   // 1. Bar gating: only process logic on new bar close
   datetime cur_bar = iTime(_Symbol, _Period, 0);
   if(cur_bar == g_last_bar_time) {
      ManageOpenPositions();
      return;
   }
   g_last_bar_time = cur_bar;

   // 2. Risk gates
   if(!RiskMgr_DailyLossOK(InpMaxDailyLossPct))   return;
   if(!RiskMgr_DrawdownOK(InpMaxDDPct))           return;
   if(!IsTradingSession())                        return;

   // 3. Concurrent positions check
   int total_open       = RiskMgr_OpenPositions(InpMagic);
   int open_this_symbol = RiskMgr_OpenPositions(InpMagic, _Symbol);
   bool can_open = (total_open       < InpMaxConcurrent) &&
                   (open_this_symbol < InpMaxConcurrentPerSym);

   // 4. Manage existing
   ManageOpenPositions();

   // 5. Entry logic — AI section below
   if(!can_open) return;

   double atr[];
   if(CopyBuffer(g_atr_handle, 0, 0, 1, atr) <= 0) return;
   double current_atr = atr[0];
   if(current_atr <= 0.0) return;

   // === AI GENERATED LOGIC START ===
   // AI EDIT: Implement entry conditions per YAML spec.
   //
   // Required functions to implement below:
   //   bool CheckLongEntry()  — returns true if long conditions met
   //   bool CheckShortEntry() — returns true if short conditions met
   //   (they are stub-defined later in this file under "AI HOOKS")
   //
   // Then in this block, decide whether to OpenPosition and which side:
   //   if(CheckLongEntry())  OpenPosition(POSITION_TYPE_BUY,  current_atr);
   //   if(CheckShortEntry()) OpenPosition(POSITION_TYPE_SELL, current_atr);

   // Default: do nothing until AI fills the entry logic.
   // === AI GENERATED LOGIC END ===
  }

//+------------------------------------------------------------------+
//| Trading-session check                                            |
//+------------------------------------------------------------------+
bool IsTradingSession()
  {
   datetime now = TimeTradeServer();
   MqlDateTime dt;
   TimeToStruct(now, dt);

   int cur_min = dt.hour * 60 + dt.min;
   int s_h = (int)StringToInteger(StringSubstr(InpSessionStartUTC, 0, 2));
   int s_m = (int)StringToInteger(StringSubstr(InpSessionStartUTC, 3, 2));
   int e_h = (int)StringToInteger(StringSubstr(InpSessionEndUTC,   0, 2));
   int e_m = (int)StringToInteger(StringSubstr(InpSessionEndUTC,   3, 2));
   int start_min = s_h * 60 + s_m;
   int end_min   = e_h * 60 + e_m;

   if(InpSessionCrossesMidnight)
      return(cur_min >= start_min || cur_min < end_min);
   return(cur_min >= start_min && cur_min < end_min);
  }

//+------------------------------------------------------------------+
//| Open a position with calculated SL/TP                            |
//+------------------------------------------------------------------+
bool OpenPosition(ENUM_POSITION_TYPE side, double current_atr)
  {
   double price = (side == POSITION_TYPE_BUY)
                  ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                  : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(point <= 0.0) return(false);

   double sl_dist = current_atr * InpStopLossATRMult;
   double tp_dist = current_atr * InpTakeProfitATRMult;
   double sl, tp;

   if(side == POSITION_TYPE_BUY) { sl = price - sl_dist; tp = price + tp_dist; }
   else                           { sl = price + sl_dist; tp = price - tp_dist; }

   double sl_dist_pts = sl_dist / point;
   double lot = RiskMgr_CalcLot(_Symbol, sl_dist_pts, InpRiskPerTradePct,
                                InpFixedLot, InpSizingMethod);
   if(lot <= 0.0) {
      Logger_Msg(LOG_WARN, "Calculated lot <= 0, skip");
      return(false);
   }

   bool ok = (side == POSITION_TYPE_BUY)
             ? Trade.Buy(lot, _Symbol, price, sl, tp, STRATEGY_NAME)
             : Trade.Sell(lot, _Symbol, price, sl, tp, STRATEGY_NAME);

   if(!ok) {
      Logger_Msg(LOG_ERROR, StringFormat("Trade failed code=%d", Trade.ResultRetcode()));
      return(false);
   }

   ulong ticket = Trade.ResultDeal();
   AddPosTracker(ticket, TimeTradeServer());

   if(InpDrawLevels) DrawTradeLevels(ticket, price, sl, tp);
   return(true);
  }

//+------------------------------------------------------------------+
//| Manage open positions: time exit + trailing if enabled           |
//+------------------------------------------------------------------+
void ManageOpenPositions()
  {
   int total = PositionsTotal();
   for(int i = total - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      // Time-based exit
      if(InpTimeExitEnabled) {
         int bars_held = BarsHeldByTicket(ticket);
         if(bars_held >= InpMaxHoldingBars) {
            Logger_Msg(LOG_INFO, StringFormat("Time exit ticket=%I64u bars=%d",
                                              ticket, bars_held));
            Trade.PositionClose(ticket);
            OnPositionClosed(ticket, "time_exit");
            continue;
         }
      }

      // === AI GENERATED LOGIC START ===
      // AI EDIT (optional): Add custom exit logic here.
      // Example: trailing stop after price has moved 1 ATR in profit.
      // === AI GENERATED LOGIC END ===
   }
  }

//+------------------------------------------------------------------+
//| Track position opens/closes for time-based exit and decomp        |
//+------------------------------------------------------------------+
void AddPosTracker(ulong ticket, datetime open_time)
  {
   int n = ArraySize(g_pos_trackers);
   ArrayResize(g_pos_trackers, n + 1);
   g_pos_trackers[n].ticket          = ticket;
   g_pos_trackers[n].open_time       = open_time;
   g_pos_trackers[n].open_bar_count  = (int)Bars(_Symbol, _Period);
  }

int BarsHeldByTicket(ulong ticket)
  {
   ulong pos_ticket = PositionGetInteger(POSITION_TICKET);
   for(int i = 0; i < ArraySize(g_pos_trackers); i++) {
      if(g_pos_trackers[i].ticket == pos_ticket || g_pos_trackers[i].ticket == ticket) {
         int now_bars = (int)Bars(_Symbol, _Period);
         return(now_bars - g_pos_trackers[i].open_bar_count);
      }
   }
   return(0);
  }

//+------------------------------------------------------------------+
//| OnTradeTransaction — capture closes for decomp + log              |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest&     request,
                        const MqlTradeResult&      result)
  {
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(!HistoryDealSelect(trans.deal)) return;
   if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != InpMagic) return;
   long entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY) return;

   ulong pos_id = HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);
   OnPositionClosed(pos_id, "natural_close");
  }

//+------------------------------------------------------------------+
//| Common close handler — decomp + log                              |
//+------------------------------------------------------------------+
void OnPositionClosed(ulong position_id, string exit_reason)
  {
   PnLBreakdown b = PnL_DecomposePosition(position_id);

   // Find trackers; pick latest matching
   datetime t_open = 0;
   int      bars_held = 0;
   for(int i = 0; i < ArraySize(g_pos_trackers); i++) {
      if(g_pos_trackers[i].ticket == position_id) {
         t_open    = g_pos_trackers[i].open_time;
         bars_held = (int)Bars(_Symbol, _Period) - g_pos_trackers[i].open_bar_count;
         break;
      }
   }
   if(t_open == 0) t_open = TimeTradeServer();

   // Find first entry deal for entry price
   double entry_price = 0.0, lots = 0.0, sl = 0.0, tp = 0.0;
   string direction = "?";
   HistorySelectByPosition(position_id);
   int n = HistoryDealsTotal();
   for(int i = 0; i < n; i++) {
      ulong dt = HistoryDealGetTicket(i);
      long e = HistoryDealGetInteger(dt, DEAL_ENTRY);
      if(e == DEAL_ENTRY_IN) {
         entry_price = HistoryDealGetDouble(dt, DEAL_PRICE);
         lots        = HistoryDealGetDouble(dt, DEAL_VOLUME);
         long type   = HistoryDealGetInteger(dt, DEAL_TYPE);
         direction   = (type == DEAL_TYPE_BUY) ? "long" : "short";
         break;
      }
   }
   double exit_price = 0.0;
   datetime t_close  = TimeTradeServer();
   for(int i = n - 1; i >= 0; i--) {
      ulong dt = HistoryDealGetTicket(i);
      long e = HistoryDealGetInteger(dt, DEAL_ENTRY);
      if(e == DEAL_ENTRY_OUT || e == DEAL_ENTRY_OUT_BY) {
         exit_price = HistoryDealGetDouble(dt, DEAL_PRICE);
         t_close    = (datetime)HistoryDealGetInteger(dt, DEAL_TIME);
         break;
      }
   }

   Logger_LogTrade(t_open, t_close, _Symbol, direction, InpMagic, lots,
                   entry_price, exit_price, sl, tp,
                   b.directional, b.swap, b.commission,
                   bars_held, exit_reason, "regime_unknown",
                   "");
  }

//+------------------------------------------------------------------+
//| Visual debugging — draw entry/SL/TP on chart                     |
//+------------------------------------------------------------------+
void DrawTradeLevels(ulong ticket, double entry, double sl, double tp)
  {
   string base = StringFormat("SF_%I64u_", ticket);
   ObjectCreate(0, base + "ENTRY", OBJ_HLINE, 0, 0, entry);
   ObjectSetInteger(0, base + "ENTRY", OBJPROP_COLOR, clrAqua);
   ObjectSetInteger(0, base + "ENTRY", OBJPROP_STYLE, STYLE_DOT);

   ObjectCreate(0, base + "SL", OBJ_HLINE, 0, 0, sl);
   ObjectSetInteger(0, base + "SL", OBJPROP_COLOR, clrRed);
   ObjectSetInteger(0, base + "SL", OBJPROP_STYLE, STYLE_DASH);

   ObjectCreate(0, base + "TP", OBJ_HLINE, 0, 0, tp);
   ObjectSetInteger(0, base + "TP", OBJPROP_COLOR, clrLime);
   ObjectSetInteger(0, base + "TP", OBJPROP_STYLE, STYLE_DASH);
  }

//+------------------------------------------------------------------+
//| OnTester — return custom Sharpe + decomp data                    |
//+------------------------------------------------------------------+
double OnTester()
  {
   PnLBreakdown b = PnL_GetAggregate();
   PnL_PrintAggregate();

   double profit  = TesterStatistics(STAT_PROFIT);
   double trades  = TesterStatistics(STAT_TRADES);
   double sharpe  = TesterStatistics(STAT_SHARPE_RATIO);

   // Custom score: penalize swap-heavy edges (WTI lesson)
   double pct_dir = (b.total != 0.0) ? (b.directional / b.total * 100.0) : 0.0;
   double swap_penalty = (pct_dir < 60.0) ? 0.5 : 1.0;

   if(trades < 50) return(0.0);

   double score = sharpe * swap_penalty;
   PrintFormat("[OnTester] sharpe=%.3f pct_dir=%.1f%% penalty=%.2f score=%.3f",
               sharpe, pct_dir, swap_penalty, score);
   return(score);
  }

//+------------------------------------------------------------------+
//| AI HOOKS — implemented by AI generation                          |
//+------------------------------------------------------------------+

// === AI GENERATED LOGIC START ===
// AI EDIT: Implement these two function bodies.
// CheckLongEntry / CheckShortEntry must return true ONLY when ALL
// conditions in the YAML entry_rules.long / entry_rules.short are met.
//
// Default stub: returns false so the EA never trades until AI fills it.
//
// Example template (Asian MR FX):
//
//   bool CheckLongEntry() {
//      double rsi[];
//      if(CopyBuffer(g_rsi_handle, 0, 0, 1, rsi) <= 0) return(false);
//      double lower[];
//      if(CopyBuffer(g_bb_handle, LOWER_BAND, 0, 1, lower) <= 0) return(false);
//      double close = iClose(_Symbol, _Period, 1);
//      if(rsi[0] < InpRSIOversold && close < lower[0]) return(true);
//      return(false);
//   }

bool CheckLongEntry()
  {
   return(false); // stub
  }

bool CheckShortEntry()
  {
   return(false); // stub
  }
// === AI GENERATED LOGIC END ===
//+------------------------------------------------------------------+
