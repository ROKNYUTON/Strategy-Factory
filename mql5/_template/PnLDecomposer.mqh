//+------------------------------------------------------------------+
//|                                            PnLDecomposer.mqh    |
//|                          StrategyFactory — DO NOT MODIFY BY AI   |
//|                                                                  |
//| Decomposes closed-position PnL into:                             |
//|   - directional (price movement only)                            |
//|   - swap                                                         |
//|   - commission                                                   |
//|                                                                  |
//| The "WTI lesson" — a strategy with strong total profit but most  |
//| of it from swap is a carry trade in disguise. We MUST track this.|
//+------------------------------------------------------------------+
#property strict

struct PnLBreakdown
  {
   double directional;
   double swap;
   double commission;
   double total;
  };

// Aggregate totals (updated as positions close)
double g_pnl_total_directional = 0.0;
double g_pnl_total_swap        = 0.0;
double g_pnl_total_commission  = 0.0;

//+------------------------------------------------------------------+
//| Reset aggregate counters (call from OnInit)                      |
//+------------------------------------------------------------------+
void PnL_Reset()
  {
   g_pnl_total_directional = 0.0;
   g_pnl_total_swap        = 0.0;
   g_pnl_total_commission  = 0.0;
  }

//+------------------------------------------------------------------+
//| Decompose a single closed deal by ticket                         |
//| For an MT5 position closed via deal: DEAL_PROFIT is the          |
//| directional PnL (price-based), DEAL_SWAP is accumulated swap,    |
//| DEAL_COMMISSION is commission.                                   |
//+------------------------------------------------------------------+
PnLBreakdown PnL_DecomposeDeal(ulong deal_ticket)
  {
   PnLBreakdown b;
   b.directional = 0.0;
   b.swap        = 0.0;
   b.commission  = 0.0;
   b.total       = 0.0;

   if(!HistoryDealSelect(deal_ticket))
      return(b);

   b.directional = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
   b.swap        = HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
   b.commission  = HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
   b.total       = b.directional + b.swap + b.commission;
   return(b);
  }

//+------------------------------------------------------------------+
//| Decompose a closed POSITION (sum of all its deals)               |
//+------------------------------------------------------------------+
PnLBreakdown PnL_DecomposePosition(ulong position_id)
  {
   PnLBreakdown b;
   b.directional = 0.0;
   b.swap        = 0.0;
   b.commission  = 0.0;
   b.total       = 0.0;

   HistorySelectByPosition(position_id);
   int n = HistoryDealsTotal();
   for(int i = 0; i < n; i++)
     {
      ulong t = HistoryDealGetTicket(i);
      if(t == 0) continue;
      b.directional += HistoryDealGetDouble(t, DEAL_PROFIT);
      b.swap        += HistoryDealGetDouble(t, DEAL_SWAP);
      b.commission  += HistoryDealGetDouble(t, DEAL_COMMISSION);
     }
   b.total = b.directional + b.swap + b.commission;

   // Update aggregate
   g_pnl_total_directional += b.directional;
   g_pnl_total_swap        += b.swap;
   g_pnl_total_commission  += b.commission;
   return(b);
  }

//+------------------------------------------------------------------+
//| Get aggregate breakdown                                          |
//+------------------------------------------------------------------+
PnLBreakdown PnL_GetAggregate()
  {
   PnLBreakdown b;
   b.directional = g_pnl_total_directional;
   b.swap        = g_pnl_total_swap;
   b.commission  = g_pnl_total_commission;
   b.total       = b.directional + b.swap + b.commission;
   return(b);
  }

//+------------------------------------------------------------------+
//| Print aggregate breakdown to journal (call in OnTester or end)   |
//+------------------------------------------------------------------+
void PnL_PrintAggregate()
  {
   PnLBreakdown b = PnL_GetAggregate();
   double pct_dir = (b.total != 0.0) ? (b.directional / b.total * 100.0) : 0.0;
   double pct_swp = (b.total != 0.0) ? (b.swap        / b.total * 100.0) : 0.0;
   double pct_cmm = (b.total != 0.0) ? (b.commission  / b.total * 100.0) : 0.0;

   PrintFormat("[PnL_Decomp] Directional: %.2f (%.1f%%)", b.directional, pct_dir);
   PrintFormat("[PnL_Decomp] Swap:        %.2f (%.1f%%)", b.swap,        pct_swp);
   PrintFormat("[PnL_Decomp] Commission:  %.2f (%.1f%%)", b.commission,  pct_cmm);
   PrintFormat("[PnL_Decomp] TOTAL:       %.2f",          b.total);

   if(b.total > 0.0 && pct_dir < 60.0)
      PrintFormat("[PnL_Decomp] WARNING: directional %.1f%% < 60%% threshold (WTI lesson)", pct_dir);
  }
//+------------------------------------------------------------------+
