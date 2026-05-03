//+------------------------------------------------------------------+
//|                                                  Logger.mqh     |
//|                          StrategyFactory — DO NOT MODIFY BY AI   |
//|                                                                  |
//| CSV trade logger with PnL decomposition, regime flag, and        |
//| holding-time tracking. Output: MQL5/Files/StrategyFactory_logs/  |
//+------------------------------------------------------------------+
#property strict

enum ENUM_LOG_LEVEL
  {
   LOG_DEBUG = 0,
   LOG_INFO  = 1,
   LOG_WARN  = 2,
   LOG_ERROR = 3
  };

string g_log_strategy_id   = "UNKNOWN";
string g_log_run_timestamp = "";
int    g_log_handle        = INVALID_HANDLE;
ENUM_LOG_LEVEL g_log_level = LOG_INFO;

//+------------------------------------------------------------------+
//| Initialize logger. Call from OnInit.                             |
//+------------------------------------------------------------------+
bool Logger_Init(string strategy_id, ENUM_LOG_LEVEL level = LOG_INFO)
  {
   g_log_strategy_id = strategy_id;
   g_log_level       = level;

   MqlDateTime dt;
   TimeToStruct(TimeLocal(), dt);
   g_log_run_timestamp = StringFormat("%04d%02d%02d_%02d%02d%02d",
                                       dt.year, dt.mon, dt.day,
                                       dt.hour, dt.min, dt.sec);

   string folder = "StrategyFactory_logs";
   string fname  = folder + "\\" + strategy_id + "_" + g_log_run_timestamp + ".csv";

   g_log_handle = FileOpen(fname, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(g_log_handle == INVALID_HANDLE)
     {
      PrintFormat("[Logger] Cannot open file %s. Error %d", fname, GetLastError());
      return(false);
     }

   // Header — keep stable so analysis/report_parser.py can rely on it.
   FileWrite(g_log_handle,
             "ts_open", "ts_close", "symbol", "direction", "magic",
             "lots", "entry_price", "exit_price", "sl", "tp",
             "profit_directional", "profit_swap", "profit_commission",
             "profit_total", "holding_bars", "exit_reason", "regime_flag",
             "comment");

   FileFlush(g_log_handle);
   return(true);
  }

//+------------------------------------------------------------------+
//| Close logger. Call from OnDeinit.                                |
//+------------------------------------------------------------------+
void Logger_Close()
  {
   if(g_log_handle != INVALID_HANDLE)
     {
      FileClose(g_log_handle);
      g_log_handle = INVALID_HANDLE;
     }
  }

//+------------------------------------------------------------------+
//| Log a closed position with decomposed PnL                        |
//+------------------------------------------------------------------+
void Logger_LogTrade(datetime ts_open, datetime ts_close, string symbol,
                     string direction, long magic, double lots,
                     double entry_price, double exit_price,
                     double sl, double tp,
                     double profit_directional, double profit_swap,
                     double profit_commission,
                     int holding_bars, string exit_reason,
                     string regime_flag, string comment)
  {
   if(g_log_handle == INVALID_HANDLE) return;

   double total = profit_directional + profit_swap + profit_commission;

   FileWrite(g_log_handle,
             TimeToString(ts_open, TIME_DATE | TIME_SECONDS),
             TimeToString(ts_close, TIME_DATE | TIME_SECONDS),
             symbol, direction, IntegerToString(magic),
             DoubleToString(lots, 2),
             DoubleToString(entry_price, 5),
             DoubleToString(exit_price, 5),
             DoubleToString(sl, 5),
             DoubleToString(tp, 5),
             DoubleToString(profit_directional, 2),
             DoubleToString(profit_swap, 2),
             DoubleToString(profit_commission, 2),
             DoubleToString(total, 2),
             IntegerToString(holding_bars),
             exit_reason, regime_flag, comment);
   FileFlush(g_log_handle);
  }

//+------------------------------------------------------------------+
//| Conventional log message                                         |
//+------------------------------------------------------------------+
void Logger_Msg(ENUM_LOG_LEVEL level, string msg)
  {
   if(level < g_log_level) return;
   string prefix = "[INFO]";
   if(level == LOG_DEBUG) prefix = "[DEBUG]";
   if(level == LOG_WARN)  prefix = "[WARN]";
   if(level == LOG_ERROR) prefix = "[ERROR]";
   Print(prefix, " ", g_log_strategy_id, " ", msg);
  }
//+------------------------------------------------------------------+
