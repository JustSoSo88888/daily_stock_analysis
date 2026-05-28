"""
===================================
ThsFetcher - 同花顺 THS SDK 数据源
===================================

数据来源：同花顺 THS SDK (thsdk)
特点：需安装 thsdk 包，游客模式免费使用
优势：实时行情丰富（大单/盘口/竞价/板块），分钟K线支持
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS

logger = logging.getLogger(__name__)


def _to_ths_code(stock_code: str) -> Optional[str]:
    """将纯数字代码转换为 THSCODE 格式（如 002625 → USZA002625）"""
    code = str(stock_code).strip()
    if code.startswith(("USHA", "USZA", "USTM")):
        return code
    if code.startswith("6"):
        return f"USHA{code}"
    if code.startswith(("0", "3")):
        return f"USZA{code}"
    if code.startswith(("8", "4")):
        return f"USTM{code}"
    return None


class ThsFetcher(BaseFetcher):
    """
    同花顺 THS SDK 数据源

    游客模式免费使用，无需账号配置。
    初始化成功自动提升为最高优先级 (P-2)。
    """

    name = "ThsFetcher"

    def __init__(self):
        self._ths = None
        self._available = False
        self._init_api()

    def _init_api(self):
        try:
            from thsdk import THS
            self._ths = THS()
            self._ths.connect()
            self._available = True
            logger.info("THS SDK 初始化成功（游客模式）")
        except ImportError:
            logger.warning("THS SDK 未安装 (thsdk)，跳过 ThsFetcher。安装: pip install thsdk")
        except Exception as e:
            logger.warning(f"THS SDK 初始化失败: {e}")

    def is_available(self) -> bool:
        return self._available and self._ths is not None

    @property
    def priority(self):
        return -2 if self.is_available() else 99

    @priority.setter
    def priority(self, value):
        pass

    def _ths_call(self, method_name: str, *args, **kwargs):
        if not self.is_available():
            raise DataFetchError("THS SDK 不可用")
        method = getattr(self._ths, method_name)
        resp = method(*args, **kwargs)
        if not resp or resp.df.empty:
            raise DataFetchError(f"THS {method_name} 返回空数据")
        return resp.df

    # ---- K 线数据 ----

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        ths_code = _to_ths_code(stock_code)
        if not ths_code:
            raise DataFetchError(f"不支持的股票代码: {stock_code}")

        count = 60
        try:
            s = datetime.strptime(start_date, "%Y-%m-%d")
            e = datetime.strptime(end_date, "%Y-%m-%d")
            count = max(30, (e - s).days + 10)
        except Exception:
            pass

        df = self._ths_call("klines", ths_code, interval="day", count=count, adjust="forward")

        if "时间" in df.columns:
            df["时间"] = pd.to_datetime(df["时间"])
            mask = (df["时间"] >= start_date) & (df["时间"] <= end_date)
            df = df[mask]

        return df

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.copy()

        col_map = {
            "时间": "date",
            "开盘价": "open",
            "最高价": "high",
            "最低价": "low",
            "收盘价": "close",
            "成交量": "volume",
            "总金额": "amount",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # THS klines 不返回 pct_chg，根据 close 差值计算
        if "close" in df.columns and "pct_chg" not in df.columns:
            df["prev_close"] = df["close"].shift(1)
            df["pct_chg"] = ((df["close"] - df["prev_close"]) / df["prev_close"] * 100).round(2)
            df.drop(columns=["prev_close"], inplace=True)

        df["code"] = stock_code

        keep = ["code"] + [c for c in STANDARD_COLUMNS if c in df.columns]
        return df[keep]

    # ---- 实时行情 ----

    def get_realtime_quote(self, stock_code: str, *, log_final_failure: bool = True):
        from .realtime_types import UnifiedRealtimeQuote, RealtimeSource

        ths_code = _to_ths_code(stock_code)
        if not ths_code:
            return None

        try:
            df = self._ths_call("market_data_cn", ths_code, "汇总")
            row = df.iloc[0] if not df.empty else None
            if row is None:
                return None

            def _f(key, default=0.0):
                v = row.get(key)
                return float(v) if v is not None else default

            return UnifiedRealtimeQuote(
                code=stock_code,
                name=str(row.get("名称", "")),
                source=RealtimeSource.THS,
                price=_f("价格"),
                change_amount=_f("涨跌"),
                change_pct=_f("涨幅"),
                open_price=_f("开盘价"),
                high=_f("最高价"),
                low=_f("最低价"),
                pre_close=_f("昨收价"),
                volume=_f("成交量"),
                amount=_f("总金额"),
                volume_ratio=_f("量比"),
                turnover_rate=_f("换手率"),
                pe_ratio=_f("市盈率TTM"),
                pb_ratio=None,
                total_mv=_f("总市值"),
                circ_mv=_f("流通市值"),
                amplitude=_f("振幅"),
                limit_up=_f("涨停价"),
                limit_down=_f("跌停价"),
                main_net_inflow=_f("主力净流入"),
                main_net_amount=_f("主力净量"),
            )
        except Exception as e:
            if log_final_failure:
                logger.warning(f"[ThsFetcher] 获取 {stock_code} 实时行情失败: {str(e)[:120]}")
            return None

    # ---- 股票名称 ----

    def get_stock_name(self, stock_code: str, allow_realtime: bool = True) -> Optional[str]:
        ths_code = _to_ths_code(stock_code)
        if not ths_code:
            return None
        try:
            df = self._ths_call("market_data_cn", ths_code, "基础数据3")
            name = df.iloc[0].get("名称", None) if not df.empty else None
            if name:
                return str(name)
        except Exception:
            pass
        return None

    # ---- 大盘指数 ----

    def get_main_indices(self, region: str = "cn") -> Optional[List[Dict[str, Any]]]:
        if region != "cn":
            return None
        codes = [
            "USHI000001", "USZI399001", "USZI399006",
            "USHI000688", "USHI000300", "USHI000905",
        ]
        try:
            df = self._ths_call("market_data_index", codes)
            results = []
            for _, row in df.iterrows():
                results.append({
                    "code": row.get("代码", ""),
                    "name": row.get("名称", ""),
                    "current": float(row.get("价格", 0) or 0),
                    "change": float(row.get("涨跌", 0) or 0),
                    "change_pct": float(row.get("涨幅", 0) or 0),
                    "volume": float(row.get("成交量", 0) or 0),
                    "amount": float(row.get("总金额", 0) or 0),
                })
            return results
        except Exception as e:
            logger.warning(f"[ThsFetcher] 获取指数行情失败: {e}")
            return None

    def close(self):
        if self._ths:
            try:
                self._ths.close()
            except Exception:
                pass
            self._ths = None

    def __del__(self):
        self.close()
