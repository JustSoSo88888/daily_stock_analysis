"""
===================================
WaveCounter - 波浪理论预计算引擎
===================================

对 K 线数据进行波峰波谷检测、浪型分类、铁律验证和斐波那契计算。
纯 numpy/pandas 实现，无外部依赖。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WavePoint:
    date: str
    price: float
    index: int
    is_peak: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"date": self.date, "price": round(self.price, 2), "type": "peak" if self.is_peak else "trough"}


@dataclass
class WaveSegment:
    label: str
    start: WavePoint
    end: WavePoint
    direction: str
    amplitude: float
    amplitude_pct: float
    duration_bars: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "start_date": self.start.date, "start_price": round(self.start.price, 2),
            "end_date": self.end.date, "end_price": round(self.end.price, 2),
            "direction": self.direction,
            "change": f"{'+' if self.direction == 'up' else ''}{round(self.amplitude_pct, 2)}%",
            "bars": self.duration_bars,
        }


@dataclass
class IronRuleResult:
    rule_id: int
    description: str
    passed: bool
    detail: str


@dataclass
class WaveAnalysis:
    stock_code: str
    timeframe: str
    points: List[WavePoint] = field(default_factory=list)
    waves: List[WaveSegment] = field(default_factory=list)
    iron_rules: List[IronRuleResult] = field(default_factory=list)
    fibonacci_levels: Dict[str, Any] = field(default_factory=dict)
    current_position: str = ""
    trend_direction: str = "neutral"
    confidence: str = "low"
    price_range: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stock_code": self.stock_code,
            "timeframe": self.timeframe,
            "trend_direction": self.trend_direction,
            "current_position": self.current_position,
            "confidence": self.confidence,
            "price_range": self.price_range,
            "wave_count": len(self.waves),
            "waves": [w.to_dict() for w in self.waves],
            "iron_rules": [{"id": r.rule_id, "description": r.description,
                            "passed": r.passed, "detail": r.detail} for r in self.iron_rules],
            "fibonacci_levels": self.fibonacci_levels,
        }


def _find_peaks_troughs(
    prices: np.ndarray, dates: np.ndarray,
    min_distance: int = 5, min_change_pct: float = 2.0,
) -> List[WavePoint]:
    """滑动窗口检测局部波峰和波谷。"""
    n = len(prices)
    if n < min_distance * 2:
        return []

    points: List[WavePoint] = []
    last_peak_idx: int = -min_distance
    last_trough_idx: int = -min_distance
    last_peak_price: Optional[float] = None
    last_trough_price: Optional[float] = None
    threshold = min_change_pct / 100.0

    for i in range(min_distance, n - min_distance):
        window_before = prices[i - min_distance : i]
        window_after = prices[i + 1 : i + min_distance + 1]
        current = prices[i]

        is_max = all(current > wb for wb in window_before) and all(current >= wa for wa in window_after)
        is_min = all(current < wb for wb in window_before) and all(current <= wa for wa in window_after)

        if is_max and (i - last_peak_idx >= min_distance):
            if last_peak_price is not None:
                change = abs(current - last_peak_price) / last_peak_price
                if change < threshold:
                    continue
            points.append(WavePoint(date=str(dates[i])[:10], price=float(current), index=i, is_peak=True))
            last_peak_idx = i
            last_peak_price = current

        elif is_min and (i - last_trough_idx >= min_distance):
            if last_trough_price is not None:
                change = abs(current - last_trough_price) / last_trough_price
                if change < threshold:
                    continue
            points.append(WavePoint(date=str(dates[i])[:10], price=float(current), index=i, is_peak=False))
            last_trough_idx = i
            last_trough_price = current

    return points


def _build_waves(points: List[WavePoint]) -> List[WaveSegment]:
    """将交替的波峰波谷点串联成浪段。"""
    if len(points) < 2:
        return []
    ordered = sorted(points, key=lambda p: p.index)
    waves = []
    for i in range(len(ordered) - 1):
        a, b = ordered[i], ordered[i + 1]
        direction = "up" if b.price > a.price else "down"
        amp = abs(b.price - a.price)
        amp_pct = (abs(b.price - a.price) / a.price) * 100 if a.price else 0
        waves.append(WaveSegment(
            label=f"W{i+1}", start=a, end=b, direction=direction,
            amplitude=amp, amplitude_pct=amp_pct, duration_bars=b.index - a.index,
        ))
    return waves


def _classify_waves(waves: List[WaveSegment]) -> Tuple[str, str]:
    """分类浪型，需通过交替验证。"""
    if not waves:
        return "neutral", "无法判断"

    for i in range(len(waves) - 1):
        if waves[i].direction == waves[i + 1].direction:
            return "neutral", "浪型交替异常，无法分类"

    up = [w for w in waves if w.direction == "up"]
    down = [w for w in waves if w.direction == "down"]

    if len(up) >= 3 and len(down) >= 2 and waves[0].direction == "up":
        last_up = up[-1]
        prev_up = up[-2] if len(up) > 1 else None
        if last_up.amplitude > (prev_up.amplitude if prev_up else 0):
            return "up", "第3浪推进中（主升浪）"
        return "up", "第5浪推进中（末升段）"
    if len(down) >= 3 and len(up) >= 2 and waves[0].direction == "down":
        return "down", "下跌推动浪进行中"
    if up:
        return "up", "推动浪进行中"
    return "down", "调整或下跌中"


def _verify_iron_rules(waves: List[WaveSegment], points: List[WavePoint]) -> List[IronRuleResult]:
    """验证波浪理论三大铁律。根据起点方向自适应标签。"""
    results = []
    ordered = sorted(points, key=lambda p: p.index)
    if not ordered or not waves:
        results.extend([
            IronRuleResult(1, "铁律1: 第2浪不能跌破第1浪起点", False, "无数据"),
            IronRuleResult(2, "铁律2: 第3浪不能是最短推动浪", False, "无数据"),
            IronRuleResult(3, "铁律3: 第4浪不与第1浪高点重叠", False, "无数据"),
        ])
        return results

    up_waves = [w for w in waves if w.direction == "up"]
    down_waves = [w for w in waves if w.direction == "down"]

    # Detect: if first point is a peak, we are in a downtrend context
    # impulse (1-3-5) and corrective (2-4) indices invert
    starts_with_peak = ordered[0].is_peak
    if starts_with_peak:
        imp = down_waves
        corr = up_waves
        trend_label = "下跌推动"
    else:
        imp = up_waves
        corr = down_waves
        trend_label = "上涨推动"

    enough = len(imp) >= 3 and len(corr) >= 2

    # Rule 1: Wave 2 not below Wave 1 start
    if len(imp) >= 2 and len(corr) >= 1:
        w1 = imp[0]
        w2 = corr[0]
        ok = w2.end.price > w1.start.price if not starts_with_peak else w2.end.price < w1.start.price
        results.append(IronRuleResult(1, "铁律1: 第2浪不能跌破第1浪起点", ok,
            f"1浪起点{w1.start.price:.2f}, 2浪终点{w2.end.price:.2f} {'> ✅' if ok else '<= ❌'} [{trend_label}]"))
    else:
        results.append(IronRuleResult(1, "铁律1: 第2浪不能跌破第1浪起点", False, "浪段不足，无法验证"))

    # Rule 2: Wave 3 not shortest
    if len(imp) >= 3:
        amps = [imp[0].amplitude, imp[1].amplitude, imp[2].amplitude]
        w3_shortest = amps[1] == min(amps) and amps.count(min(amps)) == 1
        results.append(IronRuleResult(2, "铁律2: 第3浪不能是最短推动浪", not w3_shortest,
            f"1浪={amps[0]:.2f} 3浪={amps[1]:.2f} 5浪={amps[2]:.2f} {'✅' if not w3_shortest else '❌ 3浪最短，五浪被证伪'}"))
    else:
        results.append(IronRuleResult(2, "铁律2: 第3浪不能是最短推动浪", False, "浪段不足，无法验证"))

    # Rule 3: Wave 4 not overlap Wave 1 top
    if len(imp) >= 3 and len(corr) >= 2:
        w1 = imp[0]
        w4 = corr[1]
        ok = w4.end.price > w1.end.price if not starts_with_peak else w4.end.price < w1.end.price
        results.append(IronRuleResult(3, "铁律3: 第4浪不与第1浪高点重叠", ok,
            f"1浪高点{w1.end.price:.2f}, 4浪低点{w4.end.price:.2f} {'> ✅' if ok else '<= ❌ 重叠，五浪被证伪'} [{trend_label}]"))
    else:
        results.append(IronRuleResult(3, "铁律3: 第4浪不与第1浪高点重叠", False, "浪段不足，无法验证"))

    return results


def _compute_fibonacci(waves: List[WaveSegment], points: List[WavePoint]) -> Dict[str, Any]:
    """计算斐波那契回撤和扩展位。"""
    ordered = sorted(points, key=lambda p: p.index)
    if len(ordered) < 2:
        return {}
    prices = [p.price for p in ordered]
    hi, lo = max(prices), min(prices)
    diff = hi - lo
    if diff <= 0:
        return {}
    trend_up = ordered[-1].price > ordered[0].price
    def level(ratio):
        return round(hi - diff * ratio, 2) if trend_up else round(lo + diff * ratio, 2)
    fib = {
        "range_high": round(hi, 2), "range_low": round(lo, 2),
        "trend": "up" if trend_up else "down",
        "0.382": level(0.382), "0.5": level(0.5), "0.618": level(0.618),
        "1.618": level(1.618), "2.618": level(2.618),
    }
    cur = ordered[-1].price
    hits = [r for r in ["0.382", "0.5", "0.618"] if abs(cur - fib[r]) < diff * 0.015]
    fib["current_near_levels"] = hits if hits else "none"
    return fib


def _assess_confidence(waves: List[WaveSegment], iron_rules: List[IronRuleResult]) -> str:
    if not waves:
        return "low"
    all_ok = all(r.passed for r in iron_rules)
    if all_ok and len(waves) >= 5:
        return "high"
    if all_ok and len(waves) >= 3:
        return "medium"
    return "low"


def analyze_waves(
    df: pd.DataFrame, stock_code: str = "", timeframe: str = "day",
    min_distance: int = 5, min_change_pct: float = 2.0,
) -> WaveAnalysis:
    """
    分析 K 线数据的波浪结构。

    Args:
        df: K 线 DataFrame，需含 date/close 列
        stock_code: 股票代码
        timeframe: 时间框架 (day/60m/30m/15m/5m)
        min_distance: 峰谷最小 K 线间距
        min_change_pct: 最小涨跌幅阈值 (%)
    """
    analysis = WaveAnalysis(stock_code=stock_code, timeframe=timeframe)

    if df is None or df.empty or len(df) < min_distance * 2:
        analysis.current_position = "数据不足，无法进行波浪分析"
        return analysis

    date_col = next((c for c in ["date", "时间"] if c in df.columns), None)
    close_col = next((c for c in ["close", "收盘价"] if c in df.columns), None)
    high_col = next((c for c in ["high", "最高价"] if c in df.columns), None)
    low_col = next((c for c in ["low", "最低价"] if c in df.columns), None)

    if not date_col or not close_col:
        analysis.current_position = "数据列名不匹配"
        return analysis

    dates = df[date_col].values
    closes = df[close_col].values.astype(float)
    highs = df[high_col].values.astype(float) if high_col else closes
    lows = df[low_col].values.astype(float) if low_col else closes

    analysis.points = _find_peaks_troughs(closes, dates, min_distance, min_change_pct)

    if len(analysis.points) < 3:
        analysis.current_position = f"仅检测到{len(analysis.points)}个转折点，不足以构成波浪结构"
        return analysis

    analysis.waves = _build_waves(analysis.points)
    analysis.iron_rules = _verify_iron_rules(analysis.waves, analysis.points)
    analysis.fibonacci_levels = _compute_fibonacci(analysis.waves, analysis.points)
    analysis.trend_direction, analysis.current_position = _classify_waves(analysis.waves)
    analysis.confidence = _assess_confidence(analysis.waves, analysis.iron_rules)

    if closes.size > 0:
        analysis.price_range = {
            "current": round(float(closes[-1]), 2),
            "high": round(float(np.max(highs)), 2),
            "low": round(float(np.min(lows)), 2),
        }

    return analysis


def analyze_multi_timeframe(
    daily_df: pd.DataFrame, hourly_df: Optional[pd.DataFrame], stock_code: str = "",
) -> Dict[str, Any]:
    """多时间框架波浪分析。"""
    result: Dict[str, Any] = {}
    daily = analyze_waves(daily_df, stock_code, "day", min_distance=5, min_change_pct=2.0)
    result["daily"] = daily.to_dict()

    if hourly_df is not None and not hourly_df.empty:
        hourly = analyze_waves(hourly_df, stock_code, "60m", min_distance=8, min_change_pct=1.0)
        result["intraday"] = hourly.to_dict()
        if daily.confidence == "high" and hourly.confidence in ("high", "medium"):
            result["combined_confidence"] = "high"
        elif daily.confidence == "high" or hourly.confidence in ("high", "medium"):
            result["combined_confidence"] = "medium"
        else:
            result["combined_confidence"] = "low"
        dp, hp = daily.current_position, hourly.current_position
        if "3浪" in dp and "3" in hp:
            result["fractal_note"] = "日线第3浪 + 小时线子浪推进，分形共振"
        elif "5浪" in dp and "5" in hp:
            result["fractal_note"] = "日线第5浪 + 小时线末段，注意顶背离"
        else:
            result["fractal_note"] = "日线与小时线浪型不一致，分形验证不足"
    else:
        result["combined_confidence"] = daily.confidence

    return result
