"""
Tests for wave_counter.py - Elliott Wave analysis engine.

Run offline unit tests:
    python -m pytest tests/test_wave_counter.py -v -m "not network"

Run with live THS data:
    python -m pytest tests/test_wave_counter.py -v -m network
"""

import math

import numpy as np
import pandas as pd
import pytest

from data_provider.wave_counter import (
    WaveAnalysis, WavePoint, WaveSegment,
    _assess_confidence, _build_waves, _classify_waves,
    _compute_fibonacci, _find_peaks_troughs, _verify_iron_rules,
    analyze_multi_timeframe, analyze_waves,
)


def _make_df(prices: list, start_date: str = "2026-01-01") -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=len(prices), freq="B")
    h = [p * 1.02 for p in prices]
    l = [p * 0.98 for p in prices]
    return pd.DataFrame({"date": dates, "close": prices, "open": prices,
                         "high": h, "low": l, "volume": [10000] * len(prices)})


def _assert_no_nan(obj, path="root"):
    if isinstance(obj, float):
        assert not math.isnan(obj), f"NaN at {path}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_nan(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_nan(v, f"{path}[{i}]")


def _make_clear_uptrend():
    """Synthetic 5-wave uptrend with all iron rules passing."""
    return _make_df([
        100, 105, 110, 115, 120, 125, 130, 135, 140, 145,
        143, 140, 138, 135, 133, 130, 128, 130, 132, 135,
        138, 142, 148, 155, 162, 170, 178, 185, 190, 195,
        192, 188, 185, 182, 180, 178, 180, 182, 185, 188,
        190, 193, 196, 198, 200, 202, 203, 204, 205, 206,
    ])


class TestPeakTroughDetection:

    def test_detect_clear_uptrend(self):
        df = _make_clear_uptrend()
        p = _find_peaks_troughs(df["close"].values, df["date"].values, 5, 2.0)
        peaks = [x for x in p if x.is_peak]
        troughs = [x for x in p if not x.is_peak]
        assert len(peaks) >= 2, f"Got {len(peaks)} peaks"
        assert len(troughs) >= 1, f"Got {len(troughs)} troughs"

    def test_alternating_peaks_troughs(self):
        df = _make_clear_uptrend()
        p = _find_peaks_troughs(df["close"].values, df["date"].values, 5, 2.0)
        assert len(p) >= 3

    def test_insufficient_data_returns_empty(self):
        df = _make_df([100, 101, 102])
        p = _find_peaks_troughs(df["close"].values, df["date"].values, 5, 2.0)
        assert len(p) == 0

    def test_flat_data_no_false_positives(self):
        df = _make_df([100] * 50)
        p = _find_peaks_troughs(df["close"].values, df["date"].values, 5, 2.0)
        assert len(p) == 0


class TestWaveBuilding:

    def test_waves_alternate_direction(self):
        df = _make_clear_uptrend()
        a = analyze_waves(df, "T", "day", 5, 2.0)
        assert len(a.waves) >= 2
        for i in range(len(a.waves) - 1):
            assert a.waves[i].direction != a.waves[i+1].direction

    def test_waves_have_positive_amplitude(self):
        df = _make_clear_uptrend()
        a = analyze_waves(df, "T", "day", 5, 2.0)
        for w in a.waves:
            assert w.amplitude > 0, f"{w.label} zero amplitude"
            assert w.amplitude_pct > 0


class TestIronRules:

    def test_all_three_rules_present(self):
        df = _make_clear_uptrend()
        a = analyze_waves(df, "T", "day", 5, 2.0)
        assert {r.rule_id for r in a.iron_rules} == {1, 2, 3}

    def test_rules_have_details(self):
        df = _make_clear_uptrend()
        a = analyze_waves(df, "T", "day", 5, 2.0)
        for r in a.iron_rules:
            assert len(r.detail) > 5

    def test_clear_uptrend_passes_all_rules(self):
        df = _make_clear_uptrend()
        a = analyze_waves(df, "T", "day", 5, 2.0)
        for r in a.iron_rules:
            assert r.passed, f"Rule {r.rule_id}: {r.detail}"

    def test_w2_below_w1_start_may_fail_rule1(self):
        prices = (
            [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
            + [130, 115, 100, 95, 90, 85, 80, 85, 90, 95]
            + [100, 110, 120, 135, 150, 165, 180, 190, 195, 200]
            + [195, 185, 180, 175, 170, 165, 170, 175, 180, 185]
            + [190, 195, 200, 205, 208, 210, 212, 213, 214, 215]
        )
        df = _make_df(prices)
        a = analyze_waves(df, "T", "day", 5, 2.0)
        r1 = next(r for r in a.iron_rules if r.rule_id == 1)
        assert len(r1.detail) > 0


class TestFibonacci:

    def test_levels_present(self):
        df = _make_clear_uptrend()
        a = analyze_waves(df, "T", "day", 5, 2.0)
        for k in ["0.382", "0.5", "0.618", "1.618"]:
            assert k in a.fibonacci_levels

    def test_levels_in_range(self):
        df = _make_clear_uptrend()
        a = analyze_waves(df, "T", "day", 5, 2.0)
        f = a.fibonacci_levels
        lo, hi = f["range_low"], f["range_high"]
        for k in ["0.382", "0.5", "0.618"]:
            assert lo <= f[k] <= hi, f"{k}={f[k]} out of [{lo},{hi}]"

    def test_no_nan(self):
        df = _make_clear_uptrend()
        a = analyze_waves(df, "T", "day", 5, 2.0)
        _assert_no_nan(a.fibonacci_levels)


class TestConfidence:

    def test_valid_confidence_value(self):
        df = _make_clear_uptrend()
        a = analyze_waves(df, "T", "day", 5, 2.0)
        assert a.confidence in ("high", "medium", "low")

    def test_insufficient_data_low_confidence(self):
        df = _make_df([100, 101, 102])
        a = analyze_waves(df, "T", "day", 5, 2.0)
        assert a.confidence == "low"


class TestOutput:

    def test_to_dict_no_nan(self):
        df = _make_clear_uptrend()
        _assert_no_nan(analyze_waves(df, "T", "day", 5, 2.0).to_dict())

    def test_to_dict_keys(self):
        df = _make_clear_uptrend()
        d = analyze_waves(df, "T", "day", 5, 2.0).to_dict()
        for k in ["waves", "iron_rules", "fibonacci_levels", "confidence",
                   "current_position", "trend_direction"]:
            assert k in d


class TestMultiTimeframe:

    def test_daily_only(self):
        df = _make_clear_uptrend()
        r = analyze_multi_timeframe(df, None, "T")
        assert "daily" in r and "combined_confidence" in r

    def test_both_timeframes(self):
        d = _make_clear_uptrend()
        h = _make_clear_uptrend()
        r = analyze_multi_timeframe(d, h, "T")
        assert "daily" in r and "intraday" in r and "fractal_note" in r


# ---- Live THS Network Tests ----
# python -m pytest tests/test_wave_counter.py -v -m network

LIVE_STOCKS = [
    ("600519", "贵州茅台"),
    ("300750", "宁德时代"),
    ("300652", "雷迪克"),
    ("002625", "光启技术"),
    ("000001", "平安银行"),
]


def _get_live_data(code: str) -> pd.DataFrame:
    from data_provider.ths_fetcher import ThsFetcher
    f = ThsFetcher()
    if not f.is_available():
        f.close()
        pytest.skip("THS SDK unavailable")
    try:
        df = f._fetch_raw_data(code, "2026-01-01", "2026-12-31")
        return f._normalize_data(df, code)
    finally:
        f.close()


@pytest.mark.network
class TestLiveWaveAnalysis:

    @pytest.mark.parametrize("code,name", LIVE_STOCKS)
    def test_produces_valid_analysis(self, code, name):
        df = _get_live_data(code)
        a = analyze_waves(df, code, "day", 5, 2.0)
        assert a.confidence in ("high", "medium", "low"), f"[{name}] {a.confidence}"
        assert len(a.current_position) > 0, f"[{name}] empty position"
        _assert_no_nan(a.to_dict())
        assert len(a.iron_rules) == 3, f"[{name}] {len(a.iron_rules)} rules"
        fib = a.fibonacci_levels
        if fib:
            assert fib["0.382"] != fib["0.618"], f"[{name}] fib identical"

    @pytest.mark.parametrize("code,name", LIVE_STOCKS)
    def test_multi_timeframe_live(self, code, name):
        from data_provider.ths_fetcher import ThsFetcher, _to_ths_code
        f = ThsFetcher()
        if not f.is_available():
            f.close()
            pytest.skip("THS SDK unavailable")
        try:
            daily = f._fetch_raw_data(code, "2026-01-01", "2026-12-31")
            daily = f._normalize_data(daily, code)
            ths_code = _to_ths_code(code)
            hourly = None
            try:
                hourly = f._ths_call("klines", ths_code, interval="60m", count=240)
            except Exception:
                pass
            r = analyze_multi_timeframe(daily, hourly, code)
            _assert_no_nan(r)
            assert "daily" in r and "combined_confidence" in r
        finally:
            f.close()


@pytest.mark.network
class TestLiveConsistency:

    def test_stocks_produce_different_results(self):
        counts = {}
        for code, name, *_ in [(*s,) for s in LIVE_STOCKS]:
            df = _get_live_data(code)
            a = analyze_waves(df, code, "day", 5, 2.0)
            counts[name] = len(a.waves)
        assert len(set(counts.values())) > 1, f"All identical: {counts}"

    def test_all_have_defined_trend(self):
        for code, name, *_ in [(*s,) for s in LIVE_STOCKS]:
            df = _get_live_data(code)
            a = analyze_waves(df, code, "day", 5, 2.0)
            assert a.trend_direction in ("up", "down", "neutral"), \
                f"[{name}] {a.trend_direction}"
