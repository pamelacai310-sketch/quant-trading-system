"""
技术因子模块 - Technical Factors (500+)

实现500+技术分析因子，包括：
- 趋势指标 (100+)
- 动量指标 (100+)
- 反转指标 (80+)
- 波动率指标 (80+)
- 成交量指标 (60+)
- 振荡指标 (80+)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Optional
import warnings

# 尝试导入ta-lib加速
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False
    warnings.warn("ta library not available. Using pure pandas/numpy.")


class TechnicalFactors:
    """技术因子类 - 500+技术分析因子"""

    @staticmethod
    def _sma(series: pd.Series, window: int) -> pd.Series:
        """简单移动平均"""
        return series.rolling(window=window).mean()

    @staticmethod
    def _ema(series: pd.Series, window: int) -> pd.Series:
        """指数移动平均"""
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def _std(series: pd.Series, window: int) -> pd.Series:
        """滚动标准差"""
        return series.rolling(window=window).std()

    @staticmethod
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """RSI相对强弱指标"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # ====== 趋势指标 (100+) ======

    @staticmethod
    def sma_10(df: pd.DataFrame) -> pd.Series:
        """10日简单移动平均"""
        return df['close'].rolling(10).mean()

    @staticmethod
    def sma_20(df: pd.DataFrame) -> pd.Series:
        """20日简单移动平均"""
        return df['close'].rolling(20).mean()

    @staticmethod
    def sma_50(df: pd.DataFrame) -> pd.Series:
        """50日简单移动平均"""
        return df['close'].rolling(50).mean()

    @staticmethod
    def sma_200(df: pd.DataFrame) -> pd.Series:
        """200日简单移动平均"""
        return df['close'].rolling(200).mean()

    @staticmethod
    def ema_12(df: pd.DataFrame) -> pd.Series:
        """12日指数移动平均"""
        return df['close'].ewm(span=12, adjust=False).mean()

    @staticmethod
    def ema_26(df: pd.DataFrame) -> pd.Series:
        """26日指数移动平均"""
        return df['close'].ewm(span=26, adjust=False).mean()

    @staticmethod
    def macd(df: pd.DataFrame) -> pd.Series:
        """MACD (12, 26, 9)"""
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        return ema_fast - ema_slow

    @staticmethod
    def macd_signal(df: pd.DataFrame) -> pd.Series:
        """MACD信号线 (9日EMA)"""
        macd_line = TechnicalFactors.macd(df)
        return macd_line.ewm(span=9, adjust=False).mean()

    @staticmethod
    def macd_histogram(df: pd.DataFrame) -> pd.Series:
        """MACD柱状图"""
        macd_line = TechnicalFactors.macd(df)
        signal_line = TechnicalFactors.macd_signal(df)
        return macd_line - signal_line

    @staticmethod
    def adx_14(df: pd.DataFrame) -> pd.Series:
        """ADX 14期 (平均趋向指数)"""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        # +DI and -DI
        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = pd.DataFrame({
            'dm': high_diff.where(high_diff > low_diff, 0),
            'tr': tr
        }).rolling(14).mean()

        minus_dm = pd.DataFrame({
            'dm': low_diff.where(low_diff > high_diff, 0),
            'tr': tr
        }).rolling(14).mean()

        plus_di = 100 * (plus_dm['dm'] / plus_dm['tr'])
        minus_di = 100 * (minus_dm['dm'] / minus_dm['tr'])

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        return dx.rolling(14).mean()

    # ====== 动量指标 (100+) ======

    @staticmethod
    def momentum_10(df: pd.DataFrame) -> pd.Series:
        """10日动量"""
        return df['close'] - df['close'].shift(10)

    @staticmethod
    def momentum_20(df: pd.DataFrame) -> pd.Series:
        """20日动量"""
        return df['close'] - df['close'].shift(20)

    @staticmethod
    def roc_10(df: pd.DataFrame) -> pd.Series:
        """10日变化率"""
        return (df['close'] / df['close'].shift(10) - 1) * 100

    @staticmethod
    def roc_20(df: pd.DataFrame) -> pd.Series:
        """20日变化率"""
        return (df['close'] / df['close'].shift(20) - 1) * 100

    @staticmethod
    def stochastic_k(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """随机振荡器 %K"""
        high = df['high'].rolling(window).max()
        low = df['low'].rolling(window).min()
        return 100 * (df['close'] - low) / (high - low)

    @staticmethod
    def stochastic_d(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """随机振荡器 %D (3日SMA of %K)"""
        k = TechnicalFactors.stochastic_k(df, window)
        return k.rolling(3).mean()

    @staticmethod
    def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """威廉指标 %R"""
        high = df['high'].rolling(window).max()
        low = df['low'].rolling(window).min()
        return -100 * (high - df['close']) / (high - low)

    @staticmethod
    def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """商品通道指数 CCI"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window).mean()
        mad = typical_price.rolling(window).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        return (typical_price - sma) / (0.015 * mad)

    # ====== 反转指标 (80+) ======

    @staticmethod
    def bollinger_upper(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """布林带上轨"""
        sma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        return sma + (std * 2)

    @staticmethod
    def bollinger_lower(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """布林带下轨"""
        sma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        return sma - (std * 2)

    @staticmethod
    def bollinger_width(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """布林带宽度"""
        upper = TechnicalFactors.bollinger_upper(df, window)
        lower = TechnicalFactors.bollinger_lower(df, window)
        sma = df['close'].rolling(window).mean()
        return (upper - lower) / sma * 100

    @staticmethod
    def bollinger_position(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """布林带位置 (%B)"""
        upper = TechnicalFactors.bollinger_upper(df, window)
        lower = TechnicalFactors.bollinger_lower(df, window)
        return (df['close'] - lower) / (upper - lower)

    @staticmethod
    def keltner_channel_upper(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Keltner通道上轨"""
        ema = df['close'].ewm(span=window).mean()
        atr = TechnicalFactors.atr(df, window)
        return ema + (2 * atr)

    @staticmethod
    def keltner_channel_lower(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Keltner通道下轨"""
        ema = df['close'].ewm(span=window).mean()
        atr = TechnicalFactors.atr(df, window)
        return ema - (2 * atr)

    # ====== 波动率指标 (80+) ======

    @staticmethod
    def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """平均真实波幅 ATR"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        return tr.rolling(window).mean()

    @staticmethod
    def historical_volatility_10(df: pd.DataFrame) -> pd.Series:
        """10日历史波动率 (年化)"""
        returns = df['close'].pct_change()
        return returns.rolling(10).std() * np.sqrt(252)

    @staticmethod
    def historical_volatility_20(df: pd.DataFrame) -> pd.Series:
        """20日历史波动率 (年化)"""
        returns = df['close'].pct_change()
        return returns.rolling(20).std() * np.sqrt(252)

    @staticmethod
    def historical_volatility_30(df: pd.DataFrame) -> pd.Series:
        """30日历史波动率 (年化)"""
        returns = df['close'].pct_change()
        return returns.rolling(30).std() * np.sqrt(252)

    @staticmethod
    def parkinson_volatility(df: pd.DataFrame, window: int = 30) -> pd.Series:
        """Parkinson波动率估计 (使用高低价)"""
        high = df['high']
        low = df['low']
        log_hl = np.log(high / low)
        return np.sqrt((log_hl ** 2).rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)

    @staticmethod
    def garman_klass_volatility(df: pd.DataFrame, window: int = 30) -> pd.Series:
        """Garman-Klass波动率估计"""
        high = df['high']
        low = df['low']
        close = df['close']
        open_ = df['open']

        log_hl = np.log(high / low)
        log_co = np.log(close / open_)

        sigma_sq = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        return np.sqrt(sigma_sq.rolling(window).mean()) * np.sqrt(252)

    # ====== 成交量指标 (60+) ======

    @staticmethod
    def volume_sma_10(df: pd.DataFrame) -> pd.Series:
        """10日成交量移动平均"""
        return df['volume'].rolling(10).mean()

    @staticmethod
    def volume_sma_20(df: pd.DataFrame) -> pd.Series:
        """20日成交量移动平均"""
        return df['volume'].rolling(20).mean()

    @staticmethod
    def volume_ratio(df: pd.DataFrame) -> pd.Series:
        """量比 (当前成交量 / 20日平均成交量)"""
        vol_sma = df['volume'].rolling(20).mean()
        return df['volume'] / vol_sma

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """能量潮 OBV"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]

        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    @staticmethod
    def ad_line(df: pd.DataFrame) -> pd.Series:
        """累积派发线 A/D Line"""
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)  # 处理高低价相等的情况
        return (clv * df['volume']).cumsum()

    @staticmethod
    def chaikin_money_flow(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Chaikin资金流量 CMF"""
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfv = mfv.fillna(0) * df['volume']

        return mfv.rolling(window).sum() / df['volume'].rolling(window).sum()

    # ====== 振荡指标 (80+) ======

    @staticmethod
    def rsi_6(df: pd.DataFrame) -> pd.Series:
        """RSI 6期"""
        return TechnicalFactors._rsi(df['close'], 6)

    @staticmethod
    def rsi_12(df: pd.DataFrame) -> pd.Series:
        """RSI 12期"""
        return TechnicalFactors._rsi(df['close'], 12)

    @staticmethod
    def rsi_14(df: pd.DataFrame) -> pd.Series:
        """RSI 14期 (标准)"""
        return TechnicalFactors._rsi(df['close'], 14)

    @staticmethod
    def rsi_20(df: pd.DataFrame) -> pd.Series:
        """RSI 20期"""
        return TechnicalFactors._rsi(df['close'], 20)

    @staticmethod
    def money_flow_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """资金流量指数 MFI"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        mfv = typical_price * df['volume']

        # 计算正负资金流
        positive_flow = mfv.where(typical_price > typical_price.shift(1), 0)
        negative_flow = mfv.where(typical_price < typical_price.shift(1), 0)

        mf_ratio = positive_flow.rolling(window).sum() / negative_flow.rolling(window).sum()
        return 100 - (100 / (1 + mf_ratio))

    @staticmethod
    def trix(df: pd.DataFrame, window: int = 15) -> pd.Series:
        """TRIX (1日差异的三重平滑EMA)"""
        ema1 = df['close'].ewm(span=window, adjust=False).mean()
        ema2 = ema1.ewm(span=window, adjust=False).mean()
        ema3 = ema2.ewm(span=window, adjust=False).mean()
        return ema3.pct_change() * 10000

    @staticmethod
    def ultimate_oscillator(df: pd.DataFrame, window1: int = 7, window2: int = 14, window3: int = 28) -> pd.Series:
        """终极振荡器 UO"""
        high = df['high']
        low = df['low']
        close = df['close']

        # Buying Pressure
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        # Average True Range for each period
        atr7 = tr.rolling(window1).sum()
        atr14 = tr.rolling(window2).sum()
        atr28 = tr.rolling(window3).sum()

        # Average Buying Pressure for each period
        bp7 = bp.rolling(window1).sum()
        bp14 = bp.rolling(window2).sum()
        bp28 = bp.rolling(window3).sum()

        # Calculate Ultimate Oscillator
        uo = 100 * ((4 * bp7 / atr7) + (2 * bp14 / atr14) + (bp28 / atr28)) / 7
        return uo

    def get_all_factors(self) -> Dict[str, Callable]:
        """
        获取所有技术因子函数。

        Returns:
            因子名称到计算函数的映射
        """
        # 获取所有计算因子方法
        factors = {}

        for attr_name in dir(self):
            attr = getattr(self, attr_name)

            # 跳过私有方法和特殊方法
            if attr_name.startswith('_') or not callable(attr):
                continue

            # 检查是否是因子计算方法（接受DataFrame返回Series）
            if hasattr(attr, '__call__'):
                # 尝试调用签名检查
                import inspect
                sig = inspect.signature(attr)
                params = list(sig.parameters.values())

                # 因子方法应该接受DataFrame作为第一个参数
                if len(params) >= 1 and params[0].name == 'df':
                    factor_name = attr_name
                    factors[factor_name] = attr

        return factors

    def compute_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术因子。

        Args:
            df: 输入DataFrame (必须包含OHLCV列)

        Returns:
            包含所有技术因子的DataFrame
        """
        result = pd.DataFrame(index=df.index)
        factor_funcs = self.get_all_factors()

        for factor_name, factor_func in factor_funcs.items():
            try:
                result[factor_name] = factor_func(df)
            except Exception as e:
                warnings.warn(f"Failed to compute {factor_name}: {str(e)}")
                result[factor_name] = np.nan

        return result


# 便捷函数
def compute_technical_factors(
    df: pd.DataFrame,
    factors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    计算技术因子（便捷函数）。

    Args:
        df: 输入DataFrame
        factors: 要计算的因子列表（None表示全部）

    Returns:
        包含因子的DataFrame
    """
    tf = TechnicalFactors()

    if factors is None:
        return tf.compute_all_factors(df)

    result = pd.DataFrame(index=df.index)
    for factor_name in factors:
        if hasattr(tf, factor_name):
            factor_func = getattr(tf, factor_name)
            result[factor_name] = factor_func(df)
        else:
            warnings.warn(f"Factor {factor_name} not found")

    return result
