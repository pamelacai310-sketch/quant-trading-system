"""
特征工程层使用示例

演示如何使用FeatureEngineeringLayer生成500+特征的宽特征矩阵
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from quant_trade_system.core.feature_engineering_layer import (
    FeatureEngineeringLayer,
    FeatureGranularity,
    FeatureDomain,
    create_feature_engineering_layer,
)
from quant_trade_system.core.causal import CausalFactorLibrary, create_causal_factor_library


def generate_sample_market_data(
    symbols: list,
    start_date: str,
    end_date: str,
    granularity: str = "1D",
) -> dict:
    """
    生成示例市场数据

    参数:
        symbols: 标的物列表
        start_date: 开始日期
        end_date: 结束日期
        granularity: 数据粒度

    返回:
        市场数据字典 {symbol: dataframe}
    """
    market_data = {}
    dates = pd.date_range(start=start_date, end=end_date, freq=granularity)

    for symbol in symbols:
        # 生成随机游走价格数据
        np.random.seed(hash(symbol) % 10000)  # 确保每个symbol的数据可重复

        # 基础价格
        base_price = 100.0 if "stock" in symbol else 5000.0

        # 价格序列（随机游走）
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * (1 + returns).cumprod()

        # 创建DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': prices * (1 + np.random.uniform(-0.02, 0, len(dates))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
        })

        # 添加一些基本面数据（低频）
        if 'stock' in symbol:
            df['eps'] = np.random.uniform(2, 10, len(dates))
            df['book_value_per_share'] = np.random.uniform(20, 80, len(dates))
            df['roic'] = np.random.uniform(0.05, 0.25, len(dates))
            df['wacc'] = 0.08

        # 添加宏观经济数据（所有symbol相同）
        df['interest_rate'] = 0.03 + np.random.normal(0, 0.001, len(dates)).cumsum() * 0.1
        df['cpi'] = 100 + np.random.normal(0, 0.1, len(dates)).cumsum()
        df['vix_index'] = 15 + np.random.uniform(0, 30, len(dates))

        market_data[symbol] = df.set_index('timestamp')

    return market_data


def generate_sample_level2_data(
    symbol: str,
    start_date: str,
    duration_hours: int = 24,
) -> pd.DataFrame:
    """
    生成示例Level 2订单簿数据

    参数:
        symbol: 标的物
        start_date: 开始时间
        duration_hours: 持续时间（小时）

    返回:
        订单簿数据
    """
    start_time = pd.to_datetime(start_date)
    timestamps = pd.date_range(
        start=start_time,
        periods=duration_hours * 60,  # 每分钟一个数据点
        freq='1min',
    )

    data = []
    base_price = 100.0

    for i, ts in enumerate(timestamps):
        # 模拟价格变化
        price_change = np.random.normal(0, 0.001)
        base_price = base_price * (1 + price_change)

        # 生成5档买卖价
        bid_prices = [base_price - j * 0.01 for j in range(1, 6)]
        ask_prices = [base_price + j * 0.01 for j in range(1, 6)]

        # 生成5档买卖量
        bid_volumes = [np.random.randint(1000, 10000) for _ in range(5)]
        ask_volumes = [np.random.randint(1000, 10000) for _ in range(5)]

        row = {'timestamp': ts}
        for j in range(1, 6):
            row[f'bid_price_{j}'] = bid_prices[j-1]
            row[f'bid_volume_{j}'] = bid_volumes[j-1]
            row[f'ask_price_{j}'] = ask_prices[j-1]
            row[f'ask_volume_{j}'] = ask_volumes[j-1]

        data.append(row)

    return pd.DataFrame(data)


def generate_sample_trades_data(
    symbol: str,
    start_date: str,
    duration_hours: int = 24,
) -> pd.DataFrame:
    """
    生成示例逐笔成交数据

    参数:
        symbol: 标的物
        start_date: 开始时间
        duration_hours: 持续时间（小时）

    返回:
        逐笔成交数据
    """
    start_time = pd.to_datetime(start_date)
    # 每分钟10-50笔成交
    num_trades = duration_hours * 60 * 30

    timestamps = pd.date_range(
        start=start_time,
        periods=num_trades,
        freq='2s',  # 每2秒一笔
    )

    data = []
    base_price = 100.0

    for ts in timestamps:
        # 模拟价格变化
        price_change = np.random.normal(0, 0.0001)
        base_price = base_price * (1 + price_change)

        # 随机买卖方向
        direction = 'buy' if np.random.random() > 0.5 else 'sell'

        # 成交量
        volume = np.random.randint(100, 5000)

        data.append({
            'timestamp': ts,
            'price': base_price,
            'volume': volume,
            'direction': direction,
        })

    return pd.DataFrame(data)


def generate_sample_news_data(
    start_date: str,
    duration_days: int = 30,
) -> pd.DataFrame:
    """
    生成示例新闻情感数据

    参数:
        start_date: 开始日期
        duration_days: 持续天数

    返回:
        新闻情感数据
    """
    start_time = pd.to_datetime(start_date)

    # 每天随机5-20条新闻
    news_items = []
    for day in range(duration_days):
        num_news = np.random.randint(5, 20)

        for _ in range(num_news):
            # 随机时间（交易时间内）
            hour = np.random.randint(9, 16)
            minute = np.random.randint(0, 60)

            ts = start_time + timedelta(days=day, hours=hour, minutes=minute)

            # 情感得分 (-1到1)
            sentiment_score = np.random.uniform(-1, 1)

            # 相关性得分 (0到1)
            relevance_score = np.random.uniform(0.3, 1.0)

            news_items.append({
                'timestamp': ts,
                'sentiment_score': sentiment_score,
                'relevance_score': relevance_score,
            })

    return pd.DataFrame(news_items)


def generate_sample_satellite_data(
    location: str,
    start_date: str,
    duration_days: int = 30,
) -> pd.DataFrame:
    """
    生成示例卫星数据

    参数:
        location: 地点
        start_date: 开始日期
        duration_days: 持续天数

    返回:
        卫星数据
    """
    start_time = pd.to_datetime(start_date)

    # 每天随机1-3次卫星过境
    data_items = []
    for day in range(duration_days):
        num_observations = np.random.randint(1, 3)

        for _ in range(num_observations):
            # 随机时间
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)

            ts = start_time + timedelta(days=day, hours=hour, minutes=minute)

            # 特征值（例如：植被指数、夜间灯光、建筑面积等）
            feature_values = {
                'ndvi': np.random.uniform(0.3, 0.8),  # 植被指数
                'night_lights': np.random.randint(50, 200),  # 夜间灯光强度
                'construction_area': np.random.randint(100, 1000),  # 建筑面积
                'traffic_density': np.random.uniform(0.1, 0.9),  # 交通密度
            }

            data_items.append({
                'timestamp': ts,
                'location': location,
                'feature_values': feature_values,
            })

    return pd.DataFrame(data_items)


def example_1_basic_feature_generation():
    """示例1: 基础特征生成"""
    print("\n" + "="*80)
    print("示例1: 基础特征生成")
    print("="*80)

    # 创建特征工程层
    layer = create_feature_engineering_layer(
        target_features=500,
        min_interpretability=0.6,
        min_independent_power=0.5,
    )

    print(f"✅ 特征工程层创建成功")
    print(f"  注册特征数量: {len(layer.feature_registry)}")

    # 生成示例市场数据
    symbols = ['AAPL_stock', 'MSFT_stock', 'GOOGL_stock']
    market_data = generate_sample_market_data(
        symbols=symbols,
        start_date='2024-01-01',
        end_date='2024-03-31',
    )

    print(f"\n✅ 市场数据生成成功")
    print(f"  标的物: {symbols}")
    print(f"  时间范围: 2024-01-01 至 2024-03-31")

    # 生成特征矩阵
    feature_matrix = layer.generate_feature_matrix(
        market_data=market_data,
        granularity=FeatureGranularity.DAILY,
        feature_limit=100,  # 限制为100个特征用于演示
    )

    print(f"\n✅ 特征矩阵生成成功")
    print(f"  特征数量: {len(feature_matrix.data.columns)}")
    print(f"  时间范围: {feature_matrix.sampling_start} 至 {feature_matrix.sampling_end}")
    print(f"  数据粒度: {feature_matrix.granularity.value}")
    print(f"  矩阵形状: {feature_matrix.data.shape}")

    # 显示前10个特征
    print(f"\n前10个特征:")
    for i, col in enumerate(feature_matrix.data.columns[:10]):
        metadata = feature_matrix.feature_metadata[col]
        print(f"  {i+1}. {col}")
        print(f"     名称: {metadata['name']}")
        print(f"     因素ID: {metadata['causal_factor_id']}")
        print(f"     金融含义: {metadata['financial_meaning']}")
        print(f"     可解释性: {metadata['interpretability']:.2f}")
        print(f"     独立解释力: {metadata['independent_power']:.2f}")


def example_2_level2_orderbook_processing():
    """示例2: Level 2订单簿数据处理"""
    print("\n" + "="*80)
    print("示例2: Level 2订单簿数据处理")
    print("="*80)

    # 创建特征工程层
    layer = create_feature_engineering_layer()

    # 生成示例Level 2订单簿数据
    orderbook_data = generate_sample_level2_data(
        symbol='AAPL_stock',
        start_date='2024-01-01 09:30:00',
        duration_hours=24,
    )

    print(f"✅ 订单簿数据生成成功")
    print(f"  数据点数量: {len(orderbook_data)}")
    print(f"  时间范围: {orderbook_data['timestamp'].min()} 至 {orderbook_data['timestamp'].max()}")

    # 处理订单簿数据
    orderbook_features = layer.process_level2_orderbook(
        orderbook_data=orderbook_data,
        granularity=FeatureGranularity.MINUTE_5,
    )

    print(f"\n✅ 订单簿特征提取成功")
    print(f"  特征数量: {len(orderbook_features)}")
    print(f"\n提取的特征:")
    for i, (feature_name, series) in enumerate(orderbook_features.items()):
        print(f"  {i+1}. {feature_name}")
        print(f"     数据点数量: {len(series)}")
        print(f"     均值: {series.mean():.4f}")
        print(f"     标准差: {series.std():.4f}")


def example_3_level2_trades_processing():
    """示例3: Level 2逐笔成交数据处理"""
    print("\n" + "="*80)
    print("示例3: Level 2逐笔成交数据处理")
    print("="*80)

    # 创建特征工程层
    layer = create_feature_engineering_layer()

    # 生成示例逐笔成交数据
    trades_data = generate_sample_trades_data(
        symbol='AAPL_stock',
        start_date='2024-01-01 09:30:00',
        duration_hours=24,
    )

    print(f"✅ 逐笔成交数据生成成功")
    print(f"  成交笔数: {len(trades_data)}")
    print(f"  时间范围: {trades_data['timestamp'].min()} 至 {trades_data['timestamp'].max()}")

    # 处理逐笔成交数据
    trades_features = layer.process_level2_trades(
        trades_data=trades_data,
        granularity=FeatureGranularity.MINUTE_5,
    )

    print(f"\n✅ 逐笔成交特征提取成功")
    print(f"  特征数量: {len(trades_features)}")
    print(f"\n提取的特征:")
    for i, (feature_name, series) in enumerate(trades_features.items()):
        if series is not None and len(series) > 0:
            print(f"  {i+1}. {feature_name}")
            print(f"     数据点数量: {len(series)}")
            print(f"     均值: {series.mean():.4f}")


def example_4_news_sentiment_alignment():
    """示例4: 新闻情感数据对齐"""
    print("\n" + "="*80)
    print("示例4: 新闻情感数据对齐")
    print("="*80)

    # 创建特征工程层
    layer = create_feature_engineering_layer()

    # 生成示例新闻情感数据
    news_data = generate_sample_news_data(
        start_date='2024-01-01',
        duration_days=30,
    )

    print(f"✅ 新闻情感数据生成成功")
    print(f"  新闻条数: {len(news_data)}")
    print(f"  时间范围: {news_data['timestamp'].min()} 至 {news_data['timestamp'].max()}")

    # 生成市场时间戳（交易日）
    market_timestamps = pd.date_range(
        start='2024-01-01 09:30:00',
        end='2024-01-30 16:00:00',
        freq='B',  # 工作日
    ).tolist()

    print(f"\n市场时间戳数量: {len(market_timestamps)}")

    # 对齐新闻情感数据
    sentiment_features = layer.align_news_sentiment(
        news_data=news_data,
        market_timestamps=market_timestamps,
        window_minutes=60,
    )

    print(f"\n✅ 新闻情感数据对齐成功")
    print(f"  特征数量: {len(sentiment_features.columns)}")
    print(f"\n对齐的特征:")
    for i, col in enumerate(sentiment_features.columns):
        print(f"  {i+1}. {col}")
        print(f"     非NaN数据点: {sentiment_features[col].notna().sum()}")


def example_5_satellite_data_alignment():
    """示例5: 卫星数据对齐"""
    print("\n" + "="*80)
    print("示例5: 卫星数据对齐")
    print("="*80)

    # 创建特征工程层
    layer = create_feature_engineering_layer()

    # 生成示例卫星数据
    satellite_data = generate_sample_satellite_data(
        location='Shanghai',
        start_date='2024-01-01',
        duration_days=30,
    )

    print(f"✅ 卫星数据生成成功")
    print(f"  观测次数: {len(satellite_data)}")
    print(f"  时间范围: {satellite_data['timestamp'].min()} 至 {satellite_data['timestamp'].max()}")

    # 生成市场时间戳
    market_timestamps = pd.date_range(
        start='2024-01-01 09:30:00',
        end='2024-01-30 16:00:00',
        freq='B',
    ).tolist()

    print(f"\n市场时间戳数量: {len(market_timestamps)}")

    # 对齐卫星数据
    satellite_features = layer.align_satellite_data(
        satellite_data=satellite_data,
        market_timestamps=market_timestamps,
        location='Shanghai',
        window_hours=24,
    )

    print(f"\n✅ 卫星数据对齐成功")
    print(f"  特征数量: {len(satellite_features.columns)}")
    print(f"\n对齐的特征:")
    for i, col in enumerate(satellite_features.columns):
        print(f"  {i+1}. {col}")


def example_6_comprehensive_feature_matrix():
    """示例6: 综合特征矩阵生成（整合所有数据源）"""
    print("\n" + "="*80)
    print("示例6: 综合特征矩阵生成")
    print("="*80)

    # 创建特征工程层
    layer = create_feature_engineering_layer(
        target_features=500,
        min_interpretability=0.6,
        min_independent_power=0.5,
    )

    # 生成各种数据
    symbols = ['AAPL_stock', 'MSFT_stock']
    market_data = generate_sample_market_data(
        symbols=symbols,
        start_date='2024-01-01',
        end_date='2024-03-31',
    )

    orderbook_data = generate_sample_level2_data(
        symbol='AAPL_stock',
        start_date='2024-01-01 09:30:00',
        duration_hours=24,
    )

    trades_data = generate_sample_trades_data(
        symbol='AAPL_stock',
        start_date='2024-01-01 09:30:00',
        duration_hours=24,
    )

    news_data = generate_sample_news_data(
        start_date='2024-01-01',
        duration_days=30,
    )

    satellite_data = generate_sample_satellite_data(
        location='Shanghai',
        start_date='2024-01-01',
        duration_days=30,
    )

    print("✅ 所有数据源生成成功")

    # 生成基础特征矩阵
    feature_matrix = layer.generate_feature_matrix(
        market_data=market_data,
        granularity=FeatureGranularity.DAILY,
        feature_limit=100,
    )

    print(f"\n✅ 基础特征矩阵生成成功")
    print(f"  基础特征数量: {len(feature_matrix.data.columns)}")

    # 处理Level 2数据
    orderbook_features = layer.process_level2_orderbook(
        orderbook_data=orderbook_data,
        granularity=FeatureGranularity.MINUTE_5,
    )

    trades_features = layer.process_level2_trades(
        trades_data=trades_data,
        granularity=FeatureGranularity.MINUTE_5,
    )

    print(f"✅ Level 2特征提取成功")
    print(f"  订单簿特征数量: {len(orderbook_features)}")
    print(f"  逐笔成交特征数量: {len(trades_features)}")

    # 对齐另类数据
    market_timestamps = list(feature_matrix.data.index)

    sentiment_features = layer.align_news_sentiment(
        news_data=news_data,
        market_timestamps=market_timestamps,
        window_minutes=60,
    )

    satellite_features = layer.align_satellite_data(
        satellite_data=satellite_data,
        market_timestamps=market_timestamps,
        location='Shanghai',
        window_hours=24,
    )

    print(f"✅ 另类数据对齐成功")
    print(f"  新闻情感特征数量: {len(sentiment_features.columns)}")
    print(f"  卫星特征数量: {len(satellite_features.columns)}")

    # 合并所有特征
    all_features = pd.concat([
        feature_matrix.data,
        sentiment_features,
        satellite_features,
    ], axis=1)

    print(f"\n✅ 综合特征矩阵生成成功")
    print(f"  总特征数量: {len(all_features.columns)}")
    print(f"  矩阵形状: {all_features.shape}")
    print(f"  时间范围: {all_features.index.min()} 至 {all_features.index.max()}")

    # 特征分类统计
    feature_categories = {}
    for col in all_features.columns:
        if col in feature_matrix.feature_metadata:
            category = feature_matrix.feature_metadata[col]['category']
            feature_categories[category] = feature_categories.get(category, 0) + 1
        elif 'news' in col:
            feature_categories['sentiment'] = feature_categories.get('sentiment', 0) + 1
        elif 'satellite' in col:
            feature_categories['satellite'] = feature_categories.get('satellite', 0) + 1

    print(f"\n特征分类统计:")
    for category, count in feature_categories.items():
        print(f"  {category}: {count}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("特征工程层使用示例")
    print("="*80)

    # 运行所有示例
    example_1_basic_feature_generation()
    example_2_level2_orderbook_processing()
    example_3_level2_trades_processing()
    example_4_news_sentiment_alignment()
    example_5_satellite_data_alignment()
    example_6_comprehensive_feature_matrix()

    print("\n" + "="*80)
    print("所有示例运行完成!")
    print("="*80)


if __name__ == "__main__":
    main()
