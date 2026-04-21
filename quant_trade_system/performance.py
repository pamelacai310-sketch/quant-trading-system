"""
性能优化模块

提供缓存、批量处理、性能监控等功能
"""

from __future__ import annotations

import functools
import gc
import logging
import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

import psutil
import pandas as pd

# 类型变量
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = None

    def start(self) -> None:
        """开始监控"""
        self.start_time = time.time()
        tracemalloc.start()

    def stop(self) -> Dict[str, Any]:
        """停止监控并返回统计信息"""
        if not self.start_time:
            return {}

        elapsed = time.time() - self.start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        stats = {
            "elapsed_time": elapsed,
            "memory_current_mb": current / 1024 / 1024,
            "memory_peak_mb": peak / 1024 / 1024,
            "timestamp": datetime.now().isoformat()
        }

        # 记录到指标中
        self.metrics["elapsed_time"].append(elapsed)
        self.metrics["memory_peak_mb"].append(peak / 1024 / 1024)

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.metrics["elapsed_time"]:
            return {}

        return {
            "avg_time": sum(self.metrics["elapsed_time"]) / len(self.metrics["elapsed_time"]),
            "max_time": max(self.metrics["elapsed_time"]),
            "avg_memory_mb": sum(self.metrics["memory_peak_mb"]) / len(self.metrics["memory_peak_mb"]),
            "max_memory_mb": max(self.metrics["memory_peak_mb"]),
            "total_calls": len(self.metrics["elapsed_time"])
        }


# 全局监控器实例
monitor = PerformanceMonitor()


def measure_time(func: F) -> F:
    """测量函数执行时间的装饰器"""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} 执行时间: {elapsed:.3f}秒")

            # 如果执行时间过长，记录警告
            if elapsed > 5.0:
                logger.warning(f"{func.__name__} 执行时间过长: {elapsed:.3f}秒")

    return wrapper


def measure_memory(func: F) -> F:
    """测量函数内存使用的装饰器"""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # 强制垃圾回收
        gc.collect()

        # 获取初始内存
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # 获取最终内存
            gc.collect()
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_diff = mem_after - mem_before

            logger.debug(f"{func.__name__} 内存变化: {mem_diff:+.2f}MB")

            # 如果内存增长过多，发出警告
            if mem_diff > 100:
                logger.warning(f"{func.__name__} 内存增长过多: {mem_diff:.2f}MB")

    return wrapper


class TimedCache:
    """带时间限制的缓存"""

    def __init__(self, ttl_seconds: int = 60):
        """
        初始化缓存

        Args:
            ttl_seconds: 缓存过期时间（秒）
        """
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                self.hits += 1
                return value
            else:
                # 缓存过期
                del self.cache[key]

        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        self.cache[key] = (value, datetime.now())

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()

    def cleanup_expired(self) -> int:
        """清理过期缓存"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= self.ttl
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl.total_seconds()
        }


class SmartCache:
    """智能缓存系统"""

    def __init__(self):
        self.caches: Dict[str, TimedCache] = {}

    def get_cache(self, category: str, ttl_seconds: int = 60) -> TimedCache:
        """获取或创建缓存"""
        if category not in self.caches:
            self.caches[category] = TimedCache(ttl_seconds)
        return self.caches[category]

    def get(self, category: str, key: str) -> Optional[Any]:
        """从指定类别获取缓存"""
        cache = self.get_cache(category)
        return cache.get(key)

    def set(self, category: str, key: str, value: Any, ttl_seconds: int = 60) -> None:
        """设置指定类别的缓存"""
        cache = self.get_cache(category, ttl_seconds)
        cache.set(key, value)

    def clear_category(self, category: str) -> None:
        """清空指定类别的缓存"""
        if category in self.caches:
            self.caches[category].clear()

    def clear_all(self) -> None:
        """清空所有缓存"""
        for cache in self.caches.values():
            cache.clear()

    def cleanup_all(self) -> None:
        """清理所有过期缓存"""
        total_cleaned = 0
        for cache in self.caches.values():
            total_cleaned += cache.cleanup_expired()
        logger.info(f"清理了 {total_cleaned} 个过期缓存条目")

    def get_stats(self) -> Dict[str, Any]:
        """获取所有缓存统计"""
        stats = {}
        for category, cache in self.caches.items():
            stats[category] = cache.get_stats()
        return stats


# 全局智能缓存实例
smart_cache = SmartCache()


def cached(ttl_seconds: int = 60, category: str = "default"):
    """缓存装饰器"""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 生成缓存键
            cache_key = f"{func.__name__}_{args}_{kwargs}"

            # 尝试从缓存获取
            cached_result = smart_cache.get(category, cache_key)
            if cached_result is not None:
                return cached_result

            # 执行函数
            result = func(*args, **kwargs)

            # 存入缓存
            smart_cache.set(category, cache_key, result, ttl_seconds)

            return result

        return wrapper
    return decorator


class BatchProcessor:
    """批量处理器"""

    def __init__(self, batch_size: int = 100):
        """
        初始化批量处理器

        Args:
            batch_size: 批量大小
        """
        self.batch_size = batch_size

    def process_in_batches(
        self,
        items: List[T],
        processor: Callable[[List[T]], Any],
        show_progress: bool = False
    ) -> List[Any]:
        """
        批量处理项目

        Args:
            items: 要处理的项目列表
            processor: 处理函数，接收一个批次的项目
            show_progress: 是否显示进度

        Returns:
            处理结果列表
        """
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            if show_progress:
                logger.info(f"处理批次 {batch_num}/{total_batches}")

            try:
                batch_result = processor(batch)
                results.append(batch_result)
            except Exception as e:
                logger.error(f"批次 {batch_num} 处理失败: {e}")
                results.append(None)

        return results

    def process_dataframe_in_batches(
        self,
        df: pd.DataFrame,
        processor: Callable[[pd.DataFrame], pd.DataFrame],
        show_progress: bool = False
    ) -> pd.DataFrame:
        """
        批量处理DataFrame

        Args:
            df: 要处理的DataFrame
            processor: 处理函数，接收一个批次的DataFrame
            show_progress: 是否显示进度

        Returns:
            处理后的DataFrame
        """
        results = []
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(df), self.batch_size):
            batch = df.iloc[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            if show_progress:
                logger.info(f"处理批次 {batch_num}/{total_batches} ({len(batch)} 行)")

            try:
                batch_result = processor(batch)
                results.append(batch_result)
            except Exception as e:
                logger.error(f"批次 {batch_num} 处理失败: {e}")
                # 保持原数据
                results.append(batch)

        return pd.concat(results, ignore_index=True)


class QueryOptimizer:
    """查询优化器"""

    def __init__(self):
        self.query_cache = TimedCache(ttl_seconds=300)  # 5分钟缓存
        self.query_stats = defaultdict(int)

    @measure_time
    def execute_query(
        self,
        query_func: Callable[[], T],
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> T:
        """
        执行查询（带缓存）

        Args:
            query_func: 查询函数
            cache_key: 缓存键
            use_cache: 是否使用缓存

        Returns:
            查询结果
        """
        # 生成缓存键
        if not cache_key:
            cache_key = f"query_{query_func.__name__}"

        # 检查缓存
        if use_cache:
            cached_result = self.query_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"查询缓存命中: {cache_key}")
                return cached_result

        # 执行查询
        result = query_func()

        # 存入缓存
        if use_cache:
            self.query_cache.set(cache_key, result)

        # 记录统计
        self.query_stats[query_func.__name__] += 1

        return result

    def clear_cache(self) -> None:
        """清空查询缓存"""
        self.query_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取查询统计"""
        return {
            "cache_stats": self.query_cache.get_stats(),
            "query_counts": dict(self.query_stats)
        }


class DataFrameOptimizer:
    """DataFrame优化器"""

    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame的数据类型以减少内存使用"""
        optimized_df = df.copy()

        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype

            if col_type == 'object':
                # 尝试转换为category类型
                if optimized_df[col].nunique() / len(optimized_df[col]) < 0.5:
                    optimized_df[col] = optimized_df[col].astype('category')

            elif col_type == 'float64':
                # 转换为float32
                optimized_df[col] = optimized_df[col].astype('float32')

            elif col_type == 'int64':
                # 转换为int32或更小的类型
                if optimized_df[col].min() >= 0:
                    if optimized_df[col].max() < 255:
                        optimized_df[col] = optimized_df[col].astype('uint8')
                    elif optimized_df[col].max() < 65535:
                        optimized_df[col] = optimized_df[col].astype('uint16')
                    else:
                        optimized_df[col] = optimized_df[col].astype('uint32')
                else:
                    if optimized_df[col].min() > -128 and optimized_df[col].max() < 127:
                        optimized_df[col] = optimized_df[col].astype('int8')
                    elif optimized_df[col].min() > -32768 and optimized_df[col].max() < 32767:
                        optimized_df[col] = optimized_df[col].astype('int16')
                    else:
                        optimized_df[col] = optimized_df[col].astype('int32')

        return optimized_df

    @staticmethod
    def reduce_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """减少DataFrame内存使用"""
        start_mem = df.memory_usage(deep=True).sum() / 1024 / 1024

        optimized_df = DataFrameOptimizer.optimize_dtypes(df)

        end_mem = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = 100 * (start_mem - end_mem) / start_mem

        if verbose:
            logger.info(f"内存优化: {start_mem:.2f}MB -> {end_mem:.2f}MB "
                       f"({reduction:.1f}% 减少)")

        return optimized_df

    @staticmethod
    def chunked_processing(
        df: pd.DataFrame,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        chunk_size: int = 10000
    ) -> pd.DataFrame:
        """分块处理大型DataFrame"""
        results = []

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            result = func(chunk)
            results.append(result)

        return pd.concat(results, ignore_index=True)


@contextmanager
def performance_context(operation_name: str):
    """性能监控上下文管理器"""
    logger.info(f"开始 {operation_name}")
    start_time = time.time()
    start_mem = psutil.virtual_memory().used / 1024 / 1024

    try:
        yield
    finally:
        elapsed = time.time() - start_time
        end_mem = psutil.virtual_memory().used / 1024 / 1024
        mem_diff = end_mem - start_mem

        logger.info(f"完成 {operation_name} - "
                   f"耗时: {elapsed:.3f}秒, "
                   f"内存变化: {mem_diff:+.2f}MB")


def optimize_calculation(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    """计算优化装饰器"""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> pd.DataFrame:
        # 执行原函数
        result = func(*args, **kwargs)

        # 如果是DataFrame，进行优化
        if isinstance(result, pd.DataFrame):
            result = DataFrameOptimizer.reduce_memory(result, verbose=False)

        return result

    return wrapper


# 便捷函数
def get_system_stats() -> Dict[str, Any]:
    """获取系统统计信息"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        "cpu_percent": cpu_percent,
        "memory": {
            "total_gb": memory.total / 1024 / 1024 / 1024,
            "available_gb": memory.available / 1024 / 1024 / 1024,
            "used_gb": memory.used / 1024 / 1024 / 1024,
            "percent": memory.percent
        },
        "disk": {
            "total_gb": disk.total / 1024 / 1024 / 1024,
            "used_gb": disk.used / 1024 / 1024 / 1024,
            "free_gb": disk.free / 1024 / 1024 / 1024,
            "percent": disk.percent
        },
        "timestamp": datetime.now().isoformat()
    }


def cleanup_memory() -> None:
    """清理内存"""
    gc.collect()
    logger.debug("执行内存清理")


# 示例使用
if __name__ == "__main__":
    # 测试缓存
    @cached(ttl_seconds=60, category="test")
    def expensive_function(n: int) -> int:
        """模拟耗时函数"""
        time.sleep(1)
        return n * 2

    # 第一次调用会执行函数
    start = time.time()
    result1 = expensive_function(5)
    print(f"第一次调用耗时: {time.time() - start:.3f}秒")

    # 第二次调用会从缓存获取
    start = time.time()
    result2 = expensive_function(5)
    print(f"第二次调用耗时: {time.time() - start:.3f}秒")

    # 测试批量处理
    batch_processor = BatchProcessor(batch_size=10)

    data = list(range(100))
    results = batch_processor.process_in_batches(
        data,
        lambda batch: [x * 2 for x in batch],
        show_progress=True
    )

    print(f"处理了 {len(results)} 个批次")

    # 测试性能监控
    with performance_context("测试操作"):
        time.sleep(0.5)

    # 获取系统统计
    stats = get_system_stats()
    print(f"系统统计: {stats}")
