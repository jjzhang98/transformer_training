"""Data loading and preprocessing for local parquet factor data.

Directory layout expected:
    {data_root}/
        {YYYYMMDD}/
            {code}_{abtype}.parquet   # e.g. 12502_3018.parquet

Each parquet file contains columns like:
    date, code, abtype, time_sec, timestamp, bid1, ask1,
    signal_rm_5, signal_std_15, ...,          ← factors (X)
    signal_y_tick_5, signal_y_tick_15, ...    ← labels  (y)

Rolling normalization:
    - Day N is z-score normalized using Day N-1's mean/std.
    - The very first day has no previous stats → all factors set to NaN → dropped.
    - No look-ahead bias; safe for live deployment.
"""
from __future__ import annotations

import json
import os
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union
import gc

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as fs
import pyarrow.parquet as pq
import torch
from scipy.stats import norm
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Global Constants
# ---------------------------------------------------------------------------

# Metadata columns always kept alongside features and labels
_META = ["date", "code", "abtype", "time_sec"]

# HDFS configuration
HDFS_URI = "viewfs://ftxz-hadoop-view/"
HDFS_USER = "zli"
HDFS_PATH = '/user/zli/lyt/data/StockIndexFutures/MSelect/FactorData'
MAIN_CONTRACT_FILE = 'main_contract.json'

_hdfs_connection = None
_main_contract_data = None
_thread_local = threading.local()
_hdfs_dir_lock = threading.Lock()
_hdfs_known_dirs = set()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def rank_gaussian(x):
    """Rank Gaussian normalization. Supports numpy array and torch tensor."""

    is_tensor = torch.is_tensor(x)
    if not is_tensor:
        x = torch.from_numpy(np.asarray(x).astype(np.float64))
    else:
        x = x.float()

    original_shape = x.shape
    x_flat = x.flatten()
    mask = torch.isfinite(x_flat)
    n_finite = mask.sum().item()

    if n_finite < 2:
        result = torch.full(original_shape, float('nan'), dtype=torch.float64)
        return result.numpy() if not is_tensor else result

    finite_vals = x_flat[mask]
    ranks = torch.argsort(torch.argsort(finite_vals)).float() + 1
    p = (ranks - 0.5) / n_finite
    p = torch.clamp(p, min=1e-6, max=1 - 1e-6)

    result_flat = torch.full(x_flat.shape, float('nan'), dtype=torch.float64)
    result_flat[mask] = torch.from_numpy(norm.ppf(p.detach().cpu().numpy()))

    result = result_flat.reshape(original_shape)

    if is_tensor:
        return result.float()
    return result.numpy()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            print("Using CUDA device")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("Using CPU device")
        return torch.device("cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# HDFS Connection Management
# ---------------------------------------------------------------------------

def get_hdfs_connection():
    """获取HDFS连接，如果连接不存在则创建，复用现有连接"""
    global _hdfs_connection
    if _hdfs_connection is None:
        print("[HDFS] Initializing HDFS connection...")
        _hdfs_connection = fs.HadoopFileSystem(HDFS_URI, user=HDFS_USER)
        print("[HDFS] HDFS connection established")
    return _hdfs_connection


def get_thread_local_hdfs_connection():
    """获取线程局部的HDFS连接，每个线程独立连接"""
    if not hasattr(_thread_local, 'hdfs_connection'):
        print(f"[HDFS] Thread {threading.current_thread().name}: Initializing HDFS connection...")
        _thread_local.hdfs_connection = fs.HadoopFileSystem(HDFS_URI, user=HDFS_USER)
        print(f"[HDFS] Thread {threading.current_thread().name}: HDFS connection established")
    return _thread_local.hdfs_connection


def close_hdfs_connection():
    """关闭HDFS连接"""
    global _hdfs_connection
    if _hdfs_connection is not None:
        _hdfs_connection = None
        print("[HDFS] HDFS connection reference released")


def ensure_hdfs_dir(hdfs: fs.HadoopFileSystem, dir_path: str) -> bool:
    """线程安全地创建HDFS目录

    Args:
        hdfs: HDFS文件系统连接
        dir_path: 要创建的目录路径

    Returns:
        是否成功创建（已存在也返回True）
    """
    if dir_path in _hdfs_known_dirs:
        return True
    with _hdfs_dir_lock:
        try:
            file_info = hdfs.get_file_info(dir_path)
            if file_info.type == fs.FileType.Directory:
                _hdfs_known_dirs.add(dir_path)
                return True

            hdfs.create_dir(dir_path, recursive=True)
            _hdfs_known_dirs.add(dir_path)
            return True
        except Exception as e:
            print(f"[HDFS] Error creating directory {dir_path}: {e}")
            return False


# ---------------------------------------------------------------------------
# Main Contract & Directory Helpers
# ---------------------------------------------------------------------------

def load_main_contract_data():
    """加载主力合约数据"""
    global _main_contract_data
    if _main_contract_data is None:
        try:
            with open(MAIN_CONTRACT_FILE, 'r') as f:
                _main_contract_data = json.load(f)
            print(f"[HDFS] Loaded main contract data from {MAIN_CONTRACT_FILE}")
        except Exception as e:
            print(f"[HDFS] Error loading main contract data: {e}")
            _main_contract_data = {}
    return _main_contract_data


def get_main_contracts(date_str):
    """获取指定日期的主力合约列表"""
    main_contract_data = load_main_contract_data()
    return main_contract_data.get(date_str, [])


def get_date_directories(hdfs_path):
    """获取HDFS路径下的所有日期目录"""
    hdfs = get_hdfs_connection()
    try:
        file_info = hdfs.get_file_info(hdfs_path)
        if file_info.type != fs.FileType.Directory:
            return []

        selector = fs.FileSelector(hdfs_path, recursive=False)
        entries = hdfs.get_file_info(selector)

        date_dirs = []
        for entry in entries:
            if entry.type == fs.FileType.Directory:
                date_dirs.append(entry.base_name)

        return sorted(date_dirs)
    except Exception as e:
        print(f"[HDFS] Error getting date directories: {e}")
        return []


# ---------------------------------------------------------------------------
# Parquet File Filtering
# ---------------------------------------------------------------------------

def _filter_parquet_file(
        fname: str,
        codes_set: Optional[set] = None,
        abtypes_set: Optional[set] = None,
        main_contracts: Optional[List[str]] = None,
) -> bool:
    """筛选parquet文件

    Args:
        fname: 文件名（不含.parquet后缀）
        codes_set: 代码集合，如果为None则不过滤
        abtypes_set: abtype集合，如果为None则不过滤
        main_contracts: 主力合约列表，如果为None则不过滤

    Returns:
        是否通过筛选
    """
    parts = fname.split("_")
    if len(parts) != 2:
        return False

    try:
        code, abtype = int(parts[0]), int(parts[1])
    except ValueError:
        return False

    if codes_set is not None and code not in codes_set:
        return False
    if abtypes_set is not None and abtype not in abtypes_set:
        return False
    if main_contracts is not None and str(code) not in main_contracts:
        return False

    return True


def get_parquet_files(date_dir, codes=None, abtypes=None, main_contracts_only=True):
    """获取指定日期目录下的parquet文件，支持筛选

    Args:
        date_dir: 日期目录名
        codes: 代码列表，如果为None则不过滤
        abtypes: abtype列表，如果为None则不过滤
        main_contracts_only: 是否只读取主力合约，默认为True

    Returns:
        parquet文件列表
    """
    hdfs = get_hdfs_connection()
    date_path = f"{HDFS_PATH}/{date_dir}"

    try:
        selector = fs.FileSelector(date_path, recursive=False)
        entries = hdfs.get_file_info(selector)

        codes_set = set(int(c) for c in codes) if codes else None
        abtypes_set = set(int(a) for a in abtypes) if abtypes else None
        main_contracts = get_main_contracts(date_dir) if main_contracts_only else None

        parquet_files = []
        for entry in entries:
            if entry.type == fs.FileType.File and entry.base_name.endswith('.parquet'):
                fname = entry.base_name.replace(".parquet", "")
                if _filter_parquet_file(fname, codes_set, abtypes_set, main_contracts):
                    parquet_files.append(entry.base_name)

        return parquet_files
    except Exception as e:
        print(f"[HDFS] Error getting parquet files for {date_dir}: {e}")
        return []


# ---------------------------------------------------------------------------
# Data Loading & Normalization Internal Helpers
# ---------------------------------------------------------------------------

def parse_dates(date_range: str) -> List[str]:
    """'YYYYMMDD-YYYYMMDD' → sorted list of trading-calendar strings."""
    if not date_range:
        return []
    start, end = date_range.split("-")
    return pd.date_range(start=start, end=end).strftime("%Y%m%d").tolist()


def _load_one_date(
        data_root: str,
        date: str,
        factor_list: List[str],
        y_list: List[str],
        abtypes: Optional[List[int]],
        codes: Optional[List[int]],
) -> Optional[pl.DataFrame]:
    """加载单日数据 (优化版：使用批量读取避免 concat 开销)"""
    date_dir = os.path.join(data_root, date)
    if not os.path.isdir(date_dir):
        return None

    codes_set = set(codes) if codes is not None else None
    abtypes_set = set(abtypes) if abtypes is not None else None

    target_files = []

    with os.scandir(date_dir) as it:
        for entry in it:
            if not entry.name.endswith('.parquet') or not entry.is_file():
                continue

            fname = entry.name.replace(".parquet", "")
            if _filter_parquet_file(fname, codes_set, abtypes_set):
                target_files.append(entry.path)

    if not target_files:
        return None


    # 构建读取列列表
    ordered_cols = _META + factor_list + y_list

    # 2. 利用 dict.fromkeys 去重 (保留顺序，速度极快)
    needed_cols = list(dict.fromkeys(ordered_cols))

    # 3. 核心优化：批量读取
    try:
        df = pl.read_parquet(target_files, columns=needed_cols, use_pyarrow=False)
        df = df.sort(by=['date', 'time_sec'], descending=[False, False])
    except Exception as e:
        # 如果批量读取失败（例如文件 Schema 不一致），回退到安全模式
        print(f"[Warning] Batch read failed for {date}, falling back. Error: {e}")
        return None

    keep = [c for c in _META if c in df.columns] + factor_list + y_list

    df = df.select(keep)
    df = df.with_columns(pl.col("date").cast(pl.Int64))
    return df


def _collect_all_files(
        data_root: str,
        dates: List[str],
        factor_list: List[str],
        y_list: List[str],
        abtypes: Optional[List[int]],
        codes: Optional[List[int]],
) -> Optional[pl.DataFrame]:
    """批量读取多个日期的数据 (非 rolling 模式优化)"""
    codes_set = set(codes) if codes is not None else None
    abtypes_set = set(abtypes) if abtypes is not None else None

    all_files = []
    for date in dates:
        date_dir = os.path.join(data_root, date)
        if not os.path.isdir(date_dir):
            continue

        with os.scandir(date_dir) as it:
            for entry in it:
                if not entry.name.endswith('.parquet') or not entry.is_file():
                    continue

                fname = entry.name.replace(".parquet", "")
                if _filter_parquet_file(fname, codes_set, abtypes_set):
                    all_files.append(entry.path)

    if not all_files:
        return None

    ordered_cols = _META + factor_list + y_list
    needed_cols = list(dict.fromkeys(ordered_cols))

    try:
        df = pl.read_parquet(all_files, columns=needed_cols, use_pyarrow=False)
        df = df.sort(by=['date', 'time_sec'], descending=[False, False])
    except Exception as e:
        print(f"[Warning] Batch read failed for {len(dates)} dates, falling back. Error: {e}")
        return None

    keep = [c for c in _META if c in df.columns] + factor_list + y_list
    df = df.select(keep)
    df = df.with_columns(pl.col("date").cast(pl.Int64))
    return df


def _compute_stats(df: Union[pd.DataFrame, pl.DataFrame], factor_list: List[str]) -> Dict[str, Dict[str, float]]:
    """
     高性能计算统计量 (Polars 版本)。
     利用 Polars 的 Rust 后端进行向量化计算。
     """
    # 1. 筛选存在的列
    cols_exist = [c for c in factor_list if c in df.columns]
    if not cols_exist:
        return {}

    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    # 2. 使用 Polars 原生聚合
    stats_df = (
        df.select(cols_exist)
        .select(pl.col(cols_exist).fill_nan(None))
        .select(
            pl.col(cols_exist).mean().name.suffix("_mean"),
            pl.col(cols_exist).std(ddof=1).name.suffix("_std"),
        )
    )

    # 3. 提取为字典 (单行 DataFrame)
    stats_row = stats_df.row(0)

    # 4. 解析结果并处理异常值
    stats = {}
    for i, col in enumerate(cols_exist):
        mean_val = stats_row[i]
        std_val = stats_row[i + len(cols_exist)]

        # 处理 None/NaN 和极小标准差
        if mean_val is None or (isinstance(mean_val, float) and np.isnan(mean_val)):
            mean_val = 0.0
        if std_val is None or (isinstance(std_val, float) and (np.isnan(std_val) or std_val < 1e-8)):
            std_val = 1.0

        stats[col] = {"mean": float(mean_val), "std": float(std_val)}

    return stats


def _apply_norm(
        df: Union[pd.DataFrame, pl.DataFrame],
        factor_list: List[str],
        stats: Dict[str, Dict[str, float]],
        winsorize_std: float,
) -> pl.DataFrame:
    """新版本：基于 Polars 表达式 (自动处理 Pandas 输入)"""
    is_pandas_input = isinstance(df, pd.DataFrame)
    if is_pandas_input:
        df = pl.from_pandas(df)

    df_cols_set = set(df.columns)
    valid_cols = [col for col in factor_list if col in df_cols_set and col in stats]
    invalid_cols_in_df = [col for col in factor_list if col in df_cols_set and col not in stats]

    exprs = []

    # 处理无效列
    if invalid_cols_in_df:
        exprs.extend([pl.lit(np.nan).alias(c) for c in invalid_cols_in_df])

    # 处理有效列
    if valid_cols:
        for col in valid_cols:
            mean_val = stats[col]["mean"]
            std_val = stats[col]["std"]

            norm_expr = (pl.col(col) - mean_val) / std_val

            if winsorize_std > 0:
                norm_expr = norm_expr.clip(-winsorize_std, winsorize_std)

            # 处理无穷大
            norm_expr = pl.when(norm_expr.is_infinite()).then(np.nan).otherwise(norm_expr)

            exprs.append(norm_expr.alias(col))

    df = df.with_columns(exprs)
    return df


def _apply_group_zscore_norm(
        df: Union[pd.DataFrame, pl.DataFrame],
        factor_list: List[str],
        group_keys: Optional[List[str]] = None,
        winsorize_std: float = 3.0,
) -> pl.DataFrame:
    """按照指定分组键进行组内 z-score 标准化

    Args:
        df: 输入 DataFrame (支持 pandas 和 polars)
        factor_list: 需要标准化的因子列列表
        group_keys: 分组键列表，默认为 ["date", "time_sec", "code"]
        winsorize_std: 截断阈值，标准化后的值会被截断到 [-winsorize_std, winsorize_std]

    Returns:
        标准化后的 polars DataFrame
    """
    if group_keys is None:
        group_keys = ["date", "time_sec", "code"]

    # 统一转为 Polars
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    valid_cols = [col for col in factor_list if col in df.columns]
    if not valid_cols:
        return df

    exprs = []
    for col in valid_cols:
        # 窗口函数直接参与表达式计算，不落地为实际列
        mean_expr = pl.col(col).mean().over(group_keys)
        std_expr = pl.col(col).std(ddof=1).over(group_keys)

        # 1. 计算 Z-Score
        zscore = (pl.col(col) - mean_expr) / std_expr

        # 2. 修复原代码逻辑缺陷：原代码先 clip 后判断 infinite，导致 std=0 产生的 inf 会被截断而无法被替换
        #    此处显式处理 std==0 的情况，量化中通常将常数列的 Z-Score 置为 0
        zscore = pl.when(std_expr == 0).then(pl.lit(0.0)).otherwise(zscore)

        # 3. 缩尾处理
        if winsorize_std > 0:
            zscore = zscore.clip(-winsorize_std, winsorize_std)

        exprs.append(zscore.alias(col))

    # 单次调用完成所有列的计算与原地替换，无需创建/删除中间列
    return df.with_columns(exprs)


# ---------------------------------------------------------------------------
# Load Split API
# ---------------------------------------------------------------------------

def _load_split_rolling(
        data_root: str,
        dates: List[str],
        factor_list: List[str],
        y_list: List[str],
        abtypes: Optional[List[int]],
        codes: Optional[List[int]],
        winsorize_std: float,
        prev_stats: Optional[Dict],
        verbose: bool,
        drop_nulls: bool,
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """Rolling normalization: process day by day with chained stats."""
    dfs: List[pl.DataFrame] = []
    cur_stats: Optional[Dict] = prev_stats

    for date in tqdm(dates, desc=f"Loading rolling", disable=not verbose):
        df_day = _load_one_date(data_root, date, factor_list, y_list, abtypes, codes)
        if df_day is None or len(df_day) == 0:
            continue

        new_stats = _compute_stats(df_day, factor_list)
        if cur_stats is not None:
            df_day = _apply_norm(df_day, factor_list, cur_stats, winsorize_std)
        else:
            # First day ever: no previous stats; mark all factors NaN
            df_day = df_day.with_columns(pl.lit(np.nan).alias(col) for col in factor_list)
        cur_stats = new_stats

        dfs.append(df_day)

    df = pl.concat(dfs) if dfs else pl.DataFrame()

    # Clean NaN/Null if requested
    if drop_nulls and not df.is_empty():
        drop_cols = factor_list + y_list
        df = df.with_columns(pl.col(drop_cols).fill_nan(None))
        df = df.drop_nulls(subset=drop_cols)
        df = df.to_pandas()
    elif drop_nulls and df.is_empty():
        df = df.to_pandas()

    return df, cur_stats


def load_split(
        data_root: str,
        date_range: str,
        factor_list: List[str],
        y_list: List[str],
        abtypes: Optional[List[int]] = None,
        codes: Optional[List[int]] = None,
        normalize: str = "rolling",
        winsorize_std: float = 3.0,
        prev_stats: Optional[Dict] = None,
        verbose: bool = True,
        drop_nulls: bool = True,  # 新增参数：是否自动清理 NaN/Null 行
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """Load one date-range split.

    For rolling normalization, pass ``prev_stats`` from the previous split's
    return value to chain normalization across train → val → test.

    Returns:
        (df, last_day_stats)
        - df: pd.DataFrame if drop_nulls=True, else pl.DataFrame
        - ``last_day_stats`` is only meaningful when normalize=="rolling".
    """
    dates = parse_dates(date_range)

    # Rolling normalization: must process day by day
    if normalize == "rolling":
        return _load_split_rolling(
            data_root, dates, factor_list, y_list, abtypes, codes,
            winsorize_std, prev_stats, verbose, drop_nulls,
        )

    # Other modes: batch read all dates at once
    df = _collect_all_files(
        data_root, dates, factor_list, y_list, abtypes, codes,
    )

    if df is None or df.is_empty():
        if drop_nulls:
            return pd.DataFrame(), None
        return pl.DataFrame(), None

    # Apply normalization if needed
    cur_stats: Optional[Dict] = None
    if normalize == 'global':
        assert prev_stats is not None, "prev_stats must be provided for global normalization"
        df = _apply_norm(df, factor_list, prev_stats, winsorize_std)
        cur_stats = prev_stats
    elif normalize == "group":
        df = _apply_group_zscore_norm(df, factor_list, winsorize_std=winsorize_std)
        cur_stats = None
    # normalize == "none": no normalization needed

    # Clean NaN/Null if requested
    if drop_nulls:
        drop_cols = factor_list + y_list
        df = df.with_columns(pl.col(drop_cols).fill_nan(None))
        df = df.drop_nulls(subset=drop_cols)
        df = df.to_pandas()

    return df, cur_stats


def auto_detect_factor_list(
        data_root: str,
        date: str
) -> List[str]:
    """从HDFS自动检测数据文件中的所有factor列（排除_META和y列）。"""
    hdfs = get_hdfs_connection()
    date_path = f"{HDFS_PATH}/{date}"

    try:
        selector = fs.FileSelector(date_path, recursive=False)
        entries = hdfs.get_file_info(selector)

        parquet_files = []
        for entry in entries:
            if entry.type == fs.FileType.File and entry.base_name.endswith('.parquet'):
                parquet_files.append(f"{date_path}/{entry.base_name}")

        if not parquet_files:
            return []

        dataset = pq.ParquetDataset(parquet_files[0], filesystem=hdfs)
        table = dataset.read()
        df_sample = pl.from_arrow(table)
        all_cols = set(df_sample.columns)
        keywords = ["LOW", "MID", "HIGH", "weight1", "weight2"]
        all_cols = {col for col in all_cols if( not col.endswith('_target')) and (not any(kw in col for kw in keywords))}

        meta_cols = set(_META)

        factor_cols = list(all_cols - meta_cols - {'timestamp', 'bid1', 'ask1'})
        factor_cols.sort()

        print(f"[Data] Auto-detected {len(factor_cols)} factors from HDFS data files")
        return factor_cols
    except Exception as e:
        print(f"[HDFS] Error auto-detecting factors for {date}: {e}")
        return []


# ---------------------------------------------------------------------------
# Resample & Batch Processing
# ---------------------------------------------------------------------------

def process_single_batch(
        batch_tasks: List[Tuple],
        y_map: Dict[str, str],
        original_y_cols: List[str],
        train_date_set: set,
        val_date_set: set,
        test_date_set: set,
        train_interval: int,
        test_interval: int,
        local_path: str,
        aux_hdfs_path: str,
        max_workers: int = 256,
        save_to_local: bool = True,
        save_to_hdfs: bool = False,
        hdfs_output_path: str = "/ext1/user/zli/zjj/ResampleData",
) -> Tuple[int, int, List[Tuple[str, str]]]:
    """处理单个batch：读→算→写
    
    Args:
        batch_tasks: batch任务列表，每个元素为 (remote_file_path, local_file_path, interval, date, fname)
        y_map: y_target->original_y 的映射
        original_y_cols: 原始y列名列表
        train_date_set: 训练日期集合
        val_date_set: 验证日期集合
        test_date_set: 测试日期集合
        train_interval: 训练集interval
        test_interval: 测试集interval
        local_path: 本地输出路径
        aux_hdfs_path: 辅助数据HDFS路径
        max_workers: 写入文件的最大线程数
        save_to_local: 是否保存到本地
        save_to_hdfs: 是否上传到HDFS
        hdfs_output_path: HDFS输出根路径
        
    Returns:
        (success_count, fail_count, saved_files)
        saved_files: List of (local_path, hdfs_path) tuples for files that need to be uploaded
    """
    if not batch_tasks:
        return 0, 0, []
    
    hdfs = get_hdfs_connection()
    
    remote_paths = [t[0] for t in batch_tasks]
    task_map = {t[0]: t[1:] for t in batch_tasks}
    
    try:
        print(f"[Batch] Reading {len(remote_paths)} files from HDFS...")
        table = ds.dataset(remote_paths, format="parquet", filesystem=hdfs).to_table()
        df = pl.from_arrow(table)
        print(f"[Batch] Read done. Total rows: {len(df)}")
    except Exception as e:
        print(f"[Batch] Error reading from HDFS: {e}")
        return 0, len(batch_tasks),[]
    
    missing_original_y_cols = []
    for original_y in original_y_cols:
        if original_y not in df.columns:
            missing_original_y_cols.append(original_y)
    
    join_keys = ["date", "time_sec", "code", "abtype"]
    
    if missing_original_y_cols:
        print(f"[Batch] Missing columns {missing_original_y_cols}. Loading aux data from HDFS...")
        aux_files_to_read = set()
        
        for original_y in missing_original_y_cols:
            target_y_set = {target_y for target_y in y_map if y_map[target_y] == original_y}
            if not target_y_set.issubset(set(df.columns)):
                for r_path, (l_path, interval, date, fname) in task_map.items():
                    aux_path = f"{aux_hdfs_path}/{date}/{fname}"
                    try:
                        file_info = hdfs.get_file_info(aux_path)
                        if file_info.type == fs.FileType.File:
                            aux_files_to_read.add(aux_path)
                    except Exception:
                        pass
        
        if aux_files_to_read:
            try:
                target_y_to_read = [target_y for target_y in y_map if y_map[target_y] in missing_original_y_cols]
                cols_to_read = list(set(join_keys + target_y_to_read))
                
                print(f"[Batch] Reading {len(aux_files_to_read)} HDFS aux files...")
                aux_table = ds.dataset(list(aux_files_to_read), format="parquet", filesystem=hdfs).to_table()
                aux_df = pl.from_arrow(aux_table).select(cols_to_read)
                
                df = df.join(aux_df, on=join_keys, how="inner", validate='1:1')
                del aux_df
                gc.collect()
                print(f"[Batch] Aux data joined. Rows after join: {len(df)}")
            except Exception as e:
                print(f"[Batch] Error reading/joining aux data: {e}")
        else:
            print(f"[Batch] No auxiliary files found.")
    
    try:
        print(f"[Batch] Generating y_targets and resampling...")
        
        df = df.with_columns(
            pl.when(pl.col("date").cast(pl.Int64).cast(pl.Utf8).is_in(train_date_set | val_date_set))
            .then(pl.lit(train_interval))
            .otherwise(pl.lit(test_interval))
            .alias("__interval__")
        )
        
        group_keys = ['date', 'abtype', 'code']
        sort_keys = group_keys + ['time_sec']
        df = df.sort(by=sort_keys, descending=False)
        
        existing_orig_y = [c for c in original_y_cols if c in df.columns]
        if existing_orig_y:
            df = df.with_columns(pl.col(existing_orig_y).fill_nan(None))
        
        df = df.with_columns(
            pl.int_range(0, pl.len()).alias("__global_idx__")
        )
        
        df = df.with_columns(
            pl.int_range(0, pl.len()).over(group_keys).alias("__group_row_idx__"),
            pl.len().over(group_keys).alias("__group_len__"),
            pl.col("__global_idx__").first().over(group_keys).alias("__group_start_global_idx__")
        )
        
        for original_y in set(original_y_cols) - set(missing_original_y_cols):
            for y_target in y_map:
                if y_map[y_target] == original_y:
                    break
            else:
                raise Exception(f'[Batch] Could not find y_target for {original_y_cols}')
            
            if y_target in df.columns:
                continue
            
            if original_y not in df.columns:
                df = df.with_columns(pl.lit(None).alias(y_target))
                continue
            
            first_idx_col = f"__first_idx_{y_target}__"
            
            df = df.with_columns(
                pl.when(pl.col(original_y).is_not_null())
                .then(pl.col("__group_row_idx__"))
                .min()
                .over(group_keys)
                .alias(first_idx_col)
            )
            
            target_idx_expr = pl.col("__group_row_idx__") + pl.col(first_idx_col)
            safe_local_idx = target_idx_expr.clip(0, pl.col("__group_len__") - 1)
            global_target_idx = pl.col("__group_start_global_idx__") + safe_local_idx
            safe_global_idx_for_gather = global_target_idx.fill_null(0)
            
            gathered_val = pl.col(original_y).gather(safe_global_idx_for_gather)
            
            df = df.with_columns(
                pl.when(pl.col(first_idx_col).is_not_null())
                .then(
                    pl.when(target_idx_expr < pl.col("__group_len__"))
                    .then(gathered_val)
                    .otherwise(None)
                )
                .otherwise(None)
                .alias(y_target)
            )
            
            df = df.drop([first_idx_col])
        
        df = df.drop(["__group_row_idx__", "__group_len__", "__global_idx__", "__group_start_global_idx__"])
        
        df = df.with_columns(
            (pl.col("time_sec") // pl.col("__interval__")).alias("resample_time")
        )
        
        resample_keys = group_keys + ["resample_time"]
        df_resampled = df.group_by(resample_keys, maintain_order=True).first()
        del df
        gc.collect()
        
        df_resampled = df_resampled.drop(["resample_time", "__interval__"])
        
        print(f"[Batch] Resample done. Rows: {len(df_resampled)}")
        
        print(f"[Batch] Splitting and saving files...")

        partition_keys = ['date', 'code', 'abtype']

        write_tasks = []
        saved_files = []

        # partition_by 单次哈希划分，避免多次全表扫描 (O(N) vs O(N×G))
        # as_dict=True 返回 {(date, code, abtype): subset} 字典
        # maintain_order=True 保持与原数据相同的分组出现顺序（与原 iter_rows 行为一致）
        groups = df_resampled.partition_by(partition_keys, as_dict=True, maintain_order=True)

        for (date, code, abtype), subset in groups.items():
            # 类型转换与原逻辑保持一致
            date_val = int(date)
            code_val = int(code)
            abtype_val = int(abtype)

            fname = f"{code_val}_{abtype_val}.parquet"

            local_file_path = os.path.join(local_path, str(date_val), fname) if save_to_local else None
            hdfs_file_path = f"{hdfs_output_path}/{date_val}/{fname}" if save_to_hdfs else None

            write_tasks.append((subset, local_file_path, hdfs_file_path))
        
        success_count = 0
        fail_count = 0
        
        # 情况1：仅上传HDFS，多线程上传
        if not save_to_local and save_to_hdfs:
            print(f"[Batch] Uploading to HDFS with multiple threads...")
            
            def upload_single_df(task):
                subset, local_path, hdfs_path = task
                try:
                    hdfs_dir = "/".join(hdfs_path.split("/")[:-1])
                    if not ensure_hdfs_dir(hdfs, hdfs_dir):
                        print(f"Error creating HDFS directory {hdfs_dir}")
                        return False
                    
                    buffer = io.BytesIO()
                    subset.write_parquet(buffer)
                    buffer.seek(0)

                    with hdfs.open_output_stream(hdfs_path) as stream:
                        stream.write(buffer.read())
                    
                    return True
                except Exception as e:
                    print(f"Error writing to HDFS {hdfs_path}: {e}")
                    return False
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(upload_single_df, write_tasks))
            
            success_count = sum(results)
            fail_count = len(results) - success_count
        
        # 情况2和3：保存本地（可能还需要后续上传HDFS）
        elif save_to_local:
            def write_single_df(task):
                sub_df, local_path, hdfs_path = task
                try:
                    sub_df.write_parquet(local_path)
                    return (True, local_path, hdfs_path)
                except Exception as e:
                    print(f"Error writing to local {local_path}: {e}")
                    return (False, None, None)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(write_single_df, write_tasks))
            
            for success, local_path, hdfs_path in results:
                if success:
                    success_count += 1
                    if save_to_hdfs:
                        saved_files.append((local_path, hdfs_path))
                else:
                    fail_count += 1
        
        print(f"[Batch] Save done. Success: {success_count}, Failed: {fail_count}")
        
        return success_count, fail_count, saved_files
        
    except Exception as e:
        print(f"[Batch] Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 0, len(batch_tasks), []
    finally:
        if 'df_resampled' in locals():
            del df_resampled
        if 'task_map' in locals():
            del task_map
        gc.collect()


def upload_local_files_to_hdfs(
        files_to_upload: List[Tuple[str, str]],
        max_workers: int = 256,
) -> Tuple[int, int]:
    """将本地文件上传到HDFS
    
    Args:
        files_to_upload: List of (local_path, hdfs_path) tuples
        max_workers: 最大线程数
        
    Returns:
        (success_count, fail_count)
    """
    if not files_to_upload:
        return 0, 0
    
    print(f"[Upload] Uploading {len(files_to_upload)} files to HDFS...")
    
    hdfs_conn = get_hdfs_connection()

    def upload_single_file(task):
        local_path, hdfs_path = task
        try:
            hdfs_dir = "/".join(hdfs_path.split("/")[:-1])

            if not ensure_hdfs_dir(hdfs_conn, hdfs_dir):
                print(f"Error creating HDFS directory {hdfs_dir}")
                return False
            
            with open(local_path, 'rb') as local_file:
                file_data = local_file.read()

            with hdfs_conn.open_output_stream(hdfs_path) as hdfs_stream:
                hdfs_stream.write(file_data)
            
            return True
        except Exception as e:
            print(f"Error uploading {local_path} to {hdfs_path}: {e}")
            return False
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(upload_single_file, files_to_upload))
    
    success_count = sum(results)
    fail_count = len(results) - success_count
    
    print(f"[Upload] Upload done. Success: {success_count}, Failed: {fail_count}")
    
    return success_count, fail_count


def resample_from_hdfs(
        train_dates: str,
        val_dates: str,
        test_dates: str,
        train_interval: int,
        test_interval: int,
        y_list: List[str],
        codes: Optional[List[int]] = None,
        abtypes: Optional[List[int]] = None,
        remote_path: str = HDFS_PATH,
        local_path: str = "../ResampleData",
        aux_hdfs_path: str = "/ext1/user/zli/zjj/weight",
        max_workers: int = 256,
        save_to_local: bool = True,
        save_to_hdfs: bool = False,
        hdfs_output_path: str = "/ext1/user/zli/zjj/ResampleData",
) -> Tuple[int, int]:
    """从HDFS批量读取数据，生成y_target并resample

    注意：本函数现在一次性读取整个batch的数据进行处理
    
    Args:
        train_dates: 训练日期范围
        val_dates: 验证日期范围
        test_dates: 测试日期范围
        train_interval: 训练集interval
        test_interval: 测试集interval
        y_list: y列名列表
        codes: 代码列表
        abtypes: abtype列表
        remote_path: HDFS源数据路径
        local_path: 本地输出路径
        aux_hdfs_path: 辅助数据HDFS路径
        max_workers: 写入文件的最大线程数
        save_to_local: 是否保存到本地，默认True
        save_to_hdfs: 是否上传到HDFS，默认False
        hdfs_output_path: HDFS输出根路径，默认/ext1/user/zli/zjj/ResampleData
        
    Returns:
        (success_count, fail_count)
    """
    train_date_list = parse_dates(train_dates) if train_dates else []
    val_date_list = parse_dates(val_dates) if val_dates else []
    test_date_list = parse_dates(test_dates) if test_dates else []

    all_dates = set(train_date_list + val_date_list + test_date_list)

    # 映射: y_target -> original_y
    y_map = {y: y.rsplit('_', 1)[0] for y in y_list}
    original_y_cols = list(set(y_map.values()))

    if not all_dates:
        print("[Resample] No dates specified.")
        return 0, 0

    all_dates = sorted(all_dates)
    print(f"[Resample] Target dates: {all_dates[0]} ~ {all_dates[-1]} ({len(all_dates)} days)")

    train_date_set = set(train_date_list)
    val_date_set = set(val_date_list)
    test_date_set = set(test_date_list)

    # 获取HDFS连接和日期目录
    hdfs = get_hdfs_connection()
    hdfs_dates = set(get_date_directories(remote_path))
    dates_to_process = [d for d in all_dates if d in hdfs_dates]

    if not dates_to_process:
        print("[Resample] No dates found in HDFS.")
        return 0, 0

    print(f"[Resample] Processing {len(dates_to_process)} dates found in HDFS")

    # 收集任务
    all_tasks = []
    for date in dates_to_process:
        date_path = f"{remote_path}/{date}"
        local_date_path = os.path.join(local_path, date)
        os.makedirs(local_date_path, exist_ok=True)

        parquet_files = get_parquet_files(date, codes=codes, abtypes=abtypes, main_contracts_only=True)

        if not parquet_files:
            continue

        for fname in parquet_files:
            remote_file_path = f"{date_path}/{fname}"
            local_file_path = os.path.join(local_date_path, fname)

            # 确定该文件对应的interval
            if date in train_date_set or date in val_date_set:
                interval = train_interval
            elif date in test_date_set:
                interval = test_interval
            else:
                continue

            all_tasks.append((remote_file_path, local_file_path, interval, date, fname))

    if not all_tasks:
        print("[Resample] No files to process.")
        return 0, 0

    print(f"[Resample] Total files to process: {len(all_tasks)}")

    batch_size = 100
    total_batches = (len(all_tasks) + batch_size - 1) // batch_size
    print(f"[Resample] Processing in {total_batches} batches with 8 parallel threads")

    total_success = 0
    total_fail = 0
    all_saved_files = []
    
    # 预先收集所有本地目录，避免重复创建
    if save_to_local:
        for date in dates_to_process:
            os.makedirs(os.path.join(local_path, date), exist_ok=True)
    
    # 准备所有batch任务
    all_batches = []
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(all_tasks))
        batch_tasks = all_tasks[start_idx:end_idx]
        all_batches.append(batch_tasks)
    
    print(f"[Resample] Starting parallel processing with ThreadPoolExecutor...")
    
    # 使用线程池并行处理batch
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for batch_idx, batch_tasks in enumerate(all_batches):
            future = executor.submit(
                process_single_batch,
                batch_tasks,
                y_map,
                original_y_cols,
                train_date_set,
                val_date_set,
                test_date_set,
                train_interval,
                test_interval,
                local_path,
                aux_hdfs_path,
                max_workers,
                save_to_local,
                save_to_hdfs,
                hdfs_output_path
            )
            futures.append((batch_idx, future))
        
        # 收集结果
        for batch_idx, future in tqdm(futures, total=len(futures), desc="[Resample] Processing batches"):
            try:
                success, fail, saved_files = future.result()
                total_success += success
                total_fail += fail
                all_saved_files.extend(saved_files)
            except Exception as e:
                print(f"[Resample] Batch {batch_idx + 1} failed with exception: {e}")
                import traceback
                traceback.print_exc()
                total_fail += len(all_batches[batch_idx])

    # 如果既保存本地又上传HDFS，在所有batch完成后上传
    if save_to_local and save_to_hdfs and all_saved_files:
        print(f"[Resample] All batches completed. Starting upload to HDFS...")
        upload_local_files_to_hdfs(all_saved_files, max_workers)

    print(f"[Resample] All done. Total Success: {total_success}, Total Failed: {total_fail}")
    return total_success, total_fail


# ---------------------------------------------------------------------------
# Main Public API
# ---------------------------------------------------------------------------

def load_all_splits(
        data_root: str,
        train_dates: str,
        val_dates: str,
        test_dates: str,
        factor_list: Optional[List[str]],
        y_list: List[str],
        abtypes: Optional[List[int]] = None,
        codes: Optional[List[int]] = None,
        normalize: str = "rolling",
        winsorize_std: float = 3.0,
        verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Optional[Dict]]:
    """Load and normalise all three data splits.

    Rolling: train → val → test chained (no look-ahead bias).
    Global:  scaler fitted on train, applied to val/test.
    None:    raw values returned.

    Returns:
        (df_train, df_val, df_test, factor_list, cur_stats)
        Each DataFrame has columns: metadata + factor_list + y_list
        Rows with NaN in any factor or label are dropped.
        ``cur_stats`` is the final stats dict (last day for rolling,
        train-fitted for global, None for none).
    """

    cur_stats: Optional[Dict] = None

    # Auto-detect factor_list if not provided
    if factor_list is None:
        dates = parse_dates(train_dates)
        if not dates:
            raise ValueError("No valid dates in train_dates and factor_list is None")
        factor_list = []
        date_index = 0
        len_dates = len(dates)
        while (not factor_list) and (date_index < len_dates):
            factor_list = auto_detect_factor_list(data_root, dates[date_index])
            date_index += 1

        if not factor_list:
            raise ValueError("Could not auto-detect factor_list from data files")

    if normalize == "rolling":
        print("[Data] Rolling normalisation: train split, date range: ", train_dates)
        df_train, last = load_split(
            data_root, train_dates, factor_list, y_list,
            abtypes, codes, "rolling", winsorize_std, None, verbose,
            drop_nulls=True,  # 启用自动清洗
        )
        print("[Data] Rolling normalisation: val split, date range: ", val_dates)
        df_val, last = load_split(
            data_root, val_dates, factor_list, y_list,
            abtypes, codes, "rolling", winsorize_std, last, verbose,
            drop_nulls=True,
        )

        if test_dates:
            print("[Data] Rolling normalisation: test split, date range: ", test_dates)
            df_test, last = load_split(
                data_root, test_dates, factor_list, y_list,
                abtypes, codes, "rolling", winsorize_std, last, verbose,
                drop_nulls=True,
            )
        else:
            df_test = pd.DataFrame()

        cur_stats = last

    elif normalize == "group":
        print("[Data] Group z-score normalisation: train split, date range: ", train_dates)
        df_train, _ = load_split(
            data_root, train_dates, factor_list, y_list,
            abtypes, codes, "group", winsorize_std, None, verbose,
            drop_nulls=True,
        )
        print("[Data] Group z-score normalisation: val split, date range: ", val_dates)
        df_val, _ = load_split(
            data_root, val_dates, factor_list, y_list,
            abtypes, codes, "group", winsorize_std, None, verbose,
            drop_nulls=True,
        )

        if test_dates:
            print("[Data] Group z-score normalisation: test split, date range: ", test_dates)
            df_test, _ = load_split(
                data_root, test_dates, factor_list, y_list,
                abtypes, codes, "group", winsorize_std, None, verbose,
                drop_nulls=True,
            )
        else:
            df_test = pd.DataFrame()

        cur_stats = None

    elif normalize == "global":
        # Step 1: Load train split (raw) to compute global stats
        print("[Data] Global normalisation: loading train split to compute stats...")
        df_train_raw, _ = load_split(
            data_root, train_dates, factor_list, y_list,
            abtypes, codes, "none", winsorize_std, None, verbose,
            drop_nulls=False,  # 先不清洗，用于计算统计量
        )

        # Step 2: Compute stats on train only
        global_stats = _compute_stats(df_train_raw, factor_list)
        cur_stats = global_stats
        print(f"[Data] Global stats computed from {len(df_train_raw):,} train rows")

        # Step 3: Apply global stats to all splits (with cleaning)
        df_train = _apply_norm(df_train_raw, factor_list, global_stats, winsorize_std)
        # 手动清洗（因为 load_split with drop_nulls=False 返回 polars）
        drop_cols = factor_list + y_list
        df_train = df_train.with_columns(pl.col(drop_cols).fill_nan(None)).drop_nulls(subset=drop_cols)
        df_train = df_train.to_pandas()

        print("[Data] Global normalisation: val split...")
        df_val_raw, _ = load_split(
            data_root, val_dates, factor_list, y_list,
            abtypes, codes, "none", winsorize_std, None, verbose,
            drop_nulls=False,
        )
        df_val = _apply_norm(df_val_raw, factor_list, global_stats, winsorize_std)
        df_val = df_val.with_columns(pl.col(drop_cols).fill_nan(None)).drop_nulls(subset=drop_cols)
        df_val = df_val.to_pandas()

        if test_dates:
            print("[Data] Global normalisation: test split...")
            df_test_raw, _ = load_split(
                data_root, test_dates, factor_list, y_list,
                abtypes, codes, "none", winsorize_std, None, verbose,
                drop_nulls=False,
            )
            df_test = _apply_norm(df_test_raw, factor_list, global_stats, winsorize_std)
            df_test = df_test.with_columns(pl.col(drop_cols).fill_nan(None)).drop_nulls(subset=drop_cols)
            df_test = df_test.to_pandas()
        else:
            df_test = pd.DataFrame()

    else:  # normalize == "none"
        print("[Data] No normalisation.")
        df_train, _ = load_split(
            data_root, train_dates, factor_list, y_list,
            abtypes, codes, "none", winsorize_std, None, verbose,
            drop_nulls=True,  # 直接启用清洗
        )
        df_val, _ = load_split(
            data_root, val_dates, factor_list, y_list,
            abtypes, codes, "none", winsorize_std, None, verbose,
            drop_nulls=True,
        )
        if test_dates:
            df_test, _ = load_split(
                data_root, test_dates, factor_list, y_list,
                abtypes, codes, "none", winsorize_std, None, verbose,
                drop_nulls=True,
            )
        else:
            df_test = pd.DataFrame()

    # 检查 rolling 模式下训练集是否为空
    if isinstance(df_train, pd.DataFrame) and df_train.empty and normalize == "rolling":
        print(
            "[Data] WARNING: Training set is empty after rolling normalisation.\n"
            "  With rolling norm, the first trading day of each split always\n"
            "  becomes NaN (no previous-day stats) and is dropped.\n"
            "  If train_dates contains only 1 date, all training rows will be\n"
            "  removed.  Either add more dates or use normalize='global'/'none'."
        )

    print(
        f"[Data] Loaded → train:{len(df_train):,}  val:{len(df_val):,}  test:{len(df_test):,} rows"
    )
    return df_train, df_val, df_test, factor_list, cur_stats
