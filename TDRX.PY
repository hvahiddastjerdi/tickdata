"""
TDR_V5.py - Dukascopy Tick Data Downloader & Storage (Optimized v5.0)
======================================================================

Specialized for:
  - Downloading tick data from Dukascopy with cancel support
  - Optimized Parquet storage with Zstd compression + Delta Encoding
  - Smart filename conventions for ticks and candles
  - Session detection and timezone conversion

Features:
  1. download_ticks()     - Download with cancel_flag + progress + auto_save
  2. save_ticks()         - Parquet + Zstd(3) + Delta Encoding
  3. load_ticks()         - Auto Delta Decode from metadata
  4. ticks_to_candles()   - Convert with session detection + timezone
  5. save_candles()       - Parquet + Zstd compression
  6. load_candles()       - Load candle data

Filename Conventions:
  - Ticks:   {SYMBOL}_{YYMMDD_start}_{YYMMDD_end}.parquet
  - Candles: {SYMBOL}_{YYMMDD_start}_{YYMMDD_end}_{TF}.parquet

Storage Optimization:
  - Zstandard compression level 3 (best speed/ratio balance)
  - Delta encoding for timestamp_us, ask, bid columns
  - ~60-70% smaller files vs uncompressed Parquet
  - ~10x smaller than CSV

Session Columns (Binary):
  - sydney, tokyo, london, newyork (True/False for each)

Author : TDR Project
Version: 5.0.1
Date   : 1404/11/28 (2026-02-17)
"""

from __future__ import annotations

import lzma
import os
import re
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests as req

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tdr_v5.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TDR_V5')

# ============================================================
# CONSTANTS & DEFINITIONS
# ============================================================

SYMBOLS: Dict[str, Dict[str, float]] = {
    "Forex Major": {
        "EURUSD": 1e-5, "GBPUSD": 1e-5, "USDJPY": 1e-3,
        "USDCHF": 1e-5, "AUDUSD": 1e-5, "USDCAD": 1e-5, "NZDUSD": 1e-5,
    },
    "Forex Minor": {
        "EURGBP": 1e-5, "EURJPY": 1e-3, "GBPJPY": 1e-3,
        "AUDNZD": 1e-5, "EURCHF": 1e-5, "AUDCAD": 1e-5,
        "GBPCHF": 1e-5, "CADJPY": 1e-3, "EURAUD": 1e-5, "GBPAUD": 1e-5,
    },
    "Metals": {
        "XAUUSD": 1e-3, "XAGUSD": 1e-5, "XPTUSD": 1e-2, "XPDUSD": 1e-2,
    },
    "Crypto": {
        "BTCUSD": 1e-2, "ETHUSD": 1e-2, "LTCUSD": 1e-2,
    },
    "Indices": {
        "USA500IDXUSD": 1e-2, "USATECHIDXUSD": 1e-2, "USA30IDXUSD": 1e-2,
        "DEUIDXEUR": 1e-2, "JPNIDXJPY": 1e-0, "FRAIDXEUR": 1e-2,
        "UKBIDXGBP": 1e-2, "ESPIDXEUR": 1e-2, "CHNIDXUSD": 1e-2, "HKGIDXHKD": 1e-2,
    },
    "Energy": {
        "USOUSD": 1e-3, "UKOUSD": 1e-3, "NGASUSD": 1e-3,
    },
}

TIMEFRAMES: Dict[str, str] = {
    "1m": "1min", "2m": "2min", "3m": "3min", "5m": "5min",
    "10m": "10min", "15m": "15min", "30m": "30min",
    "1H": "1h", "2H": "2h", "4H": "4h",
}

TIMEFRAME_DURATIONS: Dict[str, timedelta] = {
    "1m": timedelta(minutes=1), "2m": timedelta(minutes=2),
    "3m": timedelta(minutes=3), "5m": timedelta(minutes=5),
    "10m": timedelta(minutes=10), "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30), "1H": timedelta(hours=1),
    "2H": timedelta(hours=2), "4H": timedelta(hours=4),
}

SESSION_HOURS: Dict[str, Tuple[int, int]] = {
    "sydney":  (21, 6),   # 21:00 - 06:00 UTC (crosses midnight)
    "tokyo":   (0, 9),    # 00:00 - 09:00 UTC
    "london":  (7, 16),   # 07:00 - 16:00 UTC
    "newyork": (13, 22),  # 13:00 - 22:00 UTC
}

# Session column names (for binary output)
SESSION_COLUMNS: List[str] = ["sydney", "tokyo", "london", "newyork"]

TIMEZONES: Dict[str, str] = {
    "UTC": "UTC",
    "EET": "Europe/Athens",        # Eastern European Time (MT4/MT5 default)
    "Tehran": "Asia/Tehran",
    "Moscow": "Europe/Moscow",
    "London": "Europe/London",
    "NewYork": "America/New_York",
    "Tokyo": "Asia/Tokyo",
    "Singapore": "Asia/Singapore",
}

# Dukascopy constants
_BASE_URL: str = "https://datafeed.dukascopy.com/datafeed"
_MAX_RETRIES: int = 3
_RETRY_DELAY: float = 1.0
_DOWNLOAD_WORKERS: int = 20

# Global cancel flag
cancel_flag: bool = False

# Numpy dtype for tick data
_TICK_DTYPE = np.dtype([
    ('timestamp_us', 'i8'),
    ('ask',          'f8'),
    ('bid',          'f8'),
    ('ask_volume',   'f4'),
    ('bid_volume',   'f4'),
])

# Raw Dukascopy binary format
_RAW_DTYPE = np.dtype([
    ('ms_offset', '>u4'),
    ('ask_int',   '>u4'),
    ('bid_int',   '>u4'),
    ('ask_vol',   '>f4'),
    ('bid_vol',   '>f4'),
])

# Parquet compression settings
PARQUET_COMPRESSION = 'zstd'
PARQUET_COMPRESSION_LEVEL = 3

# ============================================================
# THREAD-LOCAL HTTP SESSION
# ============================================================

_thread_local = threading.local()

def _get_thread_session() -> req.Session:
    """Returns a per-thread HTTP session with connection pooling."""
    if not hasattr(_thread_local, 'session'):
        s = req.Session()
        adapter = req.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=0,
        )
        s.mount('https://', adapter)
        s.mount('http://', adapter)
        _thread_local.session = s
    return _thread_local.session

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_point(symbol: str) -> float:
    """Returns the point (pip precision) for a symbol."""
    for group in SYMBOLS.values():
        if symbol in group:
            return group[symbol]
    raise ValueError(f"Symbol '{symbol}' not found in SYMBOLS")

def flat_symbols() -> Dict[str, float]:
    """Returns flat dictionary of all symbols."""
    result: Dict[str, float] = {}
    for group in SYMBOLS.values():
        result.update(group)
    return result

def get_timezone(tz_key: str) -> ZoneInfo:
    """Returns ZoneInfo for timezone key."""
    tz_name = TIMEZONES.get(tz_key, "UTC")
    return ZoneInfo(tz_name)

# ============================================================
# FILENAME UTILITIES
# ============================================================

def generate_tick_filename(
    symbol: str,
    start: datetime,
    end: datetime,
    directory: Optional[str] = None,
) -> str:
    """
    Generates tick filename: {SYMBOL}_{YYMMDD_start}_{YYMMDD_end}.parquet

    Args:
        symbol: Trading symbol (e.g., "XAUUSD")
        start: Start datetime
        end: End datetime
        directory: Optional directory path

    Returns:
        Full filepath string

    Example:
        >>> generate_tick_filename("XAUUSD", datetime(2024,1,1), datetime(2024,6,15))
        'XAUUSD_240101_240615.parquet'
    """
    start_str = start.strftime("%y%m%d")
    end_str = end.strftime("%y%m%d")
    filename = f"{symbol}_{start_str}_{end_str}.parquet"

    if directory:
        return os.path.join(directory, filename)
    return filename

def generate_candle_filename(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
    directory: Optional[str] = None,
) -> str:
    """
    Generates candle filename: {SYMBOL}_{YYMMDD_start}_{YYMMDD_end}_{TF}.parquet

    Args:
        symbol: Trading symbol
        start: Start datetime
        end: End datetime
        timeframe: Timeframe string (e.g., "1H", "4H")
        directory: Optional directory path

    Returns:
        Full filepath string

    Example:
        >>> generate_candle_filename("XAUUSD", datetime(2024,1,1), datetime(2024,6,15), "1H")
        'XAUUSD_240101_240615_1H.parquet'
    """
    start_str = start.strftime("%y%m%d")
    end_str = end.strftime("%y%m%d")
    tf_upper = timeframe.upper()
    filename = f"{symbol}_{start_str}_{end_str}_{tf_upper}.parquet"

    if directory:
        return os.path.join(directory, filename)
    return filename

def parse_tick_filename(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Parses tick filename to extract metadata.

    Args:
        filepath: Path to tick parquet file

    Returns:
        Dict with symbol, start, end or None if invalid

    Example:
        >>> parse_tick_filename("XAUUSD_240101_240615.parquet")
        {'symbol': 'XAUUSD', 'start': datetime(2024,1,1), 'end': datetime(2024,6,15)}
    """
    filename = os.path.basename(filepath)
    pattern = r'^([A-Z0-9]+)_(\d{6})_(\d{6})\.parquet$'
    match = re.match(pattern, filename)

    if not match:
        return None

    symbol = match.group(1)
    start_str = match.group(2)
    end_str = match.group(3)

    try:
        start = datetime.strptime(start_str, "%y%m%d")
        end = datetime.strptime(end_str, "%y%m%d")
        return {"symbol": symbol, "start": start, "end": end}
    except ValueError:
        return None

def parse_candle_filename(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Parses candle filename to extract metadata.

    Args:
        filepath: Path to candle parquet file

    Returns:
        Dict with symbol, start, end, timeframe or None if invalid

    Example:
        >>> parse_candle_filename("XAUUSD_240101_240615_1H.parquet")
        {'symbol': 'XAUUSD', 'start': datetime(2024,1,1), 'end': datetime(2024,6,15), 'timeframe': '1H'}
    """
    filename = os.path.basename(filepath)
    pattern = r'^([A-Z0-9]+)_(\d{6})_(\d{6})_(\d+[mhMH]|[A-Z0-9]+)\.parquet$'
    match = re.match(pattern, filename)

    if not match:
        return None

    symbol = match.group(1)
    start_str = match.group(2)
    end_str = match.group(3)
    timeframe = match.group(4).upper()

    try:
        start = datetime.strptime(start_str, "%y%m%d")
        end = datetime.strptime(end_str, "%y%m%d")
        return {"symbol": symbol, "start": start, "end": end, "timeframe": timeframe}
    except ValueError:
        return None

# ============================================================
# DELTA ENCODING / DECODING
# ============================================================

def delta_encode(arr: np.ndarray) -> np.ndarray:
    """
    Applies delta encoding to a 1D array.
    First value stays, rest become differences.

    Args:
        arr: Input array (int64 or float64)

    Returns:
        Delta-encoded array

    Example:
        >>> delta_encode(np.array([100, 102, 105, 103]))
        array([100, 2, 3, -2])
    """
    if len(arr) == 0:
        return arr.copy()

    result = np.empty_like(arr)
    result[0] = arr[0]
    result[1:] = np.diff(arr)
    return result

def delta_decode(arr: np.ndarray) -> np.ndarray:
    """
    Reverses delta encoding (cumulative sum).

    Args:
        arr: Delta-encoded array

    Returns:
        Original array

    Example:
        >>> delta_decode(np.array([100, 2, 3, -2]))
        array([100, 102, 105, 103])
    """
    if len(arr) == 0:
        return arr.copy()

    return np.cumsum(arr)

def delta_encode_prices(arr: np.ndarray, precision: int = 6) -> np.ndarray:
    """
    Delta encodes float prices by converting to integers first.

    Args:
        arr: Float price array
        precision: Decimal precision (default 6)

    Returns:
        Delta-encoded integer array
    """
    multiplier = 10 ** precision
    int_arr = (arr * multiplier).astype(np.int64)
    return delta_encode(int_arr)

def delta_decode_prices(arr: np.ndarray, precision: int = 6) -> np.ndarray:
    """
    Decodes delta-encoded prices back to floats.

    Args:
        arr: Delta-encoded integer array
        precision: Decimal precision used during encoding

    Returns:
        Original float price array
    """
    multiplier = 10 ** precision
    int_arr = delta_decode(arr)
    return int_arr.astype(np.float64) / multiplier

# ============================================================
# DOWNLOAD: SINGLE HOUR
# ============================================================

def _download_hour_numpy(
    symbol: str,
    dt: datetime,
    point: float,
    http_session: Optional[req.Session] = None,
) -> Optional[np.ndarray]:
    """
    Downloads ONE hour of tick data from Dukascopy.

    Args:
        symbol: Trading symbol
        dt: Hour datetime
        point: Price precision multiplier
        http_session: Optional HTTP session for connection reuse

    Returns:
        Numpy structured array or None
    """
    if http_session is None:
        http_session = _get_thread_session()

    # Dukascopy uses 0-based months
    url = (
        f"{_BASE_URL}/{symbol}/"
        f"{dt.year}/{dt.month - 1:02d}/{dt.day:02d}/"
        f"{dt.hour:02d}h_ticks.bi5"
    )

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            res = http_session.get(url, timeout=10)

            if res.status_code != 200 or len(res.content) == 0:
                return None

            data = lzma.decompress(res.content)

            if len(data) == 0 or len(data) % 20 != 0:
                return None

            n_ticks = len(data) // 20
            raw = np.frombuffer(data, dtype=_RAW_DTYPE, count=n_ticks)

            base_us = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)

            out = np.empty(n_ticks, dtype=_TICK_DTYPE)
            out['timestamp_us'] = base_us + raw['ms_offset'].astype(np.int64) * 1000
            out['ask'] = raw['ask_int'].astype(np.float64) * point
            out['bid'] = raw['bid_int'].astype(np.float64) * point
            out['ask_volume'] = raw['ask_vol']
            out['bid_volume'] = raw['bid_vol']

            return out

        except lzma.LZMAError:
            return None
        except (req.exceptions.RequestException, Exception):
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
                continue
            return None

    return None

# ============================================================
# 1. DOWNLOAD TICKS
# ============================================================

def download_ticks(
    symbol: str,
    start: datetime,
    end: datetime,
    point: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_workers: int = _DOWNLOAD_WORKERS,
    auto_save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Downloads tick data from Dukascopy with parallel requests.

    Features:
        - Parallel downloading with ThreadPoolExecutor
        - Cancel support via global cancel_flag
        - Progress callback for GUI integration
        - Optional auto-save on completion or cancellation

    Args:
        symbol: Trading symbol (e.g., "XAUUSD")
        start: Start datetime (inclusive)
        end: End datetime (exclusive)
        point: Price precision (auto-detected if None)
        progress_callback: callback(current, total, message)
        max_workers: Number of parallel threads
        auto_save_path: If provided, saves partial data on cancel

    Returns:
        Numpy structured array with tick data

    Example:
        >>> ticks = download_ticks("XAUUSD", datetime(2024,1,1), datetime(2024,1,31))
        >>> len(ticks)
        5000000
    """
    global cancel_flag
    cancel_flag = False

    if point is None:
        point = get_point(symbol)

    # Build hour list
    hours: List[datetime] = []
    current = start.replace(minute=0, second=0, microsecond=0)
    while current < end:
        hours.append(current)
        current += timedelta(hours=1)

    total = len(hours)
    if total == 0:
        return np.empty(0, dtype=_TICK_DTYPE)

    results: Dict[int, Optional[np.ndarray]] = {}
    completed_count = 0
    was_cancelled = False

    def _worker(idx: int, hour_dt: datetime) -> Tuple[int, Optional[np.ndarray]]:
        session = _get_thread_session()
        arr = _download_hour_numpy(symbol, hour_dt, point, session)
        return (idx, arr)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker, idx, hour_dt): idx
            for idx, hour_dt in enumerate(hours)
        }

        for future in as_completed(futures):
            if cancel_flag:
                was_cancelled = True
                for f in futures:
                    f.cancel()
                if progress_callback:
                    progress_callback(completed_count, total, "❌ Cancelled by user")
                break

            idx, arr = future.result()
            results[idx] = arr
            completed_count += 1

            if progress_callback:
                hour_dt = hours[idx]
                msg = f"Downloaded {hour_dt.strftime('%Y-%m-%d %H:%M')} ({completed_count}/{total})"
                progress_callback(completed_count, total, msg)

    # Concatenate in order
    arrays: List[np.ndarray] = []
    for idx in range(total):
        arr = results.get(idx)
        if arr is not None and len(arr) > 0:
            arrays.append(arr)

    if not arrays:
        return np.empty(0, dtype=_TICK_DTYPE)

    all_ticks = np.concatenate(arrays)

    # Auto-save if path provided
    if auto_save_path and len(all_ticks) > 0:
        if was_cancelled:
            # Save partial data with _partial suffix
            base, ext = os.path.splitext(auto_save_path)
            partial_path = f"{base}_partial{ext}"
            save_ticks(all_ticks, partial_path)
            logger.info(f"Partial data saved to: {partial_path}")
        else:
            save_ticks(all_ticks, auto_save_path)
            logger.info(f"Data saved to: {auto_save_path}")

    if progress_callback and not was_cancelled:
        progress_callback(total, total, f"✅ Done — {len(all_ticks):,} ticks downloaded")

    return all_ticks

# ============================================================
# 2. SAVE TICKS (Parquet + Zstd + Delta Encoding)
# ============================================================

def save_ticks(
    ticks: np.ndarray,
    filepath: str,
    use_delta_encoding: bool = True,
) -> None:
    """
    Saves tick data to Parquet with Zstd compression and Delta Encoding.

    Storage format:
        - Parquet columnar format
        - Zstandard compression (level 3)
        - Delta encoding for timestamp_us, ask, bid
        - Metadata stores encoding info for auto-decode on load

    Args:
        ticks: Numpy structured array with tick data
        filepath: Output file path (.parquet)
        use_delta_encoding: Whether to apply delta encoding (default True)

    Example:
        >>> save_ticks(tick_array, "XAUUSD_240101_240131.parquet")
    """
    if len(ticks) == 0:
        logger.warning("No ticks to save")
        return

    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Prepare data
    if use_delta_encoding:
        timestamp_data = delta_encode(ticks['timestamp_us'])
        ask_data = delta_encode_prices(ticks['ask'], precision=6)
        bid_data = delta_encode_prices(ticks['bid'], precision=6)
    else:
        timestamp_data = ticks['timestamp_us']
        ask_data = ticks['ask']
        bid_data = ticks['bid']

    # Build PyArrow table
    if use_delta_encoding:
        table = pa.table({
            'timestamp_us': pa.array(timestamp_data, type=pa.int64()),
            'ask': pa.array(ask_data, type=pa.int64()),
            'bid': pa.array(bid_data, type=pa.int64()),
            'ask_volume': pa.array(ticks['ask_volume'], type=pa.float32()),
            'bid_volume': pa.array(ticks['bid_volume'], type=pa.float32()),
        })
    else:
        table = pa.table({
            'timestamp_us': pa.array(timestamp_data, type=pa.int64()),
            'ask': pa.array(ticks['ask'], type=pa.float64()),
            'bid': pa.array(ticks['bid'], type=pa.float64()),
            'ask_volume': pa.array(ticks['ask_volume'], type=pa.float32()),
            'bid_volume': pa.array(ticks['bid_volume'], type=pa.float32()),
        })

    # Metadata for auto-decode
    metadata = {
        b'tdr_version': b'5.0',
        b'delta_encoded': b'true' if use_delta_encoding else b'false',
        b'price_precision': b'6',
        b'tick_count': str(len(ticks)).encode(),
        b'created_at': datetime.now(timezone.utc).isoformat().encode(),
    }

    table = table.replace_schema_metadata(metadata)

    # Write with Zstd compression
    pq.write_table(
        table,
        filepath,
        compression='zstd',
        compression_level=PARQUET_COMPRESSION_LEVEL,
        use_dictionary=False,  # Not useful for tick data
    )

    logger.info(f"Saved {len(ticks):,} ticks to {filepath}")

# ============================================================
# 3. LOAD TICKS (Auto Delta Decode)
# ============================================================

def load_ticks(filepath: str) -> np.ndarray:
    """
    Loads tick data from Parquet with automatic Delta Decoding.

    Reads metadata to determine if delta encoding was used,
    and automatically decodes if necessary.

    Args:
        filepath: Path to parquet file

    Returns:
        Numpy structured array with original tick data

    Example:
        >>> ticks = load_ticks("XAUUSD_240101_240131.parquet")
        >>> ticks['ask'][:5]
        array([2045.123, 2045.125, 2045.120, ...])
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read parquet file
    table = pq.read_table(filepath)
    metadata = table.schema.metadata or {}

    # Check if delta encoded
    is_delta_encoded = metadata.get(b'delta_encoded', b'false') == b'true'
    price_precision = int(metadata.get(b'price_precision', b'6'))

    # Convert to pandas for easier manipulation
    df = table.to_pandas()
    n = len(df)

    if n == 0:
        return np.empty(0, dtype=_TICK_DTYPE)

    # Prepare output array
    out = np.empty(n, dtype=_TICK_DTYPE)

    # Decode data
    if is_delta_encoded:
        out['timestamp_us'] = delta_decode(df['timestamp_us'].values)
        out['ask'] = delta_decode_prices(df['ask'].values, precision=price_precision)
        out['bid'] = delta_decode_prices(df['bid'].values, precision=price_precision)
    else:
        out['timestamp_us'] = df['timestamp_us'].values.astype(np.int64)
        out['ask'] = df['ask'].values.astype(np.float64)
        out['bid'] = df['bid'].values.astype(np.float64)

    out['ask_volume'] = df['ask_volume'].values.astype(np.float32)
    out['bid_volume'] = df['bid_volume'].values.astype(np.float32)

    logger.info(f"Loaded {n:,} ticks from {filepath}")
    return out

# ============================================================
# SESSION DETECTION (Vectorized) - Binary Columns
# ============================================================

def _detect_sessions_vectorized(
    timestamps_utc: pd.DatetimeIndex,
    duration: timedelta,
) -> pd.DataFrame:
    """
    Detects trading sessions for each candle using vectorized operations.
    Returns a DataFrame with binary columns for each session.

    Sessions are assigned based on UTC time overlap:
        - sydney:  21:00 - 06:00 UTC (crosses midnight)
        - tokyo:   00:00 - 09:00 UTC
        - london:  07:00 - 16:00 UTC
        - newyork: 13:00 - 22:00 UTC

    Args:
        timestamps_utc: Candle start times (must be UTC)
        duration: Candle duration

    Returns:
        DataFrame with boolean columns: ['sydney', 'tokyo', 'london', 'newyork']
        Each column is True if the candle overlaps with that session.

    Example:
        >>> sessions_df = _detect_sessions_vectorized(candles.index, timedelta(hours=1))
        >>> sessions_df.head()
           sydney  tokyo  london  newyork
        0    True   True   False    False
        1   False   True    True    False
    """
    session_keys = list(SESSION_HOURS.keys())
    n = len(timestamps_utc)

    # Return empty DataFrame with correct columns if no data
    if n == 0:
        return pd.DataFrame({k: pd.Series(dtype=bool) for k in session_keys})

    # Candle start time in minutes (UTC)
    start_minutes = (
        timestamps_utc.hour.to_numpy(dtype=np.int16) * 60
        + timestamps_utc.minute.to_numpy(dtype=np.int16)
    )

    # Candle end time in minutes (handle wrap around midnight)
    duration_minutes = max(1, int(duration.total_seconds() // 60))
    end_minutes = (start_minutes + duration_minutes - 1) % 1440

    # Build result dictionary
    data = {}

    for key in session_keys:
        s_start_h, s_end_h = SESSION_HOURS[key]
        s_start = s_start_h * 60  # Convert to minutes
        s_end = s_end_h * 60      # Convert to minutes

        if s_start < s_end:
            # Normal session (no midnight cross)
            # Session is active from s_start to s_end (exclusive)
            start_in = (start_minutes >= s_start) & (start_minutes < s_end)
            end_in = (end_minutes >= s_start) & (end_minutes < s_end)
        else:
            # Cross-midnight session (Sydney: 21:00 - 06:00)
            # Session is active from s_start to midnight OR from midnight to s_end
            start_in = (start_minutes >= s_start) | (start_minutes < s_end)
            end_in = (end_minutes >= s_start) | (end_minutes < s_end)

        # Candle overlaps with session if start OR end is within session
        data[key] = start_in | end_in

    return pd.DataFrame(data, index=timestamps_utc)

# ============================================================
# 4. TICKS TO CANDLES
# ============================================================

def ticks_to_candles(
    ticks: np.ndarray,
    timeframe: str = "1H",
    price_type: str = "bid",
    session_filter: Optional[Union[str, List[str]]] = None,
    target_tz: Optional[str] = "EET",
) -> pd.DataFrame:
    """
    Converts tick data to OHLCV candles with binary session columns.

    Pipeline:
        1. Ticks → 1-minute candles (always)
        2. 1-minute → target timeframe
        3. Session detection on final timeframe (binary columns)
        4. Session filter (optional)
        5. Timezone conversion (default EET for MT4/MT5 compatibility)

    Args:
        ticks: Numpy structured array from download_ticks or load_ticks
        timeframe: Target timeframe ("1m", "5m", "15m", "1H", "4H", etc.)
        price_type: "bid", "ask", or "mid"
        session_filter: Optional session(s) to filter:
                       - Single: "tokyo"
                       - Multiple: ["tokyo", "london"]
                       - None: no filtering (default)
        target_tz: Output timezone (default "EET" for MT4/MT5)

    Returns:
        DataFrame with columns:
        ['datetime', 'open', 'high', 'low', 'close', 'volume',
         'sydney', 'tokyo', 'london', 'newyork']

    Example:
        >>> candles = ticks_to_candles(ticks, "1H", "bid", target_tz="Tehran")
        >>> candles.head()
                          datetime     open     high      low    close  volume  sydney  tokyo  london  newyork
        0 2024-01-01 03:30:00+03:30  2045.50  2046.20  2045.10  2045.80  1234.5    True   True   False    False
    """
    # Define output columns
    output_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume'] + SESSION_COLUMNS

    if len(ticks) == 0:
        return pd.DataFrame(columns=output_columns)

    # Normalize timeframe
    tf_key = timeframe.upper() if timeframe[-1].lower() in ['h', 'm'] else timeframe
    tf_key = tf_key.replace('H', 'H').replace('m', 'm')

    # Map to pandas offset
    tf_map = {
        "1M": "1min", "2M": "2min", "3M": "3min", "5M": "5min",
        "10M": "10min", "15M": "15min", "30M": "30min",
        "1H": "1h", "2H": "2h", "4H": "4h",
    }

    # Handle lowercase input
    tf_upper = timeframe.upper()
    if tf_upper.endswith('M') and tf_upper[:-1].isdigit():
        pandas_offset = tf_upper[:-1] + 'min'
    elif tf_upper.endswith('H') and tf_upper[:-1].isdigit():
        pandas_offset = tf_upper[:-1] + 'h'
    else:
        pandas_offset = tf_map.get(tf_upper, "1h")

    # Duration for session detection
    duration_map = {
        "1min": timedelta(minutes=1), "2min": timedelta(minutes=2),
        "3min": timedelta(minutes=3), "5min": timedelta(minutes=5),
        "10min": timedelta(minutes=10), "15min": timedelta(minutes=15),
        "30min": timedelta(minutes=30),
        "1h": timedelta(hours=1), "2h": timedelta(hours=2), "4h": timedelta(hours=4),
    }
    target_duration = duration_map.get(pandas_offset, timedelta(hours=1))

    # Build DataFrame from ticks
    df = pd.DataFrame({
        'ask': ticks['ask'],
        'bid': ticks['bid'],
        'ask_volume': ticks['ask_volume'].astype(np.float64),
        'bid_volume': ticks['bid_volume'].astype(np.float64),
    })
    df.index = pd.to_datetime(ticks['timestamp_us'], unit='us', utc=True)

    # Compute price and volume based on price_type
    if price_type == "mid":
        df['price'] = (df['ask'] + df['bid']) * 0.5
        df['volume'] = (df['ask_volume'] + df['bid_volume']) * 0.5
    elif price_type == "ask":
        df['price'] = df['ask']
        df['volume'] = df['ask_volume']
    else:  # bid (default)
        df['price'] = df['bid']
        df['volume'] = df['bid_volume']

    df = df[['price', 'volume']]

    # Resample to 1-minute first
    ohlc = df['price'].resample('1min').ohlc()
    vol = df['volume'].resample('1min').sum()
    candles_1m = pd.concat([ohlc, vol], axis=1)
    candles_1m.columns = ['open', 'high', 'low', 'close', 'volume']
    candles_1m.dropna(inplace=True)

    if candles_1m.empty:
        return pd.DataFrame(columns=output_columns)

    # Resample to target timeframe
    if pandas_offset == "1min":
        candles = candles_1m
    else:
        resampler = candles_1m.resample(pandas_offset)
        candles = resampler.agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        })
        candles.dropna(subset=['open'], inplace=True)

    if candles.empty:
        return pd.DataFrame(columns=output_columns)

    # Session detection - returns DataFrame with binary columns
    session_df = _detect_sessions_vectorized(candles.index, target_duration)

    # Add session columns to candles
    for col in SESSION_COLUMNS:
        candles[col] = session_df[col].values

    # Session filter
    if session_filter:
        # Normalize to list
        if isinstance(session_filter, str):
            session_filter = [session_filter]

        # Validate session names
        valid_sessions = [s for s in session_filter if s in SESSION_COLUMNS]
        if not valid_sessions:
            logger.warning(f"Invalid session filter: {session_filter}. Valid options: {SESSION_COLUMNS}")
            return pd.DataFrame(columns=output_columns)

        # Filter: keep rows where ANY of the specified sessions is True
        mask = candles[valid_sessions].any(axis=1)
        candles = candles[mask]

        if candles.empty:
            return pd.DataFrame(columns=output_columns)

    # Reset index
    candles = candles.reset_index()
    candles.rename(columns={'index': 'datetime'}, inplace=True)
    if candles.columns[0] != 'datetime':
        candles.rename(columns={candles.columns[0]: 'datetime'}, inplace=True)

    # Timezone conversion
    if target_tz:
        tz_info = get_timezone(target_tz)
        dt_col = candles['datetime']
        if dt_col.dt.tz is None:
            dt_col = dt_col.dt.tz_localize('UTC')
        candles['datetime'] = dt_col.dt.tz_convert(tz_info)

    # Column ordering
    candles = candles[output_columns]

    return candles

# ============================================================
# 5. SAVE CANDLES
# ============================================================

def save_candles(
    candles: pd.DataFrame,
    filepath: str,
) -> None:
    """
    Saves candle data to Parquet with Zstd compression.

    Session columns (sydney, tokyo, london, newyork) are saved as boolean.

    Args:
        candles: DataFrame with candle data
        filepath: Output file path (.parquet)

    Example:
        >>> save_candles(candles_df, "XAUUSD_240101_240131_1H.parquet")
    """
    if candles.empty:
        logger.warning("No candles to save")
        return

    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Prepare for parquet (convert timezone-aware datetime)
    df = candles.copy()
    if 'datetime' in df.columns:
        # Store as UTC for consistent storage
        if df['datetime'].dt.tz is not None:
            df['datetime'] = df['datetime'].dt.tz_convert('UTC')

    # Ensure session columns are boolean
    for col in SESSION_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Write with compression
    df.to_parquet(
        filepath,
        engine='pyarrow',
        compression='zstd',
        index=False,
    )

    logger.info(f"Saved {len(candles):,} candles to {filepath}")

# ============================================================
# 6. LOAD CANDLES
# ============================================================

def load_candles(
    filepath: str,
    target_tz: Optional[str] = "EET",
) -> pd.DataFrame:
    """
    Loads candle data from Parquet file.

    Session columns are loaded as boolean type.

    Args:
        filepath: Path to parquet file
        target_tz: Target timezone for datetime conversion (default EET)

    Returns:
        DataFrame with candle data including boolean session columns

    Example:
        >>> candles = load_candles("XAUUSD_240101_240131_1H.parquet")
        >>> candles['tokyo'].value_counts()
        True     150
        False    100
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_parquet(filepath, engine='pyarrow')

    # Convert datetime timezone
    if 'datetime' in df.columns and target_tz:
        tz_info = get_timezone(target_tz)
        dt_col = df['datetime']
        if dt_col.dt.tz is None:
            dt_col = pd.to_datetime(dt_col, utc=True)
        df['datetime'] = dt_col.dt.tz_convert(tz_info)

    # Ensure session columns are boolean
    for col in SESSION_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    logger.info(f"Loaded {len(df):,} candles from {filepath}")
    return df

# ============================================================
# UTILITY: CANCEL DOWNLOAD
# ============================================================

def cancel_download() -> None:
    """Sets the cancel flag to stop ongoing downloads."""
    global cancel_flag
    cancel_flag = True
    logger.info("Download cancellation requested")

def reset_cancel_flag() -> None:
    """Resets the cancel flag."""
    global cancel_flag
    cancel_flag = False

# ============================================================
# UTILITY: FILE INFO
# ============================================================

def get_parquet_info(filepath: str) -> Dict[str, Any]:
    """
    Returns metadata and statistics about a parquet file.

    Args:
        filepath: Path to parquet file

    Returns:
        Dict with file info (size, rows, columns, metadata, etc.)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    file_size = os.path.getsize(filepath)
    parquet_file = pq.ParquetFile(filepath)
    metadata = parquet_file.metadata
    schema_metadata = parquet_file.schema_arrow.metadata or {}

    info = {
        'filepath': filepath,
        'file_size_bytes': file_size,
        'file_size_mb': round(file_size / (1024 * 1024), 2),
        'num_rows': metadata.num_rows,
        'num_columns': metadata.num_columns,
        'num_row_groups': metadata.num_row_groups,
        'created_by': metadata.created_by,
        'schema': [f.name for f in parquet_file.schema_arrow],
        'tdr_version': schema_metadata.get(b'tdr_version', b'unknown').decode(),
        'delta_encoded': schema_metadata.get(b'delta_encoded', b'unknown').decode(),
        'tick_count': schema_metadata.get(b'tick_count', b'unknown').decode(),
    }

    return info

# ============================================================
# UTILITY: SESSION HELPERS
# ============================================================

def get_active_sessions(candle_row: pd.Series) -> List[str]:
    """
    Returns list of active sessions for a candle row.

    Args:
        candle_row: Single row from candles DataFrame

    Returns:
        List of active session names

    Example:
        >>> get_active_sessions(candles.iloc[0])
        ['sydney', 'tokyo']
    """
    return [s for s in SESSION_COLUMNS if candle_row.get(s, False)]

def filter_by_sessions(
    candles: pd.DataFrame,
    sessions: Union[str, List[str]],
    require_all: bool = False,
) -> pd.DataFrame:
    """
    Filters candles by session(s).

    Args:
        candles: DataFrame with session columns
        sessions: Session name(s) to filter by
        require_all: If True, all specified sessions must be active.
                    If False (default), any of the sessions can be active.

    Returns:
        Filtered DataFrame

    Example:
        >>> # Get candles during Tokyo OR London
        >>> tokyo_london = filter_by_sessions(candles, ["tokyo", "london"])

        >>> # Get candles during Tokyo AND London overlap
        >>> overlap = filter_by_sessions(candles, ["tokyo", "london"], require_all=True)
    """
    if isinstance(sessions, str):
        sessions = [sessions]

    # Validate session names
    valid_sessions = [s for s in sessions if s in SESSION_COLUMNS]
    if not valid_sessions:
        logger.warning(f"No valid sessions in: {sessions}")
        return candles.iloc[0:0]  # Empty DataFrame with same structure

    if require_all:
        # All sessions must be True
        mask = candles[valid_sessions].all(axis=1)
    else:
        # Any session can be True
        mask = candles[valid_sessions].any(axis=1)

    return candles[mask].copy()

# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Example: Download, save, load, convert

    print("=" * 60)
    print("TDR_V5 - Tick Data Repository v5.0.1")
    print("With Binary Session Columns")
    print("=" * 60)

    # Example parameters
    symbol = "XAUUSD"
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)  # 2 days for demo

    # Generate filenames
    tick_file = generate_tick_filename(symbol, start, end, "data")
    candle_file = generate_candle_filename(symbol, start, end, "1H", "data")

    print(f"\nTick filename:   {tick_file}")
    print(f"Candle filename: {candle_file}")

    # Download ticks
    print(f"\n📥 Downloading {symbol} ticks from {start.date()} to {end.date()}...")

    def progress(current, total, msg):
        print(f"  [{current}/{total}] {msg}")

    ticks = download_ticks(
        symbol=symbol,
        start=start,
        end=end,
        progress_callback=progress,
        auto_save_path=tick_file,
    )

    print(f"\n📊 Downloaded {len(ticks):,} ticks")

    if len(ticks) > 0:
        # Show file info
        print(f"\n📁 File info:")
        info = get_parquet_info(tick_file)
        print(f"   Size: {info['file_size_mb']} MB")
        print(f"   Rows: {info['num_rows']:,}")
        print(f"   Delta encoded: {info['delta_encoded']}")

        # Load and verify
        print(f"\n🔄 Loading ticks...")
        loaded_ticks = load_ticks(tick_file)
        print(f"   Loaded: {len(loaded_ticks):,} ticks")

        # Convert to candles
        print(f"\n📈 Converting to 1H candles...")
        candles = ticks_to_candles(loaded_ticks, "1H", "bid", target_tz="EET")
        print(f"   Generated: {len(candles):,} candles")

        if not candles.empty:
            print(f"\n   Output columns: {list(candles.columns)}")
            print(f"\n   First candle:")
            print(candles.head(1).to_string(index=False))

            # Show session statistics
            print(f"\n📊 Session statistics:")
            for session in SESSION_COLUMNS:
                count = candles[session].sum()
                pct = (count / len(candles)) * 100
                print(f"   {session:10s}: {count:4d} candles ({pct:.1f}%)")

            # Show overlap example
            print(f"\n🔀 Session overlaps:")
            tokyo_london = candles[candles['tokyo'] & candles['london']]
            print(f"   Tokyo + London overlap: {len(tokyo_london)} candles")

            london_ny = candles[candles['london'] & candles['newyork']]
            print(f"   London + NewYork overlap: {len(london_ny)} candles")

            # Save candles
            save_candles(candles, candle_file)

            # Load candles
            loaded_candles = load_candles(candle_file, target_tz="Tehran")
            print(f"\n   Loaded with Tehran timezone:")
            print(loaded_candles.head(1).to_string(index=False))

            # Filter example
            print(f"\n🎯 Filter example - Tokyo session only:")
            tokyo_candles = filter_by_sessions(loaded_candles, "tokyo")
            print(f"   Tokyo candles: {len(tokyo_candles)}")

    print("\n✅ Demo complete!")
