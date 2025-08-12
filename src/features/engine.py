# # import pandas as pd
# # import numpy as np

# # def compute_features(ohlcv_data, ref_asset_ticker='SPY', vix_ticker='^VIX'):
# #     """
# #     Computes a set of features for the reference asset.
# #     Handles MultiIndex columns properly and avoids KeyErrors.
# #     """
# #     if ref_asset_ticker not in ohlcv_data or ohlcv_data[ref_asset_ticker].empty:
# #         raise ValueError(f"Reference asset '{ref_asset_ticker}' not found in OHLCV data.")

# #     # Extract OHLCV for the reference asset
# #     df = ohlcv_data[ref_asset_ticker].copy()

# #     # If columns are MultiIndex with second level being OHLCV fields, drop first level
# #     if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
# #         df.columns = df.columns.get_level_values(1)

# #     feature_series_list = []

# #     # --- Volatility Features ---
# #     vol_21d = df['Close'].pct_change().rolling(window=21).std() * np.sqrt(252)
# #     vol_21d.name = 'volatility_21d'
# #     feature_series_list.append(vol_21d)

# #     vol_63d = df['Close'].pct_change().rolling(window=63).std() * np.sqrt(252)
# #     vol_63d.name = 'volatility_63d'
# #     feature_series_list.append(vol_63d)

# #     # --- Momentum Features ---
# #     mom_21d = df['Close'].pct_change(periods=21)
# #     mom_21d.name = 'momentum_21d'
# #     feature_series_list.append(mom_21d)

# #     mom_63d = df['Close'].pct_change(periods=63)
# #     mom_63d.name = 'momentum_63d'
# #     feature_series_list.append(mom_63d)

# #     mom_252d = df['Close'].pct_change(periods=252)
# #     mom_252d.name = 'momentum_252d'
# #     feature_series_list.append(mom_252d)

# #     # --- Trend Features ---
# #     sma_21 = df['Close'].rolling(window=21).mean()
# #     sma_21.name = 'sma_21'
# #     feature_series_list.append(sma_21)

# #     sma_63 = df['Close'].rolling(window=63).mean()
# #     sma_63.name = 'sma_63'
# #     feature_series_list.append(sma_63)

# #     price_vs_sma63 = (df['Close'] / sma_63 - 1)
# #     price_vs_sma63.name = 'price_vs_sma63'
# #     feature_series_list.append(price_vs_sma63)

# #     # --- Assemble the final DataFrame ---
# #     final_df = df.copy()
# #     for series in feature_series_list:
# #         final_df[series.name] = series

# #     # --- Handle VIX data ---
# #     if vix_ticker in ohlcv_data:
# #         vix_df = ohlcv_data[vix_ticker].copy()
# #         if isinstance(vix_df.columns, pd.MultiIndex) and vix_df.columns.nlevels == 2:
# #             vix_df.columns = vix_df.columns.get_level_values(1)
# #         vix_series = vix_df['Close'].rename('vix_close')
# #         final_df['vix_close'] = vix_series
# #         final_df['vix_close'].ffill(inplace=True)

# #     # --- Drop NaNs from rolling calculations ---
# #     final_df.dropna(inplace=True)

# #     return final_df
# import logging
# import pandas as pd
# import numpy as np
# from typing import Union

# logger = logging.getLogger(__name__)

# def _find_field_series(df: pd.DataFrame, field: str) -> pd.Series:
#     """
#     Robustly find a single Series in df that corresponds to `field` (e.g. 'Close').
#     Tries:
#       - exact match on column (single-level)
#       - substring match (case-insensitive)
#       - MultiIndex tuple contains `field`
#       - any level in MultiIndex equals `field`
#     If multiple candidates found, picks the first and logs a warning.
#     Raises KeyError if nothing usable is found.
#     """
#     field_l = field.lower()

#     # Helper to normalize candidate to Series
#     def _to_series(candidate):
#         s = df[candidate]
#         # if selecting returns a DataFrame (due to duplicate labels), take first column
#         if isinstance(s, pd.DataFrame):
#             logger.warning("Multiple columns found for candidate %r; using the first column.", candidate)
#             s = s.iloc[:, 0]
#         # standardize name
#         s = s.rename(field)
#         return s

#     cols = df.columns

#     # If MultiIndex: check tuples for exact or substring
#     if isinstance(cols, pd.MultiIndex):
#         # 1) tuple contains exact match (case-insensitive)
#         tuple_matches = [col for col in cols if any(str(x).lower() == field_l for x in col)]
#         if tuple_matches:
#             if len(tuple_matches) > 1:
#                 logger.warning("Multiple MultiIndex columns match %r; using the first: %s", field, tuple_matches[0])
#             return _to_series(tuple_matches[0])

#         # 2) any level equals field (more generic)
#         for lvl in range(cols.nlevels):
#             level_vals = cols.get_level_values(lvl)
#             idxs = [i for i, v in enumerate(level_vals) if str(v).lower() == field_l]
#             if idxs:
#                 # find first full column where that level matches
#                 for col in cols:
#                     if str(col[lvl]).lower() == field_l:
#                         return _to_series(col)

#         # 3) substring match inside tuple elements
#         substr_matches = [col for col in cols if any(field_l in str(x).lower() for x in col)]
#         if substr_matches:
#             logger.warning("Using MultiIndex substring match for field %r -> %s", field, substr_matches[0])
#             return _to_series(substr_matches[0])

#     else:
#         # Single-level columns
#         # 1) exact
#         if field in cols:
#             return _to_series(field)
#         # 2) case-insensitive exact
#         exact_ci = [c for c in cols if str(c).lower() == field_l]
#         if exact_ci:
#             return _to_series(exact_ci[0])
#         # 3) substring matches
#         substr = [c for c in cols if field_l in str(c).lower()]
#         if substr:
#             logger.warning("Using substring match for field %r -> %s", field, substr[0])
#             return _to_series(substr[0])

#     # nothing found
#     raise KeyError(f"Could not find field '{field}' in DataFrame columns. Columns: {list(cols[:20])} (showing up to 20)")

# def compute_features(ohlcv_data: dict, ref_asset_ticker: str = 'SPY', vix_ticker: str = '^VIX') -> pd.DataFrame:
#     """
#     Robust feature computation:
#       - extracts `Close` robustly from a variety of column shapes
#       - computes features from that Close series
#       - flattens final columns to single-level strings to avoid join/merge MultiIndex errors
#     """
#     if ref_asset_ticker not in ohlcv_data or ohlcv_data[ref_asset_ticker] is None or ohlcv_data[ref_asset_ticker].empty:
#         raise ValueError(f"Reference asset '{ref_asset_ticker}' not found in OHLCV data.")

#     raw_df = ohlcv_data[ref_asset_ticker].copy()

#     # 1) Extract Close series robustly
#     close_s = _find_field_series(raw_df, 'Close')

#     # 2) Build a flattened base DataFrame (so downstream columns are single-level)
#     base_df = raw_df.copy()
#     if isinstance(base_df.columns, pd.MultiIndex):
#         base_df.columns = ['_'.join(map(str, col)).strip() for col in base_df.columns]
#     # if no 'Close' column exists after flattening, still keep original base_df but we compute using close_s
#     # Optionally, add a canonical 'Close' column for clarity (but avoid overwriting if it already exists)
#     if 'Close' not in base_df.columns:
#         base_df = base_df.assign(Close=close_s)

#     # 3) Compute features using the extracted close_s (index aligned)
#     feature_series_list = []

#     vol_21d = close_s.pct_change().rolling(window=21).std() * np.sqrt(252)
#     vol_21d.name = 'volatility_21d'
#     feature_series_list.append(vol_21d)

#     vol_63d = close_s.pct_change().rolling(window=63).std() * np.sqrt(252)
#     vol_63d.name = 'volatility_63d'
#     feature_series_list.append(vol_63d)

#     mom_21d = close_s.pct_change(periods=21)
#     mom_21d.name = 'momentum_21d'
#     feature_series_list.append(mom_21d)

#     mom_63d = close_s.pct_change(periods=63)
#     mom_63d.name = 'momentum_63d'
#     feature_series_list.append(mom_63d)

#     mom_252d = close_s.pct_change(periods=252)
#     mom_252d.name = 'momentum_252d'
#     feature_series_list.append(mom_252d)

#     sma_21 = close_s.rolling(window=21).mean()
#     sma_21.name = 'sma_21'
#     feature_series_list.append(sma_21)

#     sma_63 = close_s.rolling(window=63).mean()
#     sma_63.name = 'sma_63'
#     feature_series_list.append(sma_63)

#     price_vs_sma63 = (close_s / sma_63 - 1)
#     price_vs_sma63.name = 'price_vs_sma63'
#     feature_series_list.append(price_vs_sma63)

#     # 4) Assemble final_df (single-level columns)
#     final_df = base_df.copy()
#     for s in feature_series_list:
#         final_df[s.name] = s

#     # 5) Attach VIX close (robustly)
#     if vix_ticker in ohlcv_data and ohlcv_data[vix_ticker] is not None and not ohlcv_data[vix_ticker].empty:
#         vix_raw = ohlcv_data[vix_ticker].copy()
#         try:
#             vix_close = _find_field_series(vix_raw, 'Close').rename('vix_close')
#         except KeyError:
#             # fallback: try flattening vix_raw first then look for substring 'close'
#             if isinstance(vix_raw.columns, pd.MultiIndex):
#                 vix_raw.columns = ['_'.join(map(str, col)).strip() for col in vix_raw.columns]
#             # try substring match
#             close_cols = [c for c in vix_raw.columns if 'close' in str(c).lower()]
#             if not close_cols:
#                 logger.warning("Could not find Close in VIX data; skipping vix_close.")
#                 vix_close = None
#             else:
#                 vix_close = vix_raw[close_cols[0]].rename('vix_close')
#         if vix_close is not None:
#             final_df['vix_close'] = vix_close
#             final_df['vix_close'].ffill(inplace=True)

#     # 6) Drop rows with NaNs that came from rolling windows
#     final_df.dropna(inplace=True)

#     return final_df
"""
src/features/engine.py

Robust feature computation utilities for the backtesting-agent.

- Detects 'Close' across MultiIndex / flattened / substring column names
- Computes common features (volatility, momentum, SMA)
- Produces a final DataFrame with single-level columns (no MultiIndex)
- Safely handles VIX extraction and forward-fill without using inplace on a slice
"""

import logging
from typing import Dict, Union, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _find_field_series(df: pd.DataFrame, field: str) -> pd.Series:
    """
    Robustly find a single Series in df that corresponds to `field` (e.g. 'Close').

    Search strategy:
      - If single-level columns: exact match, case-insensitive exact, substring match.
      - If MultiIndex columns: tuple-level exact match, any level equals field, substring inside tuple elements.
      - If selecting yields multiple columns (duplicate labels), pick the first and log a warning.

    Returns:
      pd.Series with name set to `field`.

    Raises:
      KeyError if no candidate found.
    """
    field_l = field.lower()

    def _to_series(candidate):
        s = df[candidate]
        # If selection returned a DataFrame (duplicate column labels), take first column
        if isinstance(s, pd.DataFrame):
            logger.warning("Multiple columns found for candidate %r; using the first column.", candidate)
            s = s.iloc[:, 0]
        s = s.rename(field)
        return s

    cols = df.columns

    # --- MultiIndex handling ---
    if isinstance(cols, pd.MultiIndex):
        # 1) tuple contains exact match (case-insensitive)
        tuple_matches = [col for col in cols if any(str(x).lower() == field_l for x in col)]
        if tuple_matches:
            if len(tuple_matches) > 1:
                logger.warning("Multiple MultiIndex columns match %r; using the first: %s", field, tuple_matches[0])
            return _to_series(tuple_matches[0])

        # 2) any level equals field
        for lvl in range(cols.nlevels):
            level_vals = cols.get_level_values(lvl)
            idxs = [i for i, v in enumerate(level_vals) if str(v).lower() == field_l]
            if idxs:
                # find first full column where that level matches
                for col in cols:
                    if str(col[lvl]).lower() == field_l:
                        return _to_series(col)

        # 3) substring match inside tuple elements
        substr_matches = [col for col in cols if any(field_l in str(x).lower() for x in col)]
        if substr_matches:
            logger.warning("Using MultiIndex substring match for field %r -> %s", field, substr_matches[0])
            return _to_series(substr_matches[0])

    # --- Single-level columns ---
    else:
        # 1) exact
        if field in cols:
            return _to_series(field)
        # 2) case-insensitive exact
        exact_ci = [c for c in cols if str(c).lower() == field_l]
        if exact_ci:
            return _to_series(exact_ci[0])
        # 3) substring matches
        substr = [c for c in cols if field_l in str(c).lower()]
        if substr:
            logger.warning("Using substring match for field %r -> %s", field, substr[0])
            return _to_series(substr[0])

    # nothing found
    raise KeyError(f"Could not find field '{field}' in DataFrame columns. Columns sample: {list(cols[:20])}")


def compute_features(
    ohlcv_data: Dict[str, pd.DataFrame],
    ref_asset_ticker: str = 'SPY',
    vix_ticker: str = '^VIX'
) -> pd.DataFrame:
    """
    Compute features for `ref_asset_ticker` using OHLCV in ohlcv_data.

    Returns a DataFrame with:
      - original (flattened) OHLCV columns when available
      - new feature columns: volatility_21d, volatility_63d, momentum_21d, momentum_63d,
        momentum_252d, sma_21, sma_63, price_vs_sma63, vix_close (if available)

    Notes:
      - Uses a robust Close-series extractor so it works with MultiIndex, flattened names, etc.
      - Final DataFrame columns are single-level strings.
    """
    if ref_asset_ticker not in ohlcv_data or ohlcv_data[ref_asset_ticker] is None or ohlcv_data[ref_asset_ticker].empty:
        raise ValueError(f"Reference asset '{ref_asset_ticker}' not found in OHLCV data.")

    raw_df = ohlcv_data[ref_asset_ticker].copy()

    # 1) Extract Close series robustly (this will be used for all calculations)
    close_s = _find_field_series(raw_df, 'Close')

    # 2) Build a flattened base DataFrame so downstream columns are single-level strings
    base_df = raw_df.copy()
    if isinstance(base_df.columns, pd.MultiIndex):
        base_df.columns = ['_'.join(map(str, col)).strip() for col in base_df.columns]

    # If 'Close' isn't present after flattening, create a canonical 'Close' column using close_s
    if 'Close' not in base_df.columns:
        base_df = base_df.assign(Close=close_s)

    # 3) Compute features from the extracted close series
    feature_series_list = []

    vol_21d = close_s.pct_change().rolling(window=21).std() * np.sqrt(252)
    vol_21d.name = 'volatility_21d'
    feature_series_list.append(vol_21d)

    vol_63d = close_s.pct_change().rolling(window=63).std() * np.sqrt(252)
    vol_63d.name = 'volatility_63d'
    feature_series_list.append(vol_63d)

    mom_21d = close_s.pct_change(periods=21)
    mom_21d.name = 'momentum_21d'
    feature_series_list.append(mom_21d)

    mom_63d = close_s.pct_change(periods=63)
    mom_63d.name = 'momentum_63d'
    feature_series_list.append(mom_63d)

    mom_252d = close_s.pct_change(periods=252)
    mom_252d.name = 'momentum_252d'
    feature_series_list.append(mom_252d)

    sma_21 = close_s.rolling(window=21).mean()
    sma_21.name = 'sma_21'
    feature_series_list.append(sma_21)

    sma_63 = close_s.rolling(window=63).mean()
    sma_63.name = 'sma_63'
    feature_series_list.append(sma_63)

    price_vs_sma63 = (close_s / sma_63 - 1)
    price_vs_sma63.name = 'price_vs_sma63'
    feature_series_list.append(price_vs_sma63)

    # 4) Assemble final_df and attach feature columns
    final_df = base_df.copy()
    for s in feature_series_list:
        final_df[s.name] = s

    # 5) Attach VIX close (robustly)
    if vix_ticker in ohlcv_data and ohlcv_data[vix_ticker] is not None and not ohlcv_data[vix_ticker].empty:
        vix_raw = ohlcv_data[vix_ticker].copy()
        try:
            vix_close = _find_field_series(vix_raw, 'Close').rename('vix_close')
        except KeyError:
            # fallback: flatten and try substring 'close'
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = ['_'.join(map(str, col)).strip() for col in vix_raw.columns]
            close_cols = [c for c in vix_raw.columns if 'close' in str(c).lower()]
            if not close_cols:
                logger.warning("Could not find Close in VIX data; skipping vix_close.")
                vix_close = None
            else:
                vix_close = vix_raw[close_cols[0]].rename('vix_close')

        if vix_close is not None:
            # avoid chained-assignment/inplace warning by assigning result back
            final_df['vix_close'] = vix_close
            final_df['vix_close'] = final_df['vix_close'].ffill()

    # 6) Drop rows with NaNs that come from rolling windows
    final_df = final_df.dropna()

    return final_df
