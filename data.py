from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import os

NUM_COUNTIES = 3108
NUM_STACK_TYPES = 4
AGE_COLUMNS = [f"Age{i}" for i in range(100)]

# CSV file paths for fallback
CSV_PATHS = {
    'CR_FUNCTIONS': 'COBRA/input files/default data/default_CR_functions.csv',
    'DICT': 'COBRA/input files/data dictionary/SOURCEINDX to FIPS crosswalk.csv',
    'TIERS': 'COBRA/input files/data dictionary/EmissionsTier Definitions.csv',
    'STACK_HEIGHTS': 'COBRA/input files/data dictionary/typeindx - stack heights.csv',
    'POPULATION_2016': 'COBRA/input files/default data/default_2016_population_data.csv',
    'POPULATION_2023': 'COBRA/input files/default data/default_2023_population_data.csv',
    'POPULATION_2028': 'COBRA/input files/default data/default_2028_population_data.csv',
    'INCIDENCE_2016': 'COBRA/input files/default data/default_2016_incidence_data.csv',
    'INCIDENCE_2023': 'COBRA/input files/default data/default_2023_incidence_data.csv',
    'INCIDENCE_2028': 'COBRA/input files/default data/default_2028_incidence_data.csv',
    'VALUATION_2016': 'COBRA/input files/default data/default_2016_valuation_data.csv',
    'VALUATION_2023': 'COBRA/input files/default data/default_2023_valuation_data.csv',
    'VALUATION_2028': 'COBRA/input files/default data/default_2028_valuation_data.csv',
    'EMISSIONS_2016': 'COBRA/input files/emissions/Emissions_2016.csv',
    'EMISSIONS_2023': 'COBRA/input files/emissions/Emissions_2023.csv',
    'EMISSIONS_2028': 'COBRA/input files/emissions/Emissions_2028.csv',
}


@dataclass
class EmissionsRecord:
    """Single emissions record from inventory."""
    id: int
    typeindx: int
    sourceindx: int
    stid: int
    cyid: int
    tier1: int
    tier2: int
    tier3: int
    nox: float
    so2: float
    nh3: float
    soa: float
    pm25: float
    voc: float


@dataclass
class CRFunction:
    """Concentration-Response function parameters."""
    id: int
    function_id: int
    endpoint: str
    pooling_weight: float
    seasonal_metric: str
    study_author: str
    study_year: int
    start_age: int
    end_age: int
    function: str
    beta: float
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    incidence_endpoint: str = ""


@dataclass
class ValuationFunction:
    """Economic valuation function parameters."""
    id: int
    cr_function_id: int
    endpoint: str
    pooling_weight: float
    seasonal_metric: str
    start_age: int
    end_age: int
    function: str
    beta: float
    value: float
    apply_discount: str
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    incidence_endpoint: str = ""


class CobraData:
    """
    Main data access class for COBRA calculations.

    Loads all reference data from SQLite database and provides
    efficient access methods for the computation pipeline.
    """

    def __init__(self, db_path: str = "data/cobra.db"):
        self.engine = create_engine(f"sqlite+pysqlite:///{db_path}")

        self._emissions_base: Optional[pd.DataFrame] = None
        self._emissions_control: Optional[pd.DataFrame] = None
        self._sr_matrices: Optional[dict] = None
        self._population: Optional[pd.DataFrame] = None
        self._incidence: Optional[pd.DataFrame] = None
        self._cr_functions: Optional[pd.DataFrame] = None
        self._valuation_functions: Optional[pd.DataFrame] = None
        self._voc2soa: Optional[dict] = None
        self._tiers: Optional[pd.DataFrame] = None
        self._dict: Optional[pd.DataFrame] = None

        self._pop_by_dest: Optional[dict] = None
        self._incidence_by_dest_endpoint: Optional[dict] = None

    def _read_sql_with_csv_fallback(self, query: str, csv_key: Optional[str] = None) -> pd.DataFrame:
        """
        Execute SQL query and fallback to CSV if result is empty.

        Args:
            query: SQL query to execute
            csv_key: Key in CSV_PATHS dict for fallback file

        Returns:
            DataFrame from SQL or CSV
        """
        try:
            df = pd.read_sql(query, self.engine)

            # If empty and we have a CSV fallback, try reading from CSV
            if df.empty and csv_key and csv_key in CSV_PATHS:
                csv_path = CSV_PATHS[csv_key]
                if os.path.exists(csv_path):
                    print(f"SQL returned empty data, falling back to CSV: {csv_path}")
                    df = pd.read_csv(csv_path)
                    df = self._normalize_csv_columns(df, csv_key)

            return df
        except Exception as e:
            # If SQL fails completely and we have a CSV fallback, try reading from CSV
            if csv_key and csv_key in CSV_PATHS:
                csv_path = CSV_PATHS[csv_key]
                if os.path.exists(csv_path):
                    print(f"SQL query failed ({e}), falling back to CSV: {csv_path}")
                    df = pd.read_csv(csv_path)
                    df = self._normalize_csv_columns(df, csv_key)
                    return df
            raise

    def _normalize_csv_columns(self, df: pd.DataFrame, csv_key: str) -> pd.DataFrame:
        """
        Normalize CSV column names to match database schema.

        Args:
            df: DataFrame loaded from CSV
            csv_key: Key identifying the CSV type

        Returns:
            DataFrame with normalized column names
        """
        # CR Functions: CSV "Adjusted" column contains the actual beta coefficient.
        # "Parameter_1_Beta" is the standard error, NOT the beta itself.
        if csv_key == 'CR_FUNCTIONS':
            if 'Adjusted' in df.columns and 'Beta' not in df.columns:
                df['Beta'] = df['Adjusted']

        # Valuation Functions: same — use "Adjusted" for the true beta
        if csv_key and csv_key.startswith('VALUATION_'):
            if 'Adjusted' in df.columns and 'Beta' not in df.columns:
                df['Beta'] = df['Adjusted']

        return df

    def load_emissions_base(self, year: Optional[int] = None) -> pd.DataFrame:
        """
        Load base emissions inventory.

        Args:
            year: Optional year for CSV fallback (2016, 2023, or 2028)
        """
        if self._emissions_base is None:
            query = "SELECT * FROM SYS_Emissions_Base"
            # Use year for CSV fallback, default to 2023
            csv_key = f'EMISSIONS_{year}' if year else 'EMISSIONS_2023'
            self._emissions_base = self._read_sql_with_csv_fallback(query, csv_key)
        return self._emissions_base

    def load_emissions_control(self, year: Optional[int] = None) -> pd.DataFrame:
        """
        Load control (modified) emissions.

        Args:
            year: Optional year for CSV fallback (2016, 2023, or 2028)
        """
        if self._emissions_control is None:
            query = "SELECT * FROM SYS_Emissions_Control"
            # Use year for CSV fallback, default to 2023
            csv_key = f'EMISSIONS_{year}' if year else 'EMISSIONS_2023'
            self._emissions_control = self._read_sql_with_csv_fallback(query, csv_key)
        return self._emissions_control

    def load_emissions_inventory(self, inventory_id: int = 1, year: Optional[int] = None) -> pd.DataFrame:
        """
        Load emissions from inventory table for a specific year/version.

        Args:
            inventory_id: ID in SYS_Emissions_Inventory_Description
            year: Optional year for CSV fallback (2016, 2023, or 2028)
        """
        query = f"SELECT * FROM SYS_Emissions_Inventory WHERE ID = {inventory_id}"
        # If year is provided, use it for CSV fallback
        if year:
            csv_key = f'EMISSIONS_{year}'
        else:
            csv_key = 'EMISSIONS_2023'  # Default to 2023
        return self._read_sql_with_csv_fallback(query, csv_key)

    def summarize_emissions(self, emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize emissions by typeindx and sourceindx.

        Groups emissions and sums pollutants, matching the C# SummarizeEmissionsbyType() method.
        SOA is recomputed from VOC using per-tier conversion factors before aggregation.
        """
        pollutant_cols = ['NOx', 'SO2', 'NH3', 'SOA', 'PM25', 'VOC']

        # Recompute SOA from VOC per row (matching C# SummarizeEmissionsbyType)
        if all(c in emissions.columns for c in ['TIER1', 'TIER2', 'TIER3']):
            emissions = emissions.copy()
            voc2soa = self.load_voc2soa()
            tier_keys = (emissions['TIER1'].astype(int).astype(str) + '|' +
                         emissions['TIER2'].astype(int).astype(str) + '|' +
                         emissions['TIER3'].astype(int).astype(str))
            factors = tier_keys.map(voc2soa).fillna(0.0)
            emissions['SOA'] = emissions['VOC'] * factors

        summarized = emissions.groupby(['typeindx', 'sourceindx'], as_index=False)[pollutant_cols].sum()
        return summarized

    def modify_emissions(self, raw_emissions: pd.DataFrame, emissions: dict[str, float],
                         tier_ids: list, sourceindxs: list[int]) -> pd.DataFrame:
        """
        Modify emissions using ratio-based scaling, matching the COBRA API
        (C# UpdateEmissionsWithCriteria).

        The payload values represent the NEW absolute total for the matching
        sector/counties.  Each matching row is scaled by
        ``ratio = payload / current_sum`` so that the new aggregate equals
        the payload.  When the current sum is zero the payload is spread
        evenly across matching rows.  Values are clamped to >= 0 (COBRA mode).

        After scaling, SOA is recomputed from VOC using the per-tier VOC→SOA
        factor (matching C# ComputeSOAfromVOC).

        :param raw_emissions: The *unsummarized* emissions table (with tier
            columns) — typically ``load_emissions_base()``.
        :param emissions: Target totals for NOx, SO2, NH3, SOA, PM25, VOC.
        :param tier_ids: ``[TIER1, TIER2, TIER3]`` identifying the sector.
        :param sourceindxs: Counties affected (by sourceindx).
        :return: A modified copy of *raw_emissions*.
        """
        pollutant_cols = ['NOx', 'SO2', 'NH3', 'SOA', 'PM25', 'VOC']

        modified = raw_emissions.copy()

        # Build mask matching the C# criteria: tier AND location
        mask = (
            (modified['TIER1'] == int(tier_ids[0])) &
            (modified['TIER2'] == int(tier_ids[1])) &
            (modified['TIER3'] == int(tier_ids[2])) &
            (modified['sourceindx'].isin(sourceindxs))
        )

        matching_idx = modified.index[mask]
        rowcount = len(matching_idx)

        if rowcount > 0:
            # Current totals across matching rows
            current_sums = modified.loc[matching_idx, pollutant_cols].sum()

            # Per-pollutant ratio = payload / current_sum
            for col in pollutant_cols:
                current = current_sums[col]
                payload = emissions.get(col, 0)
                if current != 0:
                    ratio = payload / current
                    modified.loc[matching_idx, col] = (
                        modified.loc[matching_idx, col] * ratio
                    ).clip(lower=0)
                else:
                    # Spread evenly when no current emissions
                    modified.loc[matching_idx, col] = max(payload / rowcount, 0)
        else:
            # No matching rows — create new rows for every sourceindx × typeindx
            new_rows = []
            type_indices = [1, 2, 3, 4]
            numberofrows = len(sourceindxs) * len(type_indices)
            for sidx in sourceindxs:
                for ti in type_indices:
                    new_row = {c: 0.0 for c in modified.columns}
                    new_row['typeindx'] = ti
                    new_row['sourceindx'] = sidx
                    new_row['TIER1'] = int(tier_ids[0])
                    new_row['TIER2'] = int(tier_ids[1])
                    new_row['TIER3'] = int(tier_ids[2])
                    for col in pollutant_cols:
                        new_row[col] = max(emissions.get(col, 0) / numberofrows, 0)
                    new_rows.append(new_row)
            if new_rows:
                modified = pd.concat(
                    [modified, pd.DataFrame(new_rows)], ignore_index=True
                )
            matching_idx = modified.index[mask] if rowcount > 0 else modified.tail(len(new_rows)).index

        # Recompute SOA from VOC for all matching rows (C# ComputeSOAfromVOC)
        voc2soa = self.load_voc2soa()
        tier_key = f"{int(tier_ids[0])}|{int(tier_ids[1])}|{int(tier_ids[2])}"
        factor = voc2soa.get(tier_key, 0.0)
        # Recompute for all rows with these tiers (matching rows + any new rows)
        tier_mask = (
            (modified['TIER1'] == int(tier_ids[0])) &
            (modified['TIER2'] == int(tier_ids[1])) &
            (modified['TIER3'] == int(tier_ids[2])) &
            (modified['sourceindx'].isin(sourceindxs))
        )
        modified.loc[tier_mask, 'SOA'] = modified.loc[tier_mask, 'VOC'] * factor

        return modified

    def get_stack_ratio(self, tier_ids):
        """
        Return the ratio of emissions at each stack height for an emissions source identified by tier IDs.
        :param tier_ids:
        :return: A tuple representing the ratio of emissions at 4 stack heights.
        """
        query = ("select typeindx, sum(NOx + SO2 + NH3 + SOA + PM25 + VOC) as total_emissions"
                 f" from `SYS_Emissions_Control` where TIER1={tier_ids[0]} and TIER2={tier_ids[1]} and TIER3={tier_ids[2]}"
                 " group by typeindx")

        try:
            emissions_sums = pd.read_sql(query, self.engine)

            # If database returns empty, fall back to CSV with pandas filtering
            if emissions_sums.empty:
                # Try loading from control emissions first, then base emissions as fallback
                emissions_df = self.load_emissions_control()

                # If still empty, try base emissions
                if emissions_df.empty:
                    emissions_df = self.load_emissions_base()

                # Filter by tier IDs and aggregate
                if not emissions_df.empty:
                    filtered = emissions_df[
                        (emissions_df['TIER1'] == tier_ids[0]) &
                        (emissions_df['TIER2'] == tier_ids[1]) &
                        (emissions_df['TIER3'] == tier_ids[2])
                    ]

                    if not filtered.empty:
                        # Calculate total emissions per typeindx
                        filtered['total_emissions'] = (
                            filtered['NOx'] + filtered['SO2'] + filtered['NH3'] +
                            filtered['SOA'] + filtered['PM25'] + filtered['VOC']
                        )
                        emissions_sums = filtered.groupby('typeindx', as_index=False)['total_emissions'].sum()
        except Exception as e:
            # If SQL fails completely, fall back to CSV
            print(f"SQL query failed, using CSV fallback: {e}")
            emissions_df = self.load_emissions_control()
            if emissions_df.empty:
                emissions_df = self.load_emissions_base()

            if not emissions_df.empty:
                filtered = emissions_df[
                    (emissions_df['TIER1'] == tier_ids[0]) &
                    (emissions_df['TIER2'] == tier_ids[1]) &
                    (emissions_df['TIER3'] == tier_ids[2])
                ]

                if not filtered.empty:
                    filtered['total_emissions'] = (
                        filtered['NOx'] + filtered['SO2'] + filtered['NH3'] +
                        filtered['SOA'] + filtered['PM25'] + filtered['VOC']
                    )
                    emissions_sums = filtered.groupby('typeindx', as_index=False)['total_emissions'].sum()
                else:
                    emissions_sums = pd.DataFrame()
            else:
                emissions_sums = pd.DataFrame()

        stack_emissions = []
        for i in range(4):
            stack_emission = emissions_sums[emissions_sums["typeindx"] == i+1]["total_emissions"]
            stack_emission = 0 if stack_emission.size == 0 else stack_emission.item()

            stack_emissions.append(stack_emission)

        total_emissions = sum(stack_emissions)

        return tuple(x / total_emissions for x in stack_emissions)

    def load_sr_matrices(self, cache_path: str = "data/sr_matrices.npz") -> dict:
        """
        Load Source-Receptor matrices from database.

        Uses an .npz file cache for fast loading on subsequent runs.
        First load reads from SQLite and saves the cache (~seconds to load from cache
        vs minutes from the database).

        Returns:
            Dictionary with keys:
            - 'dp': Direct PM2.5 transport [4 x 3108 x 3108]
            - 'NOx': NOx -> Nitrate [4 x 3108 x 3108]
            - 'SO4': SO2 -> Sulfate [4 x 3108 x 3108]
            - 'O3V': VOC -> Ozone [4 x 3108 x 3108]
            - 'O3N': NOx -> Ozone [4 x 3108 x 3108]
        """
        if self._sr_matrices is not None:
            return self._sr_matrices

        pollutant_keys = ['dp', 'NOx', 'SO4', 'O3V', 'O3N']
        col_map = {'dp': 'c_PM25', 'NOx': 'c_NO3', 'SO4': 'c_SO4', 'O3V': 'c_O3V', 'O3N': 'c_O3N'}

        # Try loading from .npz cache
        if os.path.exists(cache_path):
            print(f"Loading SR matrices from cache: {cache_path}")
            data = jnp.load(cache_path)
            self._sr_matrices = {}
            for key in pollutant_keys:
                self._sr_matrices[key] = [
                    jnp.array(data[f"{key}_{t}"]) for t in range(NUM_STACK_TYPES)
                ]
            return self._sr_matrices

        print("Loading SR matrices from database (first load, will cache for next time) ...")
        import sqlite3 as _sqlite3
        import time

        start = time.time()

        # Use raw sqlite3 for fastest bulk read (avoids pandas overhead for 38M rows)
        db_url = str(self.engine.url).replace("sqlite+pysqlite:///", "")
        conn = _sqlite3.connect(db_url)

        # Build matrices using vectorized numpy operations per stack type
        arrays = {key: [np.zeros((NUM_COUNTIES, NUM_COUNTIES), dtype=np.float32)
                        for _ in range(NUM_STACK_TYPES)] for key in pollutant_keys}

        for t in range(NUM_STACK_TYPES):
            typeindx = t + 1
            cursor = conn.execute(
                "SELECT sourceindx, destindx, c_PM25, c_NO3, c_SO4, c_O3V, c_O3N "
                "FROM SYS_Srmatrix WHERE typeindx = ?",
                (typeindx,)
            )

            rows = cursor.fetchall()
            if not rows:
                continue

            data_arr = np.array(rows, dtype=np.float64)
            s_idx = (data_arr[:, 0] - 1).astype(np.intp)
            d_idx = (data_arr[:, 1] - 1).astype(np.intp)

            # Bounds check
            valid = (s_idx >= 0) & (s_idx < NUM_COUNTIES) & (d_idx >= 0) & (d_idx < NUM_COUNTIES)
            s_idx = s_idx[valid]
            d_idx = d_idx[valid]

            for i, key in enumerate(pollutant_keys):
                col_data = np.nan_to_num(data_arr[valid, 2 + i], nan=0.0).astype(np.float32)
                arrays[key][t][d_idx, s_idx] = col_data

            elapsed = time.time() - start
            print(f"  typeindx {typeindx}/4 loaded ({len(rows):,} rows, {elapsed:.1f}s elapsed)")

        conn.close()

        # Save cache
        cache_data = {}
        for key in pollutant_keys:
            for t in range(NUM_STACK_TYPES):
                cache_data[f"{key}_{t}"] = arrays[key][t]
        np.savez(cache_path, **cache_data)
        print(f"  Cached to {cache_path} ({os.path.getsize(cache_path) / 1e6:.0f} MB)")

        # Convert to JAX
        self._sr_matrices = {}
        for key in pollutant_keys:
            self._sr_matrices[key] = [jnp.array(arrays[key][t]) for t in range(NUM_STACK_TYPES)]

        total = time.time() - start
        print(f"  SR matrices loaded in {total:.1f}s")

        return self._sr_matrices

    def load_population(self, year: Optional[int] = None) -> pd.DataFrame:
        """
        Load population data by county and age.

        Args:
            year: Optional year filter (uses SYS_POP_INVENTORY if specified)

        Note: Data is ordered by DestinationID to align with pollution vectors
              from SR matrices (which use destindx = DestinationID).
        """
        if self._population is None:
            if year:
                query = f"SELECT * FROM SYS_POP_INVENTORY WHERE Year = {year} ORDER BY DestinationID"
                csv_key = f'POPULATION_{year}'
            else:
                query = "SELECT * FROM SYS_POP ORDER BY DestinationID"
                csv_key = 'POPULATION_2023'  # Default to 2023
            self._population = self._read_sql_with_csv_fallback(query, csv_key)
        return self._population

    def get_population_dict(self) -> dict:
        """
        Get population indexed by DestinationID for fast lookup.

        Returns:
            Dict mapping DestinationID -> JAX array of age populations [100]
        """
        if self._pop_by_dest is None:
            pop = self.load_population()
            self._pop_by_dest = {}
            for _, row in pop.iterrows():
                dest_id = int(row['DestinationID'])
                age_values = jnp.array([row[col] or 0.0 for col in AGE_COLUMNS])
                self._pop_by_dest[dest_id] = age_values
        return self._pop_by_dest

    def load_incidence(self, year: Optional[int] = None) -> pd.DataFrame:
        """
        Load baseline incidence rates by county, endpoint, and age.

        Args:
            year: Optional year filter

        Note: Data is ordered by DestinationID to align with pollution vectors
              from SR matrices (which use destindx = DestinationID).
        """
        if self._incidence is None:
            if year:
                query = f"SELECT * FROM SYS_Incidence_Inventory WHERE Year = {year} ORDER BY DestinationID"
                csv_key = f'INCIDENCE_{year}'
            else:
                query = "SELECT * FROM SYS_Incidence ORDER BY DestinationID"
                csv_key = 'INCIDENCE_2023'  # Default to 2023
            self._incidence = self._read_sql_with_csv_fallback(query, csv_key)
        return self._incidence

    def get_incidence_dict(self) -> dict:
        """
        Get incidence indexed by (DestinationID, Endpoint) for fast lookup.

        Returns:
            Dict mapping "DestinationID|Endpoint" -> JAX array of age incidences [100]
        """
        if self._incidence_by_dest_endpoint is None:
            inc = self.load_incidence()
            self._incidence_by_dest_endpoint = {}
            for _, row in inc.iterrows():
                dest_id = int(row['DestinationID'])
                endpoint = row['Endpoint']
                key = f"{dest_id}|{endpoint}"
                age_values = jnp.array([row[col] or 0.0 for col in AGE_COLUMNS])
                self._incidence_by_dest_endpoint[key] = age_values
        return self._incidence_by_dest_endpoint

    def load_cr_functions(self, inventory_id: Optional[int] = None) -> pd.DataFrame:
        """
        Load Concentration-Response functions.

        Args:
            inventory_id: If provided, load from SYS_CR_INVENTORY with this ID
        """
        if self._cr_functions is None:
            if inventory_id:
                query = f"SELECT * FROM SYS_CR_INVENTORY WHERE ID = {inventory_id}"
            else:
                query = "SELECT * FROM SYS_CR"
            self._cr_functions = self._read_sql_with_csv_fallback(query, 'CR_FUNCTIONS')
        return self._cr_functions

    def get_cr_functions_list(self) -> list[CRFunction]:
        """Get CR functions as list of dataclass objects."""
        df = self.load_cr_functions()
        functions = []
        for _, row in df.iterrows():
            func = CRFunction(
                id=int(row.get('ID', 0)),
                function_id=int(row.get('FunctionID', 0)),
                endpoint=row.get('Endpoint', ''),
                pooling_weight=float(row.get('PoolingWeight', 0) or 0),
                seasonal_metric=row.get('Seasonal_Metric', ''),
                study_author=row.get('Study_Author', ''),
                study_year=int(row.get('Study_Year', 0) or 0),
                start_age=int(row.get('Start_Age', 0) or 0),
                end_age=int(row.get('End_Age', 99) or 99),
                function=row.get('Function', ''),
                beta=float(row.get('Beta', 0) or 0),
                a=float(row['A']) if row.get('A') else None,
                b=float(row['B']) if row.get('B') else None,
                c=float(row['C']) if row.get('C') else None,
                incidence_endpoint=row.get('IncidenceEndpoint', '')
            )
            functions.append(func)
        return functions

    def load_valuation_functions(self, inventory_id: Optional[int] = None) -> pd.DataFrame:
        """
        Load economic valuation functions.

        Args:
            inventory_id: If provided, load from SYS_Valuation_INVENTORY with this ID
        """
        if self._valuation_functions is None:
            if inventory_id:
                query = f"SELECT * FROM SYS_Valuation_INVENTORY WHERE ID = {inventory_id}"
                # Try to map inventory_id to year, default to 2023
                csv_key = 'VALUATION_2023'
            else:
                query = "SELECT * FROM SYS_Valuation"
                csv_key = 'VALUATION_2023'
            self._valuation_functions = self._read_sql_with_csv_fallback(query, csv_key)
        return self._valuation_functions

    def get_valuation_functions_list(self) -> list[ValuationFunction]:
        """Get valuation functions as list of dataclass objects."""
        df = self.load_valuation_functions()
        functions = []
        for _, row in df.iterrows():
            func = ValuationFunction(
                id=int(row.get('ID', 0)),
                cr_function_id=int(row.get('CRFunctionID', 0) or 0),
                endpoint=row.get('Endpoint', ''),
                pooling_weight=float(row.get('PoolingWeight', 0) or 0),
                seasonal_metric=row.get('Seasonal_Metric', ''),
                start_age=int(row.get('Start_Age', 0) or 0),
                end_age=int(row.get('End_Age', 99) or 99),
                function=row.get('Function', ''),
                beta=float(row.get('Beta', 0) or 0),
                value=float(row.get('Value', 0) or 0),
                apply_discount=row.get('ApplyDiscount', 'NO'),
                a=float(row['A']) if row.get('A') else None,
                b=float(row['B']) if row.get('B') else None,
                c=float(row['C']) if row.get('C') else None,
                incidence_endpoint=row.get('IncidenceEndpoint', '')
            )
            functions.append(func)
        return functions

    def load_voc2soa(self) -> dict:
        """
        Load VOC to SOA conversion factors by tier.

        Returns:
            Dict mapping "TIER1|TIER2|TIER3" -> conversion factor
        """
        if self._voc2soa is None:
            query = "SELECT TIER1, TIER2, TIER3, FACTOR FROM SYS_voc2soa"
            df = self._read_sql_with_csv_fallback(query, None)  # No CSV fallback available
            self._voc2soa = {}
            for _, row in df.iterrows():
                key = f"{int(row['TIER1'])}|{int(row['TIER2'])}|{int(row['TIER3'])}"
                self._voc2soa[key] = float(row['FACTOR'] or 0)
        return self._voc2soa

    def compute_soa_from_voc(self, tier1: int, tier2: int, tier3: int, voc: float) -> float:
        """
        Convert VOC to SOA using tier-specific conversion factor.

        Args:
            tier1, tier2, tier3: Source category identifiers
            voc: VOC emissions (tons/year)
        """
        voc2soa = self.load_voc2soa()
        key = f"{tier1}|{tier2}|{tier3}"
        factor = voc2soa.get(key, 0.0)
        return voc * factor

    def load_tiers(self) -> pd.DataFrame:
        """Load tier name mappings."""
        if self._tiers is None:
            query = "SELECT * FROM SYS_Tiers"
            self._tiers = self._read_sql_with_csv_fallback(query, 'TIERS')
        return self._tiers

    def load_county_dict(self) -> pd.DataFrame:
        """Load county/state name mappings."""
        if self._dict is None:
            query = "SELECT * FROM SYS_Dict"
            self._dict = self._read_sql_with_csv_fallback(query, 'DICT')
        return self._dict

    def get_county_info(self, sourceindx: int) -> dict:
        """
        Get county information by source index.

        Args:
            sourceindx: County index (1-3108)
        """
        county_dict = self.load_county_dict()
        row = county_dict[county_dict['SOURCEINDX'] == sourceindx]
        if len(row) == 0:
            return {'FIPS': '', 'STATE': '', 'COUNTY': ''}
        row = row.iloc[0]
        return {
            'FIPS': row.get('FIPS', ''),
            'STATE': row.get('STNAME', ''),
            'COUNTY': row.get('CYNAME', '')
        }

    def load_analysis_years(self) -> pd.DataFrame:
        """Load analysis year configurations linking data versions."""
        query = "SELECT * FROM SYS_AnalysisYear"
        return self._read_sql_with_csv_fallback(query, None)  # No CSV fallback available

    def get_analysis_year_config(self, year_id: int) -> dict:
        """
        Get data version IDs for a specific analysis year.

        Args:
            year_id: Analysis year ID
        """
        years = self.load_analysis_years()
        row = years[years['ID'] == year_id]
        if len(row) == 0:
            return {}
        row = row.iloc[0]
        return {
            'PopulationID': int(row.get('PopulationID', 0) or 0),
            'IncidenceID': int(row.get('IncidenceID', 0) or 0),
            'CRID': int(row.get('CRID', 0) or 0),
            'ValueID': int(row.get('ValueID', 0) or 0),
            'EmissionsID': int(row.get('EmissionsID', 0) or 0)
        }

    def load_destinations(self) -> pd.DataFrame:
        """Load destination (county) air quality results."""
        query = "SELECT * FROM SYS_Destination"
        return self._read_sql_with_csv_fallback(query, None)  # No CSV fallback available

    def load_results(self) -> pd.DataFrame:
        """Load computed health impact results."""
        query = "SELECT * FROM SYS_Results"
        return self._read_sql_with_csv_fallback(query, None)  # No CSV fallback available


def get_cobra_data(db_path: str = "data/cobra.db") -> CobraData:
    """Create and return a CobraData instance."""
    return CobraData(db_path)
