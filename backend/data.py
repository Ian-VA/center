"""COBRA reference data access.

Loads emissions inventory, source-receptor matrices, population, incidence, CR
functions, valuation functions, and VOC→SOA factors from the slim SQLite DB,
falling back to the bundled COBRA/ CSVs when a SQL query is empty or fails.
Kept minimal: only what the live `Compute` pipeline and `prepare_emissions` use.
"""
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import pandas as pd
from sqlalchemy import create_engine

NUM_COUNTIES = 3108
NUM_STACK_TYPES = 4

ROOT = Path(__file__).parent

CSV_PATHS = {
    'CR_FUNCTIONS': ROOT / 'COBRA' / 'input files' / 'default data' / 'default_CR_functions.csv',
    'DICT': ROOT / 'COBRA' / 'input files' / 'data dictionary' / 'SOURCEINDX to FIPS crosswalk.csv',
    'TIERS': ROOT / 'COBRA' / 'input files' / 'data dictionary' / 'EmissionsTier Definitions.csv',
    'STACK_HEIGHTS': ROOT / 'COBRA' / 'input files' / 'data dictionary' / 'typeindx - stack heights.csv',
    'POPULATION_2016': ROOT / 'COBRA' / 'input files' / 'default data' / 'default_2016_population_data.csv',
    'POPULATION_2023': ROOT / 'COBRA' / 'input files' / 'default data' / 'default_2023_population_data.csv',
    'POPULATION_2028': ROOT / 'COBRA' / 'input files' / 'default data' / 'default_2028_population_data.csv',
    'INCIDENCE_2016': ROOT / 'COBRA' / 'input files' / 'default data' / 'default_2016_incidence_data.csv',
    'INCIDENCE_2023': ROOT / 'COBRA' / 'input files' / 'default data' / 'default_2023_incidence_data.csv',
    'INCIDENCE_2028': ROOT / 'COBRA' / 'input files' / 'default data' / 'default_2028_incidence_data.csv',
    'VALUATION_2016': ROOT / 'COBRA' / 'input files' / 'default data' / 'default_2016_valuation_data.csv',
    'VALUATION_2023': ROOT / 'COBRA' / 'input files' / 'default data' / 'default_2023_valuation_data.csv',
    'VALUATION_2028': ROOT / 'COBRA' / 'input files' / 'default data' / 'default_2028_valuation_data.csv',
    'EMISSIONS_2016': ROOT / 'COBRA' / 'input files' / 'emissions' / 'Emissions_2016.csv',
    'EMISSIONS_2023': ROOT / 'COBRA' / 'input files' / 'emissions' / 'Emissions_2023.csv',
    'EMISSIONS_2028': ROOT / 'COBRA' / 'input files' / 'emissions' / 'Emissions_2028.csv',
}


def _normalize_csv_columns(df: pd.DataFrame, csv_key: str) -> pd.DataFrame:
    """CR / Valuation CSVs store the true beta in 'Adjusted'; expose it as 'Beta'."""
    if csv_key == 'CR_FUNCTIONS' or (csv_key and csv_key.startswith('VALUATION_')):
        if 'Adjusted' in df.columns and 'Beta' not in df.columns:
            df['Beta'] = df['Adjusted']
    return df


class CobraData:
    """Cached SQLite-backed access to the COBRA reference tables, with CSV fallback."""

    def __init__(self, db_path: Path = ROOT / "data" / "cobra_slim.db"):
        self.engine = create_engine(f"sqlite+pysqlite:///{db_path}")
        self._emissions_base: Optional[pd.DataFrame] = None
        self._emissions_control: Optional[pd.DataFrame] = None
        self._sr_matrices: Optional[dict] = None
        self._population: Optional[pd.DataFrame] = None
        self._incidence: Optional[pd.DataFrame] = None
        self._cr_functions: Optional[pd.DataFrame] = None
        self._valuation_functions: Optional[pd.DataFrame] = None
        self._voc2soa: Optional[dict] = None

    def _read(self, query: str, csv_key: Optional[str] = None) -> pd.DataFrame:
        """Run query; if it errors or returns empty and we have a CSV fallback, use that."""
        try:
            df = pd.read_sql(query, self.engine)
            if df.empty and csv_key and csv_key in CSV_PATHS:
                csv_path = CSV_PATHS[csv_key]
                if csv_path.exists():
                    print(f"SQL returned empty data, falling back to CSV: {csv_path}")
                    df = _normalize_csv_columns(pd.read_csv(csv_path), csv_key)
            return df
        except Exception as e:
            if csv_key and csv_key in CSV_PATHS:
                csv_path = CSV_PATHS[csv_key]
                if csv_path.exists():
                    print(f"SQL query failed ({e}), falling back to CSV: {csv_path}")
                    return _normalize_csv_columns(pd.read_csv(csv_path), csv_key)
            raise

    def load_emissions_base(self, year: Optional[int] = None) -> pd.DataFrame:
        if self._emissions_base is None:
            csv_key = f'EMISSIONS_{year}' if year else 'EMISSIONS_2023'
            self._emissions_base = self._read("SELECT * FROM SYS_Emissions_Base", csv_key)
        return self._emissions_base

    def load_emissions_control(self, year: Optional[int] = None) -> pd.DataFrame:
        if self._emissions_control is None:
            csv_key = f'EMISSIONS_{year}' if year else 'EMISSIONS_2023'
            self._emissions_control = self._read("SELECT * FROM SYS_Emissions_Control", csv_key)
        return self._emissions_control

    def summarize_emissions(self, emissions: pd.DataFrame) -> pd.DataFrame:
        """Group by typeindx + sourceindx, summing pollutants. SOA recomputed from VOC per row."""
        pollutant_cols = ['NOx', 'SO2', 'NH3', 'SOA', 'PM25', 'VOC']
        if all(c in emissions.columns for c in ['TIER1', 'TIER2', 'TIER3']):
            emissions = emissions.copy()
            voc2soa = self.load_voc2soa()
            tier_keys = (emissions['TIER1'].astype(int).astype(str) + '|' +
                         emissions['TIER2'].astype(int).astype(str) + '|' +
                         emissions['TIER3'].astype(int).astype(str))
            factors = tier_keys.map(voc2soa).fillna(0.0)
            emissions['SOA'] = emissions['VOC'] * factors
        return emissions.groupby(['typeindx', 'sourceindx'], as_index=False)[pollutant_cols].sum()

    def modify_emissions(self, raw_emissions: pd.DataFrame, emissions: dict[str, float],
                         tier_ids: list, sourceindxs: list[int]) -> pd.DataFrame:
        """Set the new aggregate for matching tier/county rows = `emissions` (REPLACE).

        Each matching row is scaled by ratio = payload / current_sum so the new aggregate
        equals the payload. When current_sum is zero, payload is spread evenly. Values
        clamped to >= 0. SOA is recomputed from VOC using the per-tier factor.
        """
        pollutant_cols = ['NOx', 'SO2', 'NH3', 'SOA', 'PM25', 'VOC']
        modified = raw_emissions.copy()
        mask = (
            (modified['TIER1'] == int(tier_ids[0])) &
            (modified['TIER2'] == int(tier_ids[1])) &
            (modified['TIER3'] == int(tier_ids[2])) &
            (modified['sourceindx'].isin(sourceindxs))
        )
        matching_idx = modified.index[mask]
        rowcount = len(matching_idx)

        if rowcount > 0:
            current_sums = modified.loc[matching_idx, pollutant_cols].sum()
            for col in pollutant_cols:
                current = current_sums[col]
                payload = emissions.get(col, 0)
                if current != 0:
                    ratio = payload / current
                    modified.loc[matching_idx, col] = (
                        modified.loc[matching_idx, col] * ratio
                    ).clip(lower=0)
                else:
                    modified.loc[matching_idx, col] = max(payload / rowcount, 0)
        else:
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

        voc2soa = self.load_voc2soa()
        tier_key = f"{int(tier_ids[0])}|{int(tier_ids[1])}|{int(tier_ids[2])}"
        factor = voc2soa.get(tier_key, 0.0)
        tier_mask = (
            (modified['TIER1'] == int(tier_ids[0])) &
            (modified['TIER2'] == int(tier_ids[1])) &
            (modified['TIER3'] == int(tier_ids[2])) &
            (modified['sourceindx'].isin(sourceindxs))
        )
        modified.loc[tier_mask, 'SOA'] = modified.loc[tier_mask, 'VOC'] * factor
        return modified

    def load_sr_matrices(self, cache_path: Path = ROOT / "data" / "sr_matrices.npz") -> dict:
        """Source-Receptor matrices from the .npz cache.

        Keys: 'dp' (direct PM2.5), 'NOx', 'SO4', 'O3V', 'O3N'. Each value is a list of
        4 jax arrays (one per stack type), shape [3108 x 3108].
        """
        if self._sr_matrices is not None:
            return self._sr_matrices
        if not cache_path.exists():
            raise FileNotFoundError(
                f"SR matrix cache missing at {cache_path}. Rebuild the .npz from "
                f"the source COBRA database before running the platform."
            )
        pollutant_keys = ['dp', 'NOx', 'SO4', 'O3V', 'O3N']
        data = jnp.load(cache_path)
        self._sr_matrices = {
            key: [jnp.array(data[f"{key}_{t}"]) for t in range(NUM_STACK_TYPES)]
            for key in pollutant_keys
        }
        return self._sr_matrices

    def load_population(self, year: Optional[int] = None) -> pd.DataFrame:
        if self._population is None:
            if year:
                q = f"SELECT * FROM SYS_POP_INVENTORY WHERE Year = {year} ORDER BY DestinationID"
                csv_key = f'POPULATION_{year}'
            else:
                q = "SELECT * FROM SYS_POP ORDER BY DestinationID"
                csv_key = 'POPULATION_2023'
            self._population = self._read(q, csv_key)
        return self._population

    def load_incidence(self, year: Optional[int] = None) -> pd.DataFrame:
        if self._incidence is None:
            if year:
                q = f"SELECT * FROM SYS_Incidence_Inventory WHERE Year = {year} ORDER BY DestinationID"
                csv_key = f'INCIDENCE_{year}'
            else:
                q = "SELECT * FROM SYS_Incidence ORDER BY DestinationID"
                csv_key = 'INCIDENCE_2023'
            self._incidence = self._read(q, csv_key)
        return self._incidence

    def load_cr_functions(self) -> pd.DataFrame:
        if self._cr_functions is None:
            self._cr_functions = self._read("SELECT * FROM SYS_CR", 'CR_FUNCTIONS')
        return self._cr_functions

    def load_valuation_functions(self) -> pd.DataFrame:
        if self._valuation_functions is None:
            self._valuation_functions = self._read(
                "SELECT * FROM SYS_Valuation", 'VALUATION_2023'
            )
        return self._valuation_functions

    def load_voc2soa(self) -> dict:
        if self._voc2soa is None:
            df = self._read("SELECT TIER1, TIER2, TIER3, FACTOR FROM SYS_voc2soa")
            self._voc2soa = {
                f"{int(r['TIER1'])}|{int(r['TIER2'])}|{int(r['TIER3'])}": float(r['FACTOR'] or 0)
                for _, r in df.iterrows()
            }
        return self._voc2soa
