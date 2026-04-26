from collections import defaultdict

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

from data import CobraData

NUM_COUNTIES = 3108
NUM_STACK_TYPES = 4


class Compute:
    INFLATION_ADJUSTMENT = 1.1225  # COBRA study-year → model-year dollars

    def __init__(self, data: CobraData, load_sr: bool = True):
        self.data = data
        self.data.load_emissions_base()
        self.data.load_emissions_control()
        if load_sr:
            self.data.load_sr_matrices()
        self.data.load_cr_functions()
        self.data.load_valuation_functions()

    def vectorize(self, emissions_summary: pd.DataFrame) -> dict[str, jnp.ndarray]:
        """Apply SR matrices to summarized emissions to produce per-county AQ vectors."""
        keys = ["PM25", "SO2", "NOx", "SOA", "VOC"]
        partial_keys = ["PM25", "SO4", "NOx", "SOA", "O3V", "O3N"]

        partials = {key: [jnp.zeros(NUM_COUNTIES) for _ in range(NUM_STACK_TYPES)] for key in partial_keys}
        finals = {key: jnp.zeros(NUM_COUNTIES) for key in partial_keys}

        pol_vectors_np = {key: [np.zeros(NUM_COUNTIES) for _ in range(NUM_STACK_TYPES)] for key in keys}
        for _, row in emissions_summary.iterrows():
            t = int(row["typeindx"]) - 1
            s = int(row["sourceindx"]) - 1
            for key in keys:
                pol_vectors_np[key][t][s] = row[key] or 0.0
        pol_vectors = {key: [jnp.array(arr) for arr in arrs] for key, arrs in pol_vectors_np.items()}

        sr_matrices = self.data.load_sr_matrices()
        for i in range(NUM_STACK_TYPES):
            partials["PM25"][i] = sr_matrices["dp"][i] @ pol_vectors["PM25"][i]
            partials["NOx"][i] = sr_matrices["NOx"][i] @ pol_vectors["NOx"][i]
            partials["SO4"][i] = sr_matrices["SO4"][i] @ pol_vectors["SO2"][i]
            partials["SOA"][i] = sr_matrices["dp"][i] @ pol_vectors["SOA"][i] * 28778  # COBRA unit conversion
            partials["O3V"][i] = sr_matrices["O3V"][i] @ pol_vectors["VOC"][i]
            partials["O3N"][i] = sr_matrices["O3N"][i] @ pol_vectors["NOx"][i]

        for key in partial_keys:
            for i in range(NUM_STACK_TYPES):
                finals[key] = finals[key] + partials[key][i]
        return finals

    def compute_pm(self, aq_vectors: dict[str, jnp.ndarray]):
        return aq_vectors["PM25"] + aq_vectors["NOx"] + aq_vectors["SO4"]

    def compute_o3(self, aq_vectors: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return aq_vectors["O3V"] + aq_vectors["O3N"]

    def _get_aligned_incidence(self, incidence_df, endpoint, age_col):
        """Get incidence values aligned by DestinationID (1-3108), filling gaps with 0."""
        cache = getattr(self, "_incidence_cache", None)
        if cache is None:
            self._incidence_cache = {}
            cache = self._incidence_cache
        key = (endpoint, age_col)
        if key in cache:
            return cache[key]

        endpoint_inc = incidence_df[incidence_df["Endpoint"] == endpoint]
        arr = np.zeros(NUM_COUNTIES)
        valid = endpoint_inc[endpoint_inc["DestinationID"].between(1, NUM_COUNTIES)]
        if len(valid) > 0:
            arr[valid["DestinationID"].values - 1] = valid[age_col].values
        result = jnp.array(arr)
        cache[key] = result
        return result

    @staticmethod
    def _seasonal_metric_adjustment(seasonal_metric: str) -> float:
        """365 for daily, 152 for ozone-season, 1 otherwise."""
        if not seasonal_metric:
            return 1.0
        metric = seasonal_metric.strip().upper()
        if metric == "DAILY":
            return 365.0
        if metric == "OZONE":
            return 152.0
        return 1.0

    @staticmethod
    def _discount_adjustment(rate_pct: float) -> float:
        """20-year NPV factor matching COBRA's adjustmentfactorfromdiscountrate()."""
        factor = rate_pct / 100.0
        weights = [0.3] + [0.1] * 5 + [0.0142857142857143] * 14
        return sum(w / ((1 + factor) ** y) for y, w in enumerate(weights))

    def _compile_crfunc(self, function_str: str):
        cache = getattr(self, "_crfunc_cache", None)
        if cache is None:
            self._crfunc_cache = {}
            cache = self._crfunc_cache
        if function_str not in cache:
            cache[function_str] = compile(function_str.lower(), "<crfunc>", "eval")
        return cache[function_str]

    def crfunc(self, function_str: str, incidence, beta, delta_pm, delta_o3, pop, a=0, b=0, c=0):
        code = self._compile_crfunc(function_str)
        return eval(
            code,
            {"__builtins__": {}},
            {
                "incidence": incidence, "exp": jnp.exp, "beta": beta, "pop": pop,
                "a": a, "b": b, "c": c, "deltaq": delta_pm, "deltao": delta_o3,
            },
        )

    def compute_valuation(self, delta_pm, delta_o3, discount_rate: float = 0):
        """Apply dollar values to health cases. discount_rate is a percentage (e.g. 3 for 3%)."""
        population = self.data.load_population(2023)
        incidence = self.data.load_incidence(2023)

        endpoint_costs = defaultdict(lambda: jnp.zeros(len(population)))
        valuation_functions = self.data.load_valuation_functions()

        for _, row in valuation_functions.iterrows():
            pooling_weight = float(row.get("PoolingWeight", 0) or 0)
            metric_adj = self._seasonal_metric_adjustment(row.get("Seasonal_Metric", ""))
            value = float(row.get("Value", 0) or 0)
            apply_discount = str(row.get("ApplyDiscount", "NO")).strip().upper()

            if discount_rate == 0 or apply_discount == "NO":
                val_multiplier = value * self.INFLATION_ADJUSTMENT
            else:
                discount_adj = self._discount_adjustment(discount_rate)
                val_multiplier = value * discount_adj * self.INFLATION_ADJUSTMENT

            for age in range(int(row["Start_Age"]), int(row["End_Age"]) + 1):
                cases = self.crfunc(
                    row["Function"],
                    self._get_aligned_incidence(incidence, row["IncidenceEndpoint"], f"Age{age}"),
                    jnp.array(row["Beta"]),
                    delta_pm, delta_o3,
                    jnp.array(population[f"Age{age}"]),
                    float(row["A"] if pd.notna(row.get("A")) else 0),
                    float(row["B"] if pd.notna(row.get("B")) else 0),
                    float(row["C"] if pd.notna(row.get("C")) else 0),
                )
                endpoint_costs[row["Endpoint"]] += cases * pooling_weight * metric_adj * val_multiplier

        return endpoint_costs
