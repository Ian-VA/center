from collections import defaultdict

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

from data import CobraData

NUM_COUNTIES = 3108
NUM_STACK_TYPES = 4
AGE_COLUMNS = [f"Age{i}" for i in range(100)]


class Compute:
    def __init__(self, data: CobraData, load_sr = True):
        self.data = data
        self.data.load_emissions_base()
        self.data.load_emissions_control()

        if load_sr:
            self.data.load_sr_matrices()
        self.data.load_cr_functions()
        self.data.load_valuation_functions()

    def vectorize(self, emissions_summary: pd.DataFrame) -> dict[str, jnp.ndarray]:
        # Apply SR matrices to emissions

        # initialize pollution vectors (4 for each stack height for each pollutant in keys)
        # Each pollutant has 4 vectors (one per stack type), each of length NUM_COUNTIES
        keys = ["PM25", "SO2", "NOx", "SOA", "VOC"]
        partial_keys = ["PM25", "SO4", "NOx", "SOA", "O3V", "O3N"]

        # IMPORTANT: Create separate lists for each key to avoid shared references
        partials = {key: [jnp.zeros(NUM_COUNTIES) for _ in range(NUM_STACK_TYPES)] for key in partial_keys}
        finals = {key: jnp.zeros(NUM_COUNTIES) for key in partial_keys}

        # Convert to numpy for mutation, then back to jax
        pol_vectors_np = {key: [np.zeros(NUM_COUNTIES) for _ in range(NUM_STACK_TYPES)] for key in keys}

        # populate pollution vectors from summarized emissions
        for _, row in emissions_summary.iterrows():
            t = int(row["typeindx"]) - 1
            s = int(row["sourceindx"]) - 1

            for key in keys:
                pol_vectors_np[key][t][s] = row[key] or 0.0

        # Convert back to JAX arrays
        pol_vectors = {key: [jnp.array(arr) for arr in arrs] for key, arrs in pol_vectors_np.items()}

        # multiply SR matrices by pollution vectors to get air quality vectors at destination counties
        # SR matrix: (3108 x 3108), pol_vector: (3108,) -> result: (3108,)
        sr_matrices = self.data.load_sr_matrices()
        for i in range(NUM_STACK_TYPES):
            partials["PM25"][i] = sr_matrices["dp"][i] @ pol_vectors["PM25"][i]
            partials["NOx"][i] = sr_matrices["NOx"][i] @ pol_vectors["NOx"][i]
            partials["SO4"][i] = sr_matrices["SO4"][i] @ pol_vectors["SO2"][i]
            partials["SOA"][i] = sr_matrices["dp"][i] @ pol_vectors["SOA"][i] * 28778  # unit conversion from COBRA
            partials["O3V"][i] = sr_matrices["O3V"][i] @ pol_vectors["VOC"][i]
            partials["O3N"][i] = sr_matrices["O3N"][i] @ pol_vectors["NOx"][i]

        # sum air quality vectors across all 4 stack heights
        for key in partial_keys:
            for i in range(NUM_STACK_TYPES):
                finals[key] = finals[key] + partials[key][i]

        return finals

    def compute_pm(self, aq_vectors: dict[str, jnp.ndarray]):
        # Calculate PM2.5 concentrations
        return aq_vectors["PM25"] + aq_vectors["NOx"] + aq_vectors["SO4"]

    def compute_o3(self, aq_vectors: dict[str, jnp.ndarray]) -> jnp.ndarray:
        # Calculate O3 concentrations
        return aq_vectors["O3V"] + aq_vectors["O3N"]

    # def compute_delta(self, base_vec: jnp.ndarray, control_vec: jnp.ndarray) -> jnp.ndarray:
    #     # Calculate Base - Control
    #     return base_vec - control_vec

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
        """Return annualization factor: 365 for daily, 152 for ozone season, 1 otherwise."""
        if not seasonal_metric:
            return 1.0
        metric = seasonal_metric.strip().upper()
        if metric == "DAILY":
            return 365.0
        elif metric == "OZONE":
            return 152.0
        return 1.0

    @staticmethod
    def _discount_adjustment(rate_pct: float) -> float:
        """
        Compute 20-year NPV adjustment factor matching COBRA's adjustmentfactorfromdiscountrate().
        rate_pct: discount rate as a percentage (e.g. 3 for 3%).
        """
        factor = rate_pct / 100.0
        result = 0.0
        weights = (
            [0.3] +                                     # year 0
            [0.1] * 5 +                                 # years 1-5
            [0.0142857142857143] * 14                    # years 6-19
        )
        for year, weight in enumerate(weights):
            result += weight / ((1 + factor) ** year)
        return result

    INFLATION_ADJUSTMENT = 1.1225

    def _compile_crfunc(self, function_str: str):
        """Compile a CR function string once and cache it."""
        cache = getattr(self, "_crfunc_cache", None)
        if cache is None:
            self._crfunc_cache = {}
            cache = self._crfunc_cache
        if function_str not in cache:
            cache[function_str] = compile(function_str.lower(), "<crfunc>", "eval")
        return cache[function_str]

    def crfunc(
        self,
        function_str: str,
        incidence,
        beta,
        delta_pm,
        delta_o3,
        pop,
        a=0,
        b=0,
        c=0,
    ):
        # Apply CR functions, return health cases
        code = self._compile_crfunc(function_str)

        return eval(
            code,
            {"__builtins__": {}},
            {
                "incidence": incidence,
                "exp": jnp.exp,
                "beta": beta,
                "pop": pop,
                "a": a,
                "b": b,
                "c": c,
                "deltaq": delta_pm,
                "deltao": delta_o3,
            },
        )

    def compute_cases(self, delta_pm, delta_o3):
        # Loop over each CR function and aggregate the total cases of each endpoint
        population = self.data.load_population(2023)
        incidence = self.data.load_incidence(2023)

        infliction_cases = defaultdict(lambda: jnp.zeros(len(population)))
        cr_functions = self.data.load_cr_functions()
        for _, row in cr_functions.iterrows():
            pooling_weight = float(row.get("PoolingWeight", 0) or 0)
            metric_adj = self._seasonal_metric_adjustment(row.get("Seasonal_Metric", ""))
            for age in range(int(row["Start_Age"]), int(row["End_Age"]) + 1):
                raw = self.crfunc(
                    row["Function"],
                    self._get_aligned_incidence(incidence, row["IncidenceEndpoint"], f"Age{age}"),
                    jnp.array(row["Beta"]),
                    delta_pm,
                    delta_o3,
                    jnp.array(population[f"Age{age}"]),
                    float(row["A"] if pd.notna(row.get("A")) else 0),
                    float(row["B"] if pd.notna(row.get("B")) else 0),
                    float(row["C"] if pd.notna(row.get("C")) else 0),
                )
                infliction_cases[row["Endpoint"]] += raw * pooling_weight * metric_adj

        return infliction_cases

    def compute_valuation(self, delta_pm, delta_o3, discount_rate: float = 0):
        # Apply dollar values to health cases, return costs
        # discount_rate: percentage (e.g. 3 for 3%). 0 means no custom discounting.
        population = self.data.load_population(2023)
        incidence = self.data.load_incidence(2023)

        endpoint_costs = defaultdict(lambda: jnp.zeros(len(population)))
        valuation_functions = self.data.load_valuation_functions()

        # COBRA inflation adjustment factor (study-year to model-year dollars)
        INFLATION_FACTOR = 1.1225

        for _, row in valuation_functions.iterrows():
            pooling_weight = float(row.get("PoolingWeight", 0) or 0)
            metric_adj = self._seasonal_metric_adjustment(row.get("Seasonal_Metric", ""))
            value = float(row.get("Value", 0) or 0)
            apply_discount = str(row.get("ApplyDiscount", "NO")).strip().upper()

            # Determine valuation multiplier (matches COBRA 1.1225 inflation factor + discount)
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
                    delta_pm,
                    delta_o3,
                    jnp.array(population[f"Age{age}"]),
                    float(row["A"] if pd.notna(row.get("A")) else 0),
                    float(row["B"] if pd.notna(row.get("B")) else 0),
                    float(row["C"] if pd.notna(row.get("C")) else 0),
                )
                endpoint_costs[row["Endpoint"]] += cases * pooling_weight * metric_adj * val_multiplier

        return endpoint_costs

    def _compute_cases_and_valuation(self, delta_pm, delta_o3, discount_rate: float = 0):
        """Single-pass computation of both cases and valuation costs.

        The CR and valuation function tables share identical rows (same
        endpoints, functions, and age ranges).  This method evaluates each
        CR function once and accumulates both cases and costs, avoiding the
        duplicate work of calling compute_cases + compute_valuation separately.
        """
        population = self.data.load_population(2023)
        incidence = self.data.load_incidence(2023)
        n = len(population)

        infliction_cases = defaultdict(lambda: jnp.zeros(n))
        endpoint_costs = defaultdict(lambda: jnp.zeros(n))

        cr_functions = self.data.load_cr_functions()
        valuation_functions = self.data.load_valuation_functions()

        for (_, cr_row), (_, vf_row) in zip(cr_functions.iterrows(), valuation_functions.iterrows()):
            pooling_weight = float(cr_row.get("PoolingWeight", 0) or 0)
            metric_adj = self._seasonal_metric_adjustment(cr_row.get("Seasonal_Metric", ""))

            # Valuation multiplier
            value = float(vf_row.get("Value", 0) or 0)
            apply_discount = str(vf_row.get("ApplyDiscount", "NO")).strip().upper()
            if discount_rate == 0 or apply_discount == "NO":
                val_multiplier = value * self.INFLATION_ADJUSTMENT
            else:
                discount_adj = self._discount_adjustment(discount_rate)
                val_multiplier = value * discount_adj * self.INFLATION_ADJUSTMENT

            beta = jnp.array(cr_row["Beta"])
            a = float(cr_row["A"] if pd.notna(cr_row.get("A")) else 0)
            b = float(cr_row["B"] if pd.notna(cr_row.get("B")) else 0)
            c = float(cr_row["C"] if pd.notna(cr_row.get("C")) else 0)
            endpoint = cr_row["Endpoint"]
            case_scale = pooling_weight * metric_adj
            cost_scale = case_scale * val_multiplier

            for age in range(int(cr_row["Start_Age"]), int(cr_row["End_Age"]) + 1):
                raw = self.crfunc(
                    cr_row["Function"],
                    self._get_aligned_incidence(incidence, cr_row["IncidenceEndpoint"], f"Age{age}"),
                    beta, delta_pm, delta_o3,
                    jnp.array(population[f"Age{age}"]),
                    a, b, c,
                )
                infliction_cases[endpoint] += raw * case_scale
                endpoint_costs[endpoint] += raw * cost_scale

        return infliction_cases, endpoint_costs

    # Endpoints that represent alternative estimates of the same outcome.
    # Only one from each group should be included in a single total.
    _LOW_MORTALITY = "PM Mortality, All Cause (low)"
    _HIGH_MORTALITY = "PM Mortality, All Cause (high)"

    def compute_impacts(self, delta_pm, delta_o3, discount_rate):
        # Full pipeline — single pass computes both cases and costs
        cases, costs = self._compute_cases_and_valuation(delta_pm, delta_o3, discount_rate)

        cases_by_endpoint = {endpoint: float(jnp.sum(arr)) for endpoint, arr in cases.items()}
        costs_by_endpoint = {endpoint: float(jnp.sum(arr)) for endpoint, arr in costs.items()}

        # COBRA reports separate low/high totals because PM mortality has two
        # alternative estimates (low = ages 65-99, high = ages 18-99).
        # Summing both would double-count the mortality component.
        low_mort_cases = cases_by_endpoint.get(self._LOW_MORTALITY, 0.0)
        high_mort_cases = cases_by_endpoint.get(self._HIGH_MORTALITY, 0.0)
        low_mort_costs = costs_by_endpoint.get(self._LOW_MORTALITY, 0.0)
        high_mort_costs = costs_by_endpoint.get(self._HIGH_MORTALITY, 0.0)

        # Base totals exclude both alternative mortality endpoints
        base_cases = sum(v for k, v in cases_by_endpoint.items()
                         if k not in (self._LOW_MORTALITY, self._HIGH_MORTALITY))
        base_costs = sum(v for k, v in costs_by_endpoint.items()
                         if k not in (self._LOW_MORTALITY, self._HIGH_MORTALITY))

        summary = {
            "cases": cases_by_endpoint,
            "costs": costs_by_endpoint,
            "total_cases_low": base_cases + low_mort_cases,
            "total_cases_high": base_cases + high_mort_cases,
            "total_costs_low": base_costs + low_mort_costs,
            "total_costs_high": base_costs + high_mort_costs,
        }

        return {
            "cases": cases,
            "valuation": costs,
            "summary": summary,
        }
