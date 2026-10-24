"""Reproducible SYNTHETIC credit applications (Home Credit–style column names).

Every row is fabricated by a documented random process. It is NOT real borrower data,
and model performance on it says nothing about real lending performance.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from creditiq_core import defaults
from creditiq_core.data.csv import DatasetInfo, dataframe_fingerprint

_OCCUPATIONS = [
    "Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
    "High skill tech staff", "Accountants", "Medicine staff", "Security staff",
    "Cooking staff", "Cleaning staff", "Private service staff", "Low-skill Laborers",
    "Waiters/barmen staff", "Secretaries", "Realty agents", "HR staff", "IT staff",
]
_OCC_RISKY = {"Laborers", "Drivers", "Low-skill Laborers", "Waiters/barmen staff", "Security staff", "Cooking staff"}
_OCC_MALE_TILT = {"Laborers", "Drivers", "Security staff", "Low-skill Laborers", "IT staff", "High skill tech staff"}
_OCC_FEMALE_TILT = {"Sales staff", "Core staff", "Medicine staff", "Accountants", "Cleaning staff", "Secretaries", "HR staff", "Cooking staff"}

_ORGANIZATIONS = (
    ["Business Entity Type 3", "Self-employed", "Other", "Medicine", "Business Entity Type 2",
     "Government", "School", "Trade: type 7", "Kindergarten", "Construction",
     "Business Entity Type 1", "Transport: type 4", "Trade: type 3", "Industry: type 9",
     "Industry: type 3", "Security", "Housing", "Industry: type 11", "Military", "Bank",
     "Agriculture", "Police", "Transport: type 2", "Postal", "Security Ministries",
     "Trade: type 2", "Restaurant", "Services", "University", "Industry: type 7",
     "Transport: type 3", "Industry: type 1", "Hotel", "Electricity", "Industry: type 4",
     "Trade: type 6", "Insurance", "Telecom", "Emergency", "Realtor"]
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_credit_data(
    n_rows: int = 20_000,
    seed: int = defaults.DEFAULT_SEED,
    shifted: bool = False,
    target_rate: float = 0.08,
) -> pd.DataFrame:
    """Generate a reproducible SYNTHETIC application-level credit dataset.

    Column names follow a subset of Home Credit ``application_train.csv`` plus two
    clearly synthetic additions:

    * ``APPLICATION_DATE`` – a fabricated application date that enables the
      chronological-split demo (the real Home Credit file has no calendar dates).
    * ``POST_LOAN_COLLECTION_CALLS`` – a deliberately LEAKY post-outcome field
      generated from the target, included so the leakage detector can be
      demonstrated. It must never be used as a predictor.

    ``shifted=True`` produces a later "monitoring batch" with a deliberate
    distribution shift, for demonstrating drift detection.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    rng = np.random.default_rng(seed + (7919 if shifted else 0))
    cols, lin = _simulate_population(n_rows, rng, shifted)
    # The intercept is calibrated once on a fixed-size BASE reference population, so the
    # base data hits ``target_rate`` and a shifted batch's prevalence moves naturally.
    _, ref_lin = _simulate_population(20_000, np.random.default_rng(seed + 104_729), False)
    intercept = _calibrate_intercept(ref_lin, target_rate)
    target = (rng.random(n_rows) < _sigmoid(intercept + lin)).astype(int)
    collection_calls = np.where(target == 1, rng.poisson(4.0, n_rows) + 1, rng.poisson(0.15, n_rows))
    cols["SK_ID_CURR"] = cols["SK_ID_CURR"] + (900000 if shifted else 0)
    df = pd.DataFrame(cols)
    df.insert(2, "TARGET", target)
    df["POST_LOAN_COLLECTION_CALLS"] = collection_calls
    return df


def _calibrate_intercept(lin: np.ndarray, target_rate: float) -> float:
    lo, hi = -10.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(mid + lin).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _simulate_population(n: int, rng: np.random.Generator, shifted: bool) -> tuple[dict[str, Any], np.ndarray]:
    """Draw applicant features and the (intercept-free) default log-odds."""

    age = np.clip(rng.normal(41 if shifted else 43, 11.5, n), 21, 69)
    age_std = (age - 43) / 11.5
    z = 0.25 * age_std + rng.normal(-0.45 if shifted else 0.0, 1.0, n)  # latent creditworthiness

    gender = np.where(rng.random(n) < 0.66, "F", "M")

    pension_p = _sigmoid((age - 58) / 2.5) * 0.92
    is_pensioner = rng.random(n) < pension_p
    other_types = rng.choice(
        ["Working", "Commercial associate", "State servant", "Unemployed"],
        size=n, p=[0.63, 0.28, 0.087, 0.003],
    )
    income_type = np.where(is_pensioner, "Pensioner", other_types)

    years_employed = np.minimum(rng.exponential(6.0, n), np.maximum(age - 18, 0.5))
    days_employed = np.where(is_pensioner, 365243, -np.round(years_employed * 365)).astype(int)
    days_employed = np.where(income_type == "Unemployed", 365243, days_employed)

    edu_score = z + rng.normal(0, 1, n)
    education = np.select(
        [edu_score > 1.75, edu_score > 0.55, edu_score > 0.35, edu_score < -2.3],
        ["Academic degree", "Higher education", "Incomplete higher", "Lower secondary"],
        default="Secondary / secondary special",
    )
    education = np.where((education == "Academic degree") & (rng.random(n) > 0.03), "Higher education", education)

    fam = rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n, p=[0.64, 0.15, 0.10, 0.065, 0.045],
    )
    fam = np.where((age < 30) & (fam == "Widow"), "Single / not married", fam)
    children = np.clip(rng.poisson(0.42, n), 0, 5)
    partner = np.isin(fam, ["Married", "Civil marriage"]).astype(int)
    fam_members = (1 + partner + children).astype(float)

    housing = rng.choice(
        ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"],
        size=n, p=[0.885, 0.048, 0.036, 0.018, 0.0095, 0.0035],
    )
    housing = np.where((age < 28) & (rng.random(n) < 0.15), "With parents", housing)

    log_income = (
        11.9 + 0.18 * z + 0.25 * (education == "Higher education") - 0.25 * is_pensioner
        + 0.12 * (gender == "M") + rng.normal(0, 0.42, n) - (0.10 if shifted else 0.0)
    )
    income = np.round(np.exp(log_income) / 450) * 450

    contract = np.where(rng.random(n) < (0.93 if shifted else 0.905), "Cash loans", "Revolving loans")
    log_credit = 13.1 + 0.55 * (log_income - 11.9) + rng.normal(0, 0.55, n)
    credit = np.round(np.exp(log_credit) / 500) * 500
    credit = np.where(contract == "Revolving loans", np.round(credit * 0.45 / 500) * 500, credit)
    credit = np.maximum(credit, 45_000)

    term = rng.choice([12, 18, 24, 36, 48, 60], size=n, p=[0.12, 0.10, 0.22, 0.28, 0.15, 0.13])
    r_m = rng.uniform(0.10, 0.22, n) / 12
    annuity = credit * r_m / (1 - (1 + r_m) ** (-term))
    annuity = np.where(contract == "Revolving loans", credit * 0.05, annuity)
    annuity = np.round(annuity, 1)
    goods = np.where(contract == "Cash loans", np.round(credit * rng.uniform(0.8, 1.0, n) / 500) * 500, np.nan)
    goods[rng.random(n) < 0.01] = np.nan

    region_rating = np.select(
        [log_income + rng.normal(0, 0.5, n) > 12.55, log_income + rng.normal(0, 0.5, n) < 11.45],
        [1, 3], default=2,
    )

    occ_choice = rng.choice(_OCCUPATIONS, size=n)
    tilt = rng.random(n)
    swap_m = (gender == "M") & np.isin(occ_choice, list(_OCC_FEMALE_TILT)) & (tilt < 0.55)
    swap_f = (gender == "F") & np.isin(occ_choice, list(_OCC_MALE_TILT)) & (tilt < 0.55)
    occ_choice = np.where(swap_m, rng.choice(sorted(_OCC_MALE_TILT), size=n), occ_choice)
    occ_choice = np.where(swap_f, rng.choice(sorted(_OCC_FEMALE_TILT), size=n), occ_choice)
    occupation = occ_choice.astype(object)
    occupation[is_pensioner | (rng.random(n) < 0.12)] = np.nan

    org_p = np.linspace(2.5, 0.2, len(_ORGANIZATIONS))
    org = rng.choice(_ORGANIZATIONS, size=n, p=org_p / org_p.sum()).astype(object)
    org[is_pensioner] = "XNA"

    ext2 = _sigmoid(0.95 * z + rng.normal(0, 0.65, n) + 0.3)
    ext3 = _sigmoid(0.85 * z + 0.30 * age_std + rng.normal(0, 0.65, n))
    ext1 = _sigmoid(0.65 * z + 0.45 * age_std + rng.normal(0, 0.8, n) - 0.2)
    ext2[rng.random(n) < 0.002] = np.nan
    ext3[rng.random(n) < (0.35 if shifted else 0.20)] = np.nan
    ext1[rng.random(n) < 0.56] = np.nan

    phone_change = -np.round(rng.exponential(960, n))
    inquiries = np.clip(rng.poisson(np.where(z < -0.5, 2.6, 1.7)), 0, 15).astype(float)
    inquiries[rng.random(n) < 0.13] = np.nan

    own_car = np.where(rng.random(n) < _sigmoid(-0.9 + 0.8 * (gender == "M") + 0.6 * (log_income - 11.9)), "Y", "N")
    own_realty = np.where(rng.random(n) < 0.69, "Y", "N")

    if shifted:
        start, span_days = np.datetime64("2026-01-01"), 181
    else:
        start, span_days = np.datetime64("2023-01-01"), 1095
    offsets = rng.integers(0, span_days, n)
    app_date = start + offsets.astype("timedelta64[D]")
    year_frac = (app_date - np.datetime64("2024-01-01")).astype(int) / 365.0

    annuity_income = annuity / income
    yrs_emp_eff = np.where(days_employed > 0, 0, years_employed)
    lin = (
        -1.05 * z
        + 2.4 * np.maximum(annuity_income - 0.25, 0)
        - 0.20 * age_std
        - 0.06 * np.minimum(yrs_emp_eff, 15)
        + 0.25 * np.isin(education, ["Secondary / secondary special", "Lower secondary"])
        - 0.15 * np.isin(education, ["Higher education", "Academic degree"])
        + 0.30 * (region_rating - 2)
        + 0.30 * np.isin(occupation.astype(str), list(_OCC_RISKY))
        + 0.25 * (contract == "Cash loans")
        + 0.25 * np.isin(housing, ["Rented apartment", "With parents"])
        + 0.07 * np.nan_to_num(inquiries, nan=1.9)
        + 0.15 * (phone_change > -180)
        + 0.45 * (income_type == "Unemployed")
        + 0.30 * ((age < 30) & (credit / income > 4))
        + 0.12 * year_frac
    )

    cols = {
        "SK_ID_CURR": np.arange(100002, 100002 + n),
        "APPLICATION_DATE": pd.to_datetime(app_date).strftime("%Y-%m-%d"),
        "NAME_CONTRACT_TYPE": contract,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": fam,
        "NAME_HOUSING_TYPE": housing,
        "DAYS_BIRTH": -np.round(age * 365.25).astype(int),
        "DAYS_EMPLOYED": days_employed,
        "OCCUPATION_TYPE": occupation,
        "ORGANIZATION_TYPE": org,
        "CNT_FAM_MEMBERS": fam_members,
        "REGION_RATING_CLIENT": region_rating,
        "EXT_SOURCE_1": np.round(ext1, 6),
        "EXT_SOURCE_2": np.round(ext2, 6),
        "EXT_SOURCE_3": np.round(ext3, 6),
        "DAYS_LAST_PHONE_CHANGE": phone_change,
        "AMT_REQ_CREDIT_BUREAU_YEAR": inquiries,
    }
    return cols, lin


def synthetic_dataset(n_rows: int = 20_000, seed: int = defaults.DEFAULT_SEED) -> tuple[pd.DataFrame, DatasetInfo]:
    """Generate the synthetic demo dataset with its provenance record (no file access)."""
    df = generate_synthetic_credit_data(n_rows=n_rows, seed=seed)
    return df, synthetic_info(df, seed)


def synthetic_info(df: pd.DataFrame, seed: int = defaults.DEFAULT_SEED) -> DatasetInfo:
    info = DatasetInfo(
        name="synthetic_demo_data.csv",
        source="synthetic",
        is_synthetic=True,
        n_rows_loaded=len(df),
        n_cols=df.shape[1],
        n_rows_file=len(df),
        notes=[
            "SYNTHETIC demonstration data generated by creditiq_core.data.synthetic (seed "
            f"{seed}). Column names mimic Home Credit application_train.csv.",
            "APPLICATION_DATE and POST_LOAN_COLLECTION_CALLS are synthetic additions "
            "(the latter is deliberately leaky).",
        ],
    )
    info.fingerprint = dataframe_fingerprint(df, info.name)
    return info
