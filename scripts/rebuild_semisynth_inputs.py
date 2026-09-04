from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from crl_hte.semisynth import diagnosis_prevalence_audit


ID_COLUMNS = ["Patient_Level_ID1", "Patient_Level_ID2", "Pregnancy_Level_ID", "MHP_UserId"]
CHILD_COLUMNS = ["BIRTH_WEIGHT", "NICUREADMISSION", "STILLBORN", "EST_GEST_AGE_DAYS", "NICU_IND", "BABY_DISPOSITION"]
LEAKAGE_COLUMNS = [
    "Depression_Postpartum", "Postpartum_Depression_Postpartum", "EPDSmax_Postpartum",
    "EPDSave_Postpartum", "PHQ9max_Postpartum", "PHQ9ave_Postpartum",
    "Postpartum_Depression_No_FirstPNV", "Postpartum_Depression_Unknown",
    "Postpartum_Depression_After_FirstPNV",
]
POSTPARTUM_PATTERNS = ["_Postpartum", "Postpartum_Visit_10wk", "POSTPARTUM_LOS", "_After_FirstPNV", "_No_FirstPNV"]
DELIVERY_PATTERNS = ["_Delivery", "_Unknown"]
DIAGNOSIS_PREFIXES = (
    "Diabetes_", "Anxiety_", "Depression_", "Postpartum_Depression_", "HTN_",
    "GestHTN", "MentalBehavSUD_", "Bipolar_", "OCD_", "Trauma_Reaction_",
)
YES_NO_MAPPING = {"Yes": 1, "No": 0, "Y": 1, "N": 0}
STATUS_MAPPING = {"Done": 1, "Pending": 0}
DISPOSITION_MAPPING = {
    "HOME/SELF CARE": "home", "Home or Self Care": "home",
    "Home-Health Care Svc": "home_health", "HOME HEALTH AGENCY": "home_health",
    "SHORT TERM GEN HOSPT": "hospital", "Short Term Hospital": "hospital",
    "Short Term Hospital w/ Planned Readmission": "hospital_readmit",
    "Cancer Center/Children's Hospital": "specialty", "TO CANCR/CHILD HOSP": "specialty",
    "Rehab Facility": "rehab", "REHAB/INPT REHAB UNT": "rehab",
    "CTB": "deceased", "CTB - NO AUTOPSY": "deceased", "CTB - AUTOPSY": "deceased",
    "AGAINST MED ADVICE": "ama",
}


def parse_duration(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "0":
        return 0.0
    if "days" not in text.lower():
        return pd.to_numeric(value, errors="coerce")
    match = re.match(r"(\d+)\s*days?\s*([\d:\.]+)?", text)
    if not match:
        return np.nan
    days = float(match.group(1))
    if match.group(2):
        parts = match.group(2).split(":")
        hours = float(parts[0]) if len(parts) > 0 else 0.0
        minutes = float(parts[1]) if len(parts) > 1 else 0.0
        seconds = float(parts[2]) if len(parts) > 2 else 0.0
        days += (hours * 3600 + minutes * 60 + seconds) / 86400
    return days


def aggregate_children(child: pd.DataFrame) -> pd.DataFrame:
    if "NICU_IND" in child:
        original = child["NICU_IND"]
        mapped = original.map({"Y": 1, "N": 0})
        child["NICU_IND"] = mapped.where(mapped.notna(), pd.to_numeric(original, errors="coerce"))
    methods = {}
    for column in CHILD_COLUMNS:
        if column not in child:
            continue
        values = set(child[column].dropna().unique())
        if values.issubset({0, 1, 0.0, 1.0}):
            methods[column] = "max"
        elif pd.api.types.is_numeric_dtype(child[column]):
            methods[column] = "mean"
        else:
            methods[column] = "first"
    return child.groupby("Pregnancy_Level_ID").agg(methods).reset_index()


def create_target(df: pd.DataFrame) -> pd.Series:
    target = pd.Series(False, index=df.index)
    for column in ("Depression_Postpartum", "Postpartum_Depression_Postpartum"):
        if column in df:
            target |= df[column].eq(1).fillna(False)
    if "EPDSmax_Postpartum" in df:
        target |= df["EPDSmax_Postpartum"].ge(13).fillna(False)
    if "PHQ9max_Postpartum" in df:
        target |= df["PHQ9max_Postpartum"].ge(10).fillna(False)
    return target.astype(int)


def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    exclusions = set(ID_COLUMNS + ["PPD_Target"] + LEAKAGE_COLUMNS)
    for column in df:
        if any(pattern in column for pattern in POSTPARTUM_PATTERNS + DELIVERY_PATTERNS):
            exclusions.add(column)
    indicators = {}
    for column in df:
        if column in exclusions:
            continue
        rate = df[column].isna().mean()
        if 0.01 < rate <= 0.99:
            indicators[f"{column}_missing"] = df[column].isna().astype(int)
    return pd.concat([df, pd.DataFrame(indicators)], axis=1)


def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    preserved = {column: result[column].copy() for column in ID_COLUMNS if column in result}
    for column in result:
        if column in ID_COLUMNS or result[column].dtype != "object":
            continue
        values = set(result[column].dropna().unique())
        if values.issubset(YES_NO_MAPPING):
            result[column] = result[column].map(YES_NO_MAPPING)
        elif column == "MHP_OnboardingStatus":
            result[column] = result[column].map(STATUS_MAPPING)
        else:
            if column == "BABY_DISPOSITION":
                result[column] = result[column].map(
                    lambda value: DISPOSITION_MAPPING.get(value, "other") if pd.notna(value) else np.nan
                )
            categories = sorted(result[column].dropna().unique())
            result[column] = result[column].map({value: i for i, value in enumerate(categories)})

    empty = [column for column in result if result[column].isna().all() and column not in ID_COLUMNS]
    constant = [
        column for column in result
        if column not in empty and column not in ID_COLUMNS and result[column].nunique() <= 1
    ]
    result = result.drop(columns=empty + constant)
    features = [column for column in result if column not in ID_COLUMNS]
    for column in features:
        rate = result[column].isna().mean()
        if 0 < rate <= 0.5:
            if result[column].nunique() <= 2:
                mode = result[column].mode()
                if not mode.empty:
                    result[column] = result[column].fillna(mode.iloc[0])
            else:
                median = result[column].median()
                if pd.notna(median):
                    result[column] = result[column].fillna(median)
    result[features] = MinMaxScaler().fit_transform(result[features]).clip(0, 1)
    for column, values in preserved.items():
        if column in result:
            result[column] = values
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pregnancy", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--schema-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    pregnancy = pd.read_csv(args.pregnancy, low_memory=False)
    child = pd.read_csv(args.child, low_memory=False)
    child_aggregated = aggregate_children(child)
    frame = pregnancy.merge(child_aggregated, on="Pregnancy_Level_ID", how="left")
    frame = frame.drop(columns=[column for column in frame if frame[column].isna().all()])

    for column in frame:
        sample = frame[column].dropna().head(10)
        if len(sample) and any("days" in str(value).lower() for value in sample):
            frame[column] = frame[column].map(parse_duration)

    frame["PPD_Target"] = create_target(frame)
    frame = add_missingness_indicators(frame)

    audit = diagnosis_prevalence_audit(
        frame,
        frame["MHP_OnboardingStatus"].map(STATUS_MAPPING),
        prefixes=DIAGNOSIS_PREFIXES,
        min_reference_rate=0.01,
    )
    if not np.isfinite(audit["ratio"]) or audit["ratio"] < 0.5:
        raise ValueError(f"Corrected diagnosis audit failed: ratio={audit['ratio']}")

    ehr_schema = set(pd.read_csv(
        args.schema_root / "ehr_only" / "all_features_raw.csv", nrows=0
    ).columns)
    retained = [
        column for column in frame
        if column in ehr_schema or column in ID_COLUMNS or column == "PPD_Target"
    ]
    cohorts = {
        "comprehensive": frame,
        "ehr_only": frame[retained].copy(),
        "mhp_only": frame.loc[frame["MHP_OnboardingStatus"].eq("Done")].copy(),
    }

    shapes = {}
    for name, cohort in cohorts.items():
        raw_path = args.output_root / name / "all_features_raw.csv"
        processed_path = args.output_root / name / "all_features_processed.csv"
        write_csv(cohort, raw_path)
        processed = encode_and_scale(cohort)
        write_csv(processed, processed_path)
        shapes[name] = {"raw": list(cohort.shape), "processed": list(processed.shape)}
        print(name, shapes[name])

    manifest = {
        "pregnancy_sha256": sha256(args.pregnancy),
        "child_sha256": sha256(args.child),
        "diagnosis_prevalence_ratio": audit["ratio"],
        "diagnosis_features_audited": len(audit["columns"]),
        "shapes": shapes,
    }
    manifest_path = args.output_root / "rebuild_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(manifest_path)


if __name__ == "__main__":
    main()
