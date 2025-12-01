from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from pathlib import Path


def get_macro_ind(
    curr_date: Annotated[str, "The current date you are trading on, YYYY-mm-dd"],
) -> str:
    """
    Retrieve the most recent three years of macroeconomic indicators from a local CSV.

    The data is sourced from ``macro_data.csv`` at the repository root and includes
    monthly observations for CPI, inflation, Federal Funds Rate, GDP, labor market,
    and consumption indicators.

    Args:
        curr_date (str): The current trading date in ``YYYY-mm-dd`` format.

    Returns:
        str: A CSV-formatted string containing rows within the three-year window
            ending at ``curr_date``. If no data is found for the window, a
            descriptive message is returned instead.
    """

    try:
        current_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("curr_date must be in YYYY-mm-dd format") from exc

    start_dt = current_dt - relativedelta(years=3)

    csv_path = Path(__file__).resolve().parents[2] / "macro_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Macro data file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("macro_data.csv must contain a 'date' column")

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    filtered = df[(df["date"] >= start_dt) & (df["date"] <= current_dt)].copy()

    if filtered.empty:
        return (
            "No macroeconomic data available between "
            f"{start_dt.strftime('%Y-%m-%d')} and {curr_date}."
        )

    filtered["date"] = filtered["date"].dt.strftime("%Y-%m-%d")

    # Round numeric columns for readability
    for col in filtered.columns:
        if col != "date" and pd.api.types.is_numeric_dtype(filtered[col]):
            filtered[col] = filtered[col].round(2)

    csv_output = filtered.to_csv(index=False)

    header = (
        "## Macroeconomic indicators (monthly)\n"
        f"Window: {start_dt.strftime('%Y-%m-%d')} to {curr_date}\n"
        "Columns: date, CPI, INFLATION, FEDERAL_FUNDS_RATE, REAL_GDP, "
        "REAL_GDP_PER_CAPITA, UNEMPLOYMENT, NONFARM_PAYROLL, DURABLES, RETAIL_SALES\n\n"
    )

    return header + csv_output
