from langchain_core.tools import tool
from typing import Annotated

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_macro_ind(
    curr_date: Annotated[str, "The current date you are trading on, YYYY-mm-dd"],
) -> str:
    """Retrieve the latest three years of macroeconomic indicators.

    This tool pulls monthly macro data from a local CSV (CPI, inflation,
    Federal Funds Rate, GDP, labor market, and consumption metrics) and returns
    a CSV-formatted string limited to the past three years relative to
    ``curr_date``.
    """

    return route_to_vendor("get_macro_ind", curr_date)
