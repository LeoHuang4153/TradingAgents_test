from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel


class AnalystType(str, Enum):
    MARKET = "market"
    SENTIMENT = "sentiment"
    # Backward compatible name for sentiment analyst
    SOCIAL = "sentiment"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
