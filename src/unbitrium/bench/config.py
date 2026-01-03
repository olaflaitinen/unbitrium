"""
Benchmark Configuration Schema.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class BenchmarkConfig(BaseModel):
    name: str
    description: Optional[str] = None
    runs: int = 1
