"""
Base interface for a reader class
"""
import logging

import numpy as np

from pydelling.readers import PflotranObservationPointReader

logger = logging.getLogger(__name__)
from pydelling.config import config
import logging
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


class PflotranMassBalanceFileReader(PflotranObservationPointReader):
    observation_point: np.ndarray
    observation_boundary: str
    observation_node: int
    variables: dict

    def __init__(self, filename=None):
        self.filename = Path(filename) if filename else Path(config.pflotran_reader.filename)
        logger.info(f"Reading PFLOTRAN mass balance file from {self.filename}")
        super().__init__(filename=self.filename)
        self.results = self.data.copy()