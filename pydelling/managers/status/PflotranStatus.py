from .BaseStatus import BaseStatus
import logging
import re

logger = logging.getLogger(__name__)

class PflotranStatus(BaseStatus):
    """This class reads a pflotran status file and extracts some key information."""
    def __init__(self,
                 status_file,
                 total_time=None):
        super().__init__(status_file)
        self.times = []
        self.dts = []
        self.wall_clock_time = None
        self.total_time = total_time
        self.extract_key_information()

    def extract_key_information(self):
        """Extract total Time, Dt, and Wall Clock Time values from the status data."""

        # Search for total Time and Dt values
        pattern = re.compile(r"== REACTIVE TRANSPORT =+.*?Step.*?Time=\s+([\d\.E\+\-]+).*?Dt=\s+([\d\.E\+\-]+)",
                             re.DOTALL)
        matches = pattern.findall(self.status_data)
        for match in matches:
            self.times.append(float(match[0]))
            self.dts.append(float(match[1]))

        # Search for Wall Clock Time value
        pattern = re.compile(r"Wall Clock Time:\s+([\d\.E\+\-]+)")
        match = pattern.search(self.status_data)
        if match:
            self.wall_clock_time = float(match.group(1))

    @property
    def progress(self):
        """Return the progress of the simulation."""
        if self.total_time is None:
            return None
        return self.times[-1] / self.total_time

    @property
    def is_done(self):
        """Return whether the simulation is done."""
        if self.wall_clock_time is None:
            return False
        else:
            return True

    def add_total_time(self, total_time):
        """Add the total time of the simulation."""
        self.total_time = total_time

    def read(self, status_file = None):
        """Read the status file and extract key information."""
        self.status_file = status_file if status_file is not None else self.status_file
        self.read_status_file()
        self.extract_key_information()



