
class UnitConverter:
    """A class for converting between units of measurement."""
    def convert_time(self, value, initial_unit, final_unit):
        """Converts between time units."""
        value = float(value)
        if initial_unit == 's' and final_unit == 'd':
            return value / 86400
        elif initial_unit == 'd' and final_unit == 's':
            return value * 86400
        elif initial_unit == 's' and final_unit == 'y':
            return value / 31557600
        elif initial_unit == 'y' and final_unit == 's':
            return value * 31557600
        elif initial_unit == 'd' and final_unit == 'y':
            return value / 365
        elif initial_unit == 'y' and final_unit == 'd':
            return value * 365
        elif initial_unit == 'min' and final_unit == 'y':
            return value / 525600
        elif initial_unit == 'y' and final_unit == 'min':
            return value * 525600
        elif initial_unit == 'min' and final_unit == 'd':
            return value / 1440
        elif initial_unit == 'd' and final_unit == 'min':
            return value * 1440
        elif initial_unit == 'min' and final_unit == 's':
            return value / 60
        elif initial_unit == 's' and final_unit == 'min':
            return value * 60
        elif initial_unit == 'min' and final_unit == 'h':
            return value / 60
        elif initial_unit == 'h' and final_unit == 'min':
            return value * 60
        elif initial_unit == 'h' and final_unit == 'd':
            return value / 24
        elif initial_unit == 'd' and final_unit == 'h':
            return value * 24
        elif initial_unit == 'h' and final_unit == 's':
            return value * 3600
        elif initial_unit == 's' and final_unit == 'h':
            return value / 3600
        elif initial_unit == 'h' and final_unit == 'y':
            return value / 8760
        elif initial_unit == 'y' and final_unit == 'h':
            return value * 8760
        elif initial_unit == final_unit:
            return value
        else:
            raise ValueError(f"Invalid unit conversion: {initial_unit} to {final_unit}")





