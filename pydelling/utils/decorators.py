import functools

def set_run(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        value = func(self, *args, **kwargs)
        self.is_run = True
        return value
    return wrapper