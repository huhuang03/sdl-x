import contextlib


class Config:
    enable_backprop = True


def no_grad():
    return using_config('enable_backprop', False)


@contextlib.contextmanager
def using_config(name, val):
    old_value = getattr(Config, name)
    setattr(Config, name, val)
    try:
        yield
    finally:
        setattr(Config, name, old_value)