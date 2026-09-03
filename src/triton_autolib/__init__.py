from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("triton-autolib")
except PackageNotFoundError:
    __version__ = "unknown"


from triton_autolib import backward, forward
from triton_autolib.constants import *
