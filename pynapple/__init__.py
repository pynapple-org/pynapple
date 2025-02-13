from importlib.metadata import PackageNotFoundError, version

from .core import (
    IntervalSet,
    Ts,
    Tsd,
    TsdFrame,
    TsdTensor,
    TsGroup,
    TsIndex,
    nap_config,
)
from .io import *
from .process import *

try:
    __version__ = version("pynapple")
except PackageNotFoundError:
    # package is not installed
    pass
