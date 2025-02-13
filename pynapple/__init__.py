from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _get_version

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
    __version__ = _get_version("pynapple")
except _PackageNotFoundError:
    # package is not installed
    pass
