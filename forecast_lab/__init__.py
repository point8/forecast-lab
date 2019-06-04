"""
Tools for building and evaluating time series forecast models based on libraries like `sklearn`, `statsmodels` and `fbprophet`.
"""

__author__ = "Christian Staudt"
__copyright__ = "Copyright 2019, "
__credits__ = ["Christian Staudt", "Christophe Cauet"]
__license__ = "MIT"
__version__ = "0.1"
__email__ = "cstaudt@point-8.de"


from . import core, dummy, metrics, datasets
from forecast_lab.core import *
