# !/usr/bin/env python
# This script is based on https://github.com/bhargavvader/pycobra/blob/master/tests.py
# PLEASE pip install pytest-subtests to get a report per test when using self.subTest()

import warnings

import matplotlib
import pytest

matplotlib.use('agg')
warnings.filterwarnings("ignore", category=FutureWarning)

pytest.main(['-k-slow', '--cov=segmentation_rt', '--cov-report=term-missing'])
