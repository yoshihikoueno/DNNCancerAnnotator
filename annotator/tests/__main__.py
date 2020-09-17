'''
execute tests
'''

# built-in
import os
import unittest

# custom
from .test_region_metrics import TestRegionMetricsSingleThreshold
from .test_region_metrics import TestRegionMetricsSingleThresholdShrinked
from .test_region_metrics import TestRegionMetricsMultiThreshold
from .test_region_metrics import TestRegionMetricsMultiThresholdShrinked


if __name__ == '__main__':
    unittest.main()
