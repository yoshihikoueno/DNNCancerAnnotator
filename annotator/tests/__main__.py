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
from .test_losses import TestPositiveRate


if __name__ == '__main__':
    unittest.main()
