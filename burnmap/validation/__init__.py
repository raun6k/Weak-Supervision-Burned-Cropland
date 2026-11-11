"""Map agreement with MODIS and district fire counts.

This package does not import Earth Engine. It reads exported GeoTIFFs
and the district fire-count table.
"""

from burnmap.validation import validate as validate
from burnmap.validation import calibrate as calibrate
