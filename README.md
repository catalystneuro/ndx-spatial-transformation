# ndx-spatial-transformation Extension for NWB

An extension to store data on spatial transformations, including rigid, similarity, and affine transforms, along with 
landmark tables and a `SpatialTransformationMetadata` container for organizing them.

The specification is defined in `spec/ndx-spatial-transformation.extensions.yaml`
and `spec/ndx-spatial-transformation.namespace.yaml`.

## Installation

You can install the stable version of the extension from PyPI using pip:

```bash
pip install ndx-spatial-transformation
```

If you want to install the development version of the extension you can install it directly from the GitHub repository.
The following command installs the development version of the extension:

```bash
pip install -U git+https://github.com/catalystneuro/ndx-fscv.git
```

## Usage

A minimal example of creating and writing an NWB file with a similarity transformation:

```python
from datetime import datetime
from dateutil.tz import tzutc
import numpy as np
from pynwb import NWBFile, NWBHDF5IO

from ndx_spatial_transformation import SimilarityTransformation, SpatialTransformationMetadata

nwbfile = NWBFile(
    session_description="example session",
    identifier="spatial-example",
    session_start_time=datetime(2024, 1, 1, tzinfo=tzutc()),
)

rotation_2x2 = np.array([
    [-0.0003349493741651515, 0.9999999439044569],
    [-0.9999999439044569, -0.0003349493741651515],
])
translation = np.array([57.227564641852496, 615.2575908529723])
scale = np.array([0.9755836358284588])

similarity_transformation = SimilarityTransformation(
    name="SimilarityTransformation",
    rotation_matrix=rotation_2x2,
    translation_vector=translation,
    scale=scale,
)

spatial_transformation_metadata = SpatialTransformationMetadata(name="SpatialTransformationMetadata")
spatial_transformation_metadata.add_spatial_transformations(spatial_transformations=similarity_transformation)

nwbfile.add_lab_meta_data(spatial_transformation_metadata)

with NWBHDF5IO("spatial_example.nwb", "w") as io:
    io.write(nwbfile)
```

---
This extension was created using [ndx-template](https://github.com/nwb-extensions/ndx-template).
