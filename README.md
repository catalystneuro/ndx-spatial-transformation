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
pip install -U git+https://github.com/catalystneuro/ndx-spatial-transformation.git
```

## Usage

A minimal example of creating and writing an NWB file with a similarity transformation:

```python
from datetime import datetime
from dateutil.tz import tzutc
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from wfield import im_apply_transform

from ndx_spatial_transformation import SimilarityTransformation, SpatialTransformationMetadata, Landmarks

nwbfile = NWBFile(
    session_description="example session",
    identifier="spatial-example",
    session_start_time=datetime(2024, 1, 1, tzinfo=tzutc()),
)

rotation = -1.5711312761753244 # radiants
translation = np.array([57.227564641852496, 615.2575908529723])
scale = np.array([0.9755836358284588])

similarity_transformation = SimilarityTransformation(
    name="SimilarityTransformation",
    rotation_angle=rotation,
    translation_vector=translation,
    scale=scale,
)

#TODO: add API function to compute the trransformation matrix from parameter
transform_matrix=[[-3.26771128e-04,  9.75583581e-01,  5.72275646e+01],[-9.75583581e-01, -3.26771128e-04,  6.15257591e+02],[ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
    
landmarks_data = {
    "source_coordinates":{
        "x": [137.337774, 150.764796, 140.903437, 493.558386], 
        "y": [381.428925, 302.164284, 226.593808, 301.863883], 
        "name": ["OB_left","OB_center","OB_right", "RSP_base"], 
    "target_coordinates":{
        "x": [219.484536, 320.000000, 420.515464, 320.000000], 
        "y": [92.164948, 92.164948, 92.164948, 434.948454], 
        "name": ["OB_left","OB_center","OB_right", "RSP_base"]}
    }
    }

source_image_data = np.random.rand((512, 512), dtype=np.uint8)
target_image_data = im_apply_transform(
    image=source_image_data,
    transform_matrix=transform_matrix,
)

landmarks_table = Landmarks(
    name="Landmarks",
    source_image=source_image_data,
    target_image=target_image_data,
)

for landmark in landmarks_data.items():
    landmarks_table.add_row(
        source_x=landmark["source_coordinates"]["x"],
        source_y=landmark["source_coordinates"]["y"],
        target_x=landmark["target_coordinates"]["x"],
        target_y=landmark["target_coordinates"]["y"],
        name=landmark["source_coordinates"]["name"],
    )
    

spatial_transformation_metadata = SpatialTransformationMetadata(name="SpatialTransformationMetadata")
spatial_transformation_metadata.add_spatial_transformations(spatial_transformations=similarity_transformation)
spatial_transformation_metadata.add_landmarks(landmarks=landmarks_table)

nwbfile.add_lab_meta_data(spatial_transformation_metadata)

with NWBHDF5IO("spatial_example.nwb", "w") as io:
    io.write(nwbfile)
```

---
This extension was created using [ndx-template](https://github.com/nwb-extensions/ndx-template).
