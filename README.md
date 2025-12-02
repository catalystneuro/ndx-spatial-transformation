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
from pynwb.base import Images
from pynwb.image import GrayscaleImage
from skimage.transform import SimilarityTransform
from wfield import im_apply_transform

from ndx_spatial_transformation import (
    SimilarityTransformation,
    SpatialTransformationMetadata,
    Landmarks,
)

# Create NWB file
nwbfile = NWBFile(
    session_description="Demonstration of spatial transformation extension",
    identifier="spatial-transformation-example",
    session_start_time=datetime(2024, 1, 1, tzinfo=tzutc()),
)

# Define transformation parameters
rotation_angle = -1.5711312761753244  # radians (counter-clockwise)
translation_vector = np.array([57.23, 615.26])
scale_factor = 0.9755836358284588

# Create similarity transformation object
similarity_transformation = SimilarityTransformation(
    name="SimilarityTransformation",
    rotation_angle=rotation_angle,
    translation_vector=translation_vector,
    scale=scale_factor,
)

# Define transformation matrix
transform_matrix = np.array([
    [-3.27e-04, 9.76e-01, 5.72e+01],
    [-9.76e-01, -3.27e-04, 6.15e+02],
    [0.00e+00, 0.00e+00, 1.00e+00]
])

# Generate example images
source_image_data = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
M = SimilarityTransform(transform_matrix)
target_image_data = im_apply_transform(im=source_image_data, M=M)

# Wrap images in NWB containers
source_image = GrayscaleImage(
    name="SourceImage",
    description="Original widefield imaging frame",
    data=source_image_data,
)

target_image = GrayscaleImage(
    name="TargetImage",
    description="Transformed frame aligned to Allen CCF coordinates",
    data=target_image_data,
)

# Add images to NWB file
images = Images(
    name="Images",
    images=[source_image, target_image],
    description="Source and target images showing spatial transformation",
)
nwbfile.add_acquisition(images)

# Define anatomical landmarks used for alignment
landmarks_definitions = {
    "source_coordinates": {
        "x": [137.337774, 150.764796, 140.903437, 493.558386],
        "y": [381.428925, 302.164284, 226.593808, 301.863883],
        "name": ["OB_left","OB_center","OB_right", "RSP_base"],
    },
    "target_coordinates":{
        "x": [219.484536, 320.000000, 420.515464, 320.000000],
        "y": [92.164948, 92.164948, 92.164948, 434.948454],
        "name": ["OB_left","OB_center","OB_right", "RSP_base"],
    },
}

# Create landmarks table
landmarks_table = Landmarks(
    name="Landmarks",
    description="Anatomical landmarks for Allen CCF alignment",
    source_image=source_image,
    target_image=target_image,
)

# Populate landmarks table
for i in range(4):
    landmarks_table.add_row(
        source_coordinates=[
            landmarks_definitions["source_coordinates"]["x"][i],
            landmarks_definitions["source_coordinates"]["y"][i],
        ],
        target_coordinates=[
            landmarks_definitions["target_coordinates"]["x"][i],
            landmarks_definitions["target_coordinates"]["y"][i],
        ],
        landmark_labels=landmarks_definitions["source_coordinates"]["name"][i],
    )

# Add metadata to NWB file
spatial_metadata = SpatialTransformationMetadata(
    name="SpatialTransformationMetadata"
)
spatial_metadata.add_spatial_transformations(
    spatial_transformations=similarity_transformation
)
spatial_metadata.add_landmarks(landmarks=landmarks_table)
nwbfile.add_lab_meta_data(spatial_metadata)

# Write to file
with NWBHDF5IO("spatial_transformation_example.nwb", "w") as io:
    io.write(nwbfile)
    print("✓ NWB file created successfully")
```

---
This extension was created using [ndx-template](https://github.com/nwb-extensions/ndx-template).
