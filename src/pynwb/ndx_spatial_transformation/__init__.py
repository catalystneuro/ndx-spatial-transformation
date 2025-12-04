from importlib.resources import files
from pynwb import load_namespaces, get_class

# Get path to the namespace.yaml file with the expected location when installed not in editable mode
__location_of_this_file = files(__name__)
__spec_path = __location_of_this_file / "spec" / "ndx-spatial-transformation.namespace.yaml"

# If that path does not exist, we are likely running in editable mode. Use the local path instead
if not __spec_path.exists():
    __spec_path = __location_of_this_file.parent.parent.parent / "spec" / "ndx-spatial-transformation.namespace.yaml"

# Load the namespace
load_namespaces(str(__spec_path))

# Get base classes
SpatialTransformation = get_class("SpatialTransformation", "ndx-spatial-transformation")
RigidTransformation = get_class("RigidTransformation", "ndx-spatial-transformation")
Landmarks = get_class("Landmarks", "ndx-spatial-transformation")
SpatialTransformationMetadata = get_class("SpatialTransformationMetadata", "ndx-spatial-transformation")

# Get SimilarityTransformation and add custom method
SimilarityTransformation = get_class("SimilarityTransformation", "ndx-spatial-transformation")


def get_transform_matrix(self):
    """
    Construct and return a skimage.transform.SimilarityTransform for this object.

    - Uses these class attributes from `SimilarityTransformation`:
      - `scale` : float - uniform scale factor.
      - `rotation_angle` : float - rotation in radians.
      - `translation_vector` : sequence of length 2 - translation [tx, ty].

    Returns
    -------
    skimage.transform.SimilarityTransform
        A `SimilarityTransform` instance (2D) whose `params` attribute is the 3x3
        homogeneous transformation matrix. The object also exposes `scale`,
        `rotation`, and `translation` attributes corresponding to the inputs.

    Example
    -------
    >>> transform = SimilarityTransformation(
    ...     name="my_transform",
    ...     rotation_angle=-1.5711312761753244,
    ...     translation_vector=[10.0, 20.0],
    ...     scale=0.5
    ... )
    >>> M = transform.get_transform_matrix()
    >>> M.params  # 3x3 homogeneous matrix
    array([...])
    >>> M.scale
    0.5
    """
    from skimage.transform import SimilarityTransform

    similarity_transform = SimilarityTransform(
        scale=self.scale,
        rotation=self.rotation_angle,
        translation=self.translation_vector,
        dimensionality=2,
    )

    return similarity_transform


# Attach the method to the SimilarityTransformation class
SimilarityTransformation.get_transform_matrix = get_transform_matrix

# TODO: Add all classes to __all__ to make them accessible at the package level
__all__ = [
    "SpatialTransformation",
    "RigidTransformation",
    "SimilarityTransformation",
    "Landmarks",
    "SpatialTransformationMetadata",
]

# Remove these functions/modules from the package
del load_namespaces, get_class, files, get_transform_matrix, __location_of_this_file, __spec_path
