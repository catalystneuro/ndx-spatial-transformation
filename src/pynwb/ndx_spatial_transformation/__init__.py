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
SimilarityTransformation = get_class("SimilarityTransformation", "ndx-spatial-transformation")
AffineTransformation = get_class("AffineTransformation", "ndx-spatial-transformation")
Landmarks = get_class("Landmarks", "ndx-spatial-transformation")
SpatialTransformationMetadata = get_class("SpatialTransformationMetadata", "ndx-spatial-transformation")

# Remove these functions/modules from the package
del load_namespaces, get_class, files, __location_of_this_file, __spec_path
