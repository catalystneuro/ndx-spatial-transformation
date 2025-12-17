# -*- coding: utf-8 -*-
from pathlib import Path

from pynwb.spec import NWBNamespaceBuilder, export_spec


def main():
    """Generate namespace and extensions YAML for ndx-spatial-transformation.

    This script is typically used once during development to create or update
    the YAML specification files under the top-level ``spec/`` directory.
    The actual schema is now maintained directly in those YAML files, so this
    script only needs to (re)export them if you decide to manage the spec via
    the PyNWB Spec API instead of raw YAML.
    """

    # Namespace metadata (kept in sync with ``spec/ndx-spatial-transformation.namespace.yaml``)
    ns_builder = NWBNamespaceBuilder(
        name="ndx-spatial-transformation",
        version="0.1.0",
        doc="An extension to store data on spatial transformation",
        author=[
            "Alessandra Trapani",
            "Szonja Weigl",
        ],
        contact=[
            "alessandra.trapani@catalystneuro.com",
            "szonja.weigl@catalystneuro.com",
        ],
    )

    # Include the core NWB namespace so we can derive from core types
    ns_builder.include_namespace("core")

    # NOTE:
    # The detailed data type definitions for SpatialTransformation,
    # RigidTransformation, SimilarityTransformation, AffineTransformation,
    # Landmarks, and SpatialTransformationMetadata are defined directly in
    # ``spec/ndx-spatial-transformation.extensions.yaml``. If you wish to
    # build the spec programmatically instead, you could use NWBGroupSpec,
    # NWBDatasetSpec, etc., and then call ``export_spec`` as shown below.

    # When/if using the programmatic API, populate this list with the root
    # data type specs you define, for example:
    # from pynwb.spec import NWBGroupSpec
    # spatial_transformation = NWBGroupSpec(...)
    # new_data_types = [spatial_transformation, ...]
    new_data_types = []

    # Export the spec to YAML files in the top-level ``spec`` folder
    output_dir = str((Path(__file__).parent.parent.parent / "spec").absolute())
    export_spec(ns_builder, new_data_types, output_dir)


if __name__ == "__main__":
    # usage: python src/spec/create_extension_spec.py
    main()
