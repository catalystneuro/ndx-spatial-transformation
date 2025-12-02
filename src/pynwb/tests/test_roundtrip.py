"""Roundtrip tests for ndx-spatial-transformation data types.

This module tests that all spatial transformation data types can be added to an NWBFile,
written to disk, and read back correctly with all data preserved.
"""

from datetime import datetime
import numpy as np
from dateutil.tz import tzutc

from pynwb import NWBHDF5IO, NWBFile
from pynwb.testing import TestCase, remove_test_file

from ndx_spatial_transformation import (
    RigidTransformation,
    SimilarityTransformation,
    Landmarks,
    SpatialTransformationMetadata,
)


class TestRigidTransformationRoundtrip(TestCase):
    """Roundtrip tests for RigidTransformation in an NWBFile."""

    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="test session for rigid transformation",
            identifier="rigid-test",
            session_start_time=datetime(2024, 1, 1, tzinfo=tzutc()),
        )
        self.path = "test_rigid_transformation_roundtrip.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """RigidTransformation can be written and read back."""
        rotation_angle = np.pi / 4  # 45 degrees
        translation = np.array([10.0, 20.0])

        rt = RigidTransformation(
            name="rigid",
            rotation_angle=rotation_angle,
            translation_vector=translation,
        )

        meta = SpatialTransformationMetadata(
            name="spatial_meta",
        )
        meta.add_spatial_transformations(spatial_transformations=rt)
        self.nwbfile.add_lab_meta_data(meta)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_meta = read_nwbfile.lab_meta_data["spatial_meta"]
            self.assertIsInstance(read_meta, SpatialTransformationMetadata)

            read_rt = read_meta.spatial_transformations["rigid"]
            self.assertEqual(read_rt.rotation_angle, rotation_angle)
            np.testing.assert_array_equal(read_rt.translation_vector, translation)


class TestSimilarityTransformationRoundtrip(TestCase):
    """Roundtrip tests for SimilarityTransformation in an NWBFile."""

    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="test session for similarity transformation",
            identifier="similarity-test",
            session_start_time=datetime(2024, 1, 1),
        )
        self.path = "test_similarity_transformation_roundtrip.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """SimilarityTransformation can be written and read back."""
        rotation_angle = np.pi / 6  # 30 degrees
        translation = np.array([5.0, 10.0])
        scale = 1.5

        st = SimilarityTransformation(
            name="similarity",
            rotation_angle=rotation_angle,
            translation_vector=translation,
            scale=scale,
        )

        meta = SpatialTransformationMetadata(name="spatial_meta_similarity")
        meta.add_spatial_transformations(spatial_transformations=st)
        self.nwbfile.add_lab_meta_data(meta)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_meta = read_nwbfile.lab_meta_data["spatial_meta_similarity"]
            read_st = read_meta.spatial_transformations["similarity"]

            self.assertEqual(read_st.rotation_angle, rotation_angle)
            np.testing.assert_array_equal(read_st.translation_vector, translation)
            self.assertEqual(read_st.scale, scale)


class TestLandmarksRoundtrip(TestCase):
    """Roundtrip tests for Landmarks in an NWBFile."""

    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="test session for landmarks",
            identifier="landmarks-test",
            session_start_time=datetime(2024, 1, 1),
        )
        self.path = "test_landmarks_roundtrip.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip_basic(self):
        """Landmarks with source_coordinates can be written and read back."""
        lm = Landmarks(
            name="landmarks",
            description="Basic landmarks for testing",
        )
        lm.add_row(source_coordinates=np.array([0.0, 0.0]))
        lm.add_row(source_coordinates=np.array([10.0, 10.0]))
        lm.add_row(source_coordinates=np.array([20.0, 20.0]))

        meta = SpatialTransformationMetadata(name="spatial_meta_landmarks")
        meta.add_landmarks(landmarks=lm)
        self.nwbfile.add_lab_meta_data(meta)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_meta = read_nwbfile.lab_meta_data["spatial_meta_landmarks"]
            read_lm = read_meta.landmarks["landmarks"]

            self.assertEqual(len(read_lm.source_coordinates), 3)
            np.testing.assert_array_equal(read_lm.source_coordinates[0], np.array([0.0, 0.0]))
            np.testing.assert_array_equal(read_lm.source_coordinates[1], np.array([10.0, 10.0]))

    def test_roundtrip_with_target_coordinates(self):
        """Landmarks with target_coordinates can be written and read back."""
        lm = Landmarks(
            name="landmarks_with_target",
            description="Landmarks with target coordinates",
        )
        lm.add_row(
            source_coordinates=np.array([0.0, 0.0]),
            target_coordinates=np.array([100.0, 100.0]),
        )
        lm.add_row(
            source_coordinates=np.array([10.0, 20.0]),
            target_coordinates=np.array([110.0, 120.0]),
        )

        meta = SpatialTransformationMetadata(name="spatial_meta_landmarks_target")
        meta.add_landmarks(landmarks=lm)
        self.nwbfile.add_lab_meta_data(meta)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_meta = read_nwbfile.lab_meta_data["spatial_meta_landmarks_target"]
            read_lm = read_meta.landmarks["landmarks_with_target"]

            self.assertEqual(len(read_lm.source_coordinates), 2)
            self.assertEqual(len(read_lm.target_coordinates), 2)
            np.testing.assert_array_equal(read_lm.source_coordinates[0], np.array([0.0, 0.0]))
            np.testing.assert_array_equal(read_lm.target_coordinates[0], np.array([100.0, 100.0]))

    def test_roundtrip_with_labels_and_confidence(self):
        """Landmarks with labels and confidence can be written and read back."""
        lm = Landmarks(
            name="landmarks_detailed",
            description="Detailed landmarks",
        )
        lm.add_row(
            source_coordinates=np.array([0.0, 0.0]),
            landmark_labels="Bregma",
            confidence=0.95,
        )
        lm.add_row(
            source_coordinates=np.array([10.0, 20.0]),
            landmark_labels="Lambda",
            confidence=0.87,
        )
        lm.add_row(
            source_coordinates=np.array([20.0, 30.0]),
            landmark_labels="Interaural",
            confidence=0.92,
        )

        meta = SpatialTransformationMetadata(name="spatial_meta_landmarks_detailed")
        meta.add_landmarks(landmarks=lm)
        self.nwbfile.add_lab_meta_data(meta)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_meta = read_nwbfile.lab_meta_data["spatial_meta_landmarks_detailed"]
            read_lm = read_meta.landmarks["landmarks_detailed"]

            self.assertEqual(len(read_lm.landmark_labels), 3)
            self.assertEqual(read_lm.landmark_labels[0], "Bregma")
            self.assertEqual(read_lm.landmark_labels[1], "Lambda")
            self.assertEqual(read_lm.confidence[0], 0.95)
            self.assertEqual(read_lm.confidence[1], 0.87)

    def test_roundtrip_with_transformation_link(self):
        """Landmarks linked to a transformation can be written and read back."""
        rotation_angle = 0.0
        translation = np.array([10.0, 20.0])
        rt = RigidTransformation(
            name="rigid_for_landmarks",
            rotation_angle=rotation_angle,
            translation_vector=translation,
        )

        lm = Landmarks(
            name="landmarks_linked",
            description="Landmarks linked to transformation",
            transformation=rt,
        )
        lm.add_row(source_coordinates=np.array([0.0, 0.0]))
        lm.add_row(source_coordinates=np.array([10.0, 10.0]))

        meta = SpatialTransformationMetadata(name="spatial_meta_landmarks_linked")
        meta.add_spatial_transformations(spatial_transformations=rt)
        meta.add_landmarks(landmarks=lm)
        self.nwbfile.add_lab_meta_data(meta)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_meta = read_nwbfile.lab_meta_data["spatial_meta_landmarks_linked"]
            read_lm = read_meta.landmarks["landmarks_linked"]

            self.assertIsNotNone(read_lm.transformation)
            self.assertEqual(read_lm.transformation.name, "rigid_for_landmarks")
            self.assertEqual(read_lm.transformation.rotation_angle, rotation_angle)

    def test_roundtrip_all_fields(self):
        """Landmarks with all optional fields can be written and read back."""
        rt = RigidTransformation(
            name="transform_for_complete",
            rotation_angle=np.pi / 4,
            translation_vector=np.array([5.0, 10.0]),
        )

        lm = Landmarks(
            name="landmarks_complete",
            description="Complete landmarks with all fields",
            transformation=rt,
        )
        lm.add_row(
            source_coordinates=np.array([0.0, 0.0]),
            target_coordinates=np.array([10.0, 10.0]),
            landmark_labels="Point1",
            confidence=0.99,
        )
        lm.add_row(
            source_coordinates=np.array([5.0, 5.0]),
            target_coordinates=np.array([15.0, 15.0]),
            landmark_labels="Point2",
            confidence=0.85,
        )

        meta = SpatialTransformationMetadata(name="spatial_meta_complete")
        meta.add_spatial_transformations(spatial_transformations=rt)
        meta.add_landmarks(landmarks=lm)
        self.nwbfile.add_lab_meta_data(meta)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_meta = read_nwbfile.lab_meta_data["spatial_meta_complete"]
            read_lm = read_meta.landmarks["landmarks_complete"]

            self.assertEqual(len(read_lm.source_coordinates), 2)
            self.assertEqual(len(read_lm.target_coordinates), 2)
            self.assertEqual(len(read_lm.landmark_labels), 2)
            self.assertEqual(len(read_lm.confidence), 2)
            self.assertIsNotNone(read_lm.transformation)

    def test_roundtrip_with_source_image(self):
        """Landmarks linked to source image can be written and read back."""
        from pynwb.image import GrayscaleImage
        from pynwb.base import Images

        source_img = GrayscaleImage(
            name="source_image",
            data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            description="Source image for landmarks",
        )

        # Add image to NWB file so it has a parent
        images = Images(name="LandmarkImages", images=[source_img])
        self.nwbfile.add_acquisition(images)

        lm = Landmarks(
            name="landmarks_with_source_image",
            description="Landmarks with source image",
            source_image=source_img,
        )
        lm.add_row(
            source_coordinates=np.array([10.0, 20.0]),
            landmark_labels="Point1",
        )
        lm.add_row(
            source_coordinates=np.array([30.0, 40.0]),
            landmark_labels="Point2",
        )

        meta = SpatialTransformationMetadata(name="spatial_meta_with_images")
        meta.add_landmarks(landmarks=lm)
        self.nwbfile.add_lab_meta_data(meta)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_meta = read_nwbfile.lab_meta_data["spatial_meta_with_images"]
            read_lm = read_meta.landmarks["landmarks_with_source_image"]

            self.assertIsNotNone(read_lm.source_image)
            self.assertEqual(read_lm.source_image.name, "source_image")
            self.assertEqual(read_lm.source_image.data.shape, (100, 100))

    def test_roundtrip_with_both_images(self):
        """Landmarks linked to both source and target images can be written and read back."""
        from pynwb.image import GrayscaleImage
        from pynwb.base import Images

        source_img = GrayscaleImage(
            name="source_image",
            data=np.random.randint(0, 255, (128, 128), dtype=np.uint8),
            description="Source image",
        )

        target_img = GrayscaleImage(
            name="target_image",
            data=np.random.randint(0, 255, (128, 128), dtype=np.uint8),
            description="Target/registered image",
        )

        # Add images to NWB file
        images = Images(name="LandmarkImages", images=[source_img, target_img])
        self.nwbfile.add_acquisition(images)

        lm = Landmarks(
            name="landmarks_with_both_images",
            description="Landmarks with both images",
            source_image=source_img,
            target_image=target_img,
        )
        lm.add_row(
            source_coordinates=np.array([10.0, 20.0]),
            target_coordinates=np.array([15.0, 25.0]),
            landmark_labels="Bregma",
            confidence=0.95,
        )
        lm.add_row(
            source_coordinates=np.array([50.0, 60.0]),
            target_coordinates=np.array([55.0, 65.0]),
            landmark_labels="Lambda",
            confidence=0.88,
        )

        meta = SpatialTransformationMetadata(name="spatial_meta_both_images")
        meta.add_landmarks(landmarks=lm)
        self.nwbfile.add_lab_meta_data(meta)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_meta = read_nwbfile.lab_meta_data["spatial_meta_both_images"]
            read_lm = read_meta.landmarks["landmarks_with_both_images"]

            # Verify both image links
            self.assertIsNotNone(read_lm.source_image)
            self.assertIsNotNone(read_lm.target_image)
            self.assertEqual(read_lm.source_image.name, "source_image")
            self.assertEqual(read_lm.target_image.name, "target_image")
            self.assertEqual(read_lm.source_image.data.shape, (128, 128))
            self.assertEqual(read_lm.target_image.data.shape, (128, 128))

            # Verify landmarks data is preserved
            self.assertEqual(len(read_lm.source_coordinates), 2)
            self.assertEqual(len(read_lm.target_coordinates), 2)
            np.testing.assert_array_equal(read_lm.source_coordinates[0], np.array([10.0, 20.0]))
            np.testing.assert_array_equal(read_lm.target_coordinates[0], np.array([15.0, 25.0]))

    def test_roundtrip_with_transformation_and_images(self):
        """Landmarks with transformation and image links can be written and read back."""
        from pynwb.image import GrayscaleImage
        from pynwb.base import Images

        rt = RigidTransformation(
            name="rigid_with_images",
            rotation_angle=np.pi / 4,
            translation_vector=np.array([10.0, 20.0]),
        )

        source_img = GrayscaleImage(
            name="source_image",
            data=np.random.randint(0, 255, (200, 200), dtype=np.uint8),
            description="Source image",
        )

        target_img = GrayscaleImage(
            name="target_image",
            data=np.random.randint(0, 255, (200, 200), dtype=np.uint8),
            description="Transformed target image",
        )

        # Add images to NWB file
        images = Images(name="LandmarkImages", images=[source_img, target_img])
        self.nwbfile.add_acquisition(images)

        lm = Landmarks(
            name="landmarks_all_links",
            description="Landmarks with all possible links",
            transformation=rt,
            source_image=source_img,
            target_image=target_img,
        )
        lm.add_row(
            source_coordinates=np.array([50.0, 100.0]),
            target_coordinates=np.array([60.0, 110.0]),
            landmark_labels="Landmark1",
            confidence=0.92,
        )

        meta = SpatialTransformationMetadata(name="spatial_meta_all_links")
        meta.add_spatial_transformations(spatial_transformations=rt)
        meta.add_landmarks(landmarks=lm)
        self.nwbfile.add_lab_meta_data(meta)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_meta = read_nwbfile.lab_meta_data["spatial_meta_all_links"]
            read_lm = read_meta.landmarks["landmarks_all_links"]

            # Verify all links are preserved
            self.assertIsNotNone(read_lm.transformation)
            self.assertIsNotNone(read_lm.source_image)
            self.assertIsNotNone(read_lm.target_image)

            # Verify transformation
            self.assertEqual(read_lm.transformation.name, "rigid_with_images")
            self.assertEqual(read_lm.transformation.rotation_angle, np.pi / 4)

            # Verify images
            self.assertEqual(read_lm.source_image.name, "source_image")
            self.assertEqual(read_lm.target_image.name, "target_image")
            self.assertEqual(read_lm.source_image.data.shape, (200, 200))

            # Verify landmarks data
            self.assertEqual(len(read_lm.source_coordinates), 1)
            self.assertEqual(read_lm.landmark_labels[0], "Landmark1")
            self.assertEqual(read_lm.confidence[0], 0.92)


class TestSpatialTransformationMetadataRoundtrip(TestCase):
    """Roundtrip tests for SpatialTransformationMetadata with multiple objects."""

    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="test session for spatial metadata",
            identifier="spatial-metadata-test",
            session_start_time=datetime(2024, 1, 1),
        )
        self.path = "test_spatial_metadata_roundtrip.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip_multiple_transformations(self):
        """SpatialTransformationMetadata with multiple transformations can be written and read back."""
        rt = RigidTransformation(
            name="rigid_1",
            rotation_angle=np.pi / 4,
            translation_vector=np.array([10.0, 20.0]),
        )
        st = SimilarityTransformation(
            name="similarity_1",
            rotation_angle=np.pi / 6,
            translation_vector=np.array([5.0, 10.0]),
            scale=1.5,
        )

        meta = SpatialTransformationMetadata(name="multi_transform_meta")
        meta.add_spatial_transformations(spatial_transformations=rt)
        meta.add_spatial_transformations(spatial_transformations=st)
        self.nwbfile.add_lab_meta_data(meta)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_meta = read_nwbfile.lab_meta_data["multi_transform_meta"]

            self.assertEqual(len(read_meta.spatial_transformations), 2)
            self.assertIn("rigid_1", read_meta.spatial_transformations)
            self.assertIn("similarity_1", read_meta.spatial_transformations)

    def test_roundtrip_multiple_landmarks(self):
        """SpatialTransformationMetadata with multiple landmark sets can be written and read back."""
        lm1 = Landmarks(name="landmarks_1", description="First set")
        for i in range(3):
            lm1.add_row(source_coordinates=np.array([float(i), float(i * 2)]))

        lm2 = Landmarks(name="landmarks_2", description="Second set")
        for i in range(5):
            lm2.add_row(source_coordinates=np.array([float(i + 10), float(i * 2 + 10)]))

        meta = SpatialTransformationMetadata(name="multi_landmarks_meta")
        meta.add_landmarks(landmarks=lm1)
        meta.add_landmarks(landmarks=lm2)
        self.nwbfile.add_lab_meta_data(meta)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_meta = read_nwbfile.lab_meta_data["multi_landmarks_meta"]

            self.assertEqual(len(read_meta.landmarks), 2)
            self.assertIn("landmarks_1", read_meta.landmarks)
            self.assertIn("landmarks_2", read_meta.landmarks)
            self.assertEqual(len(read_meta.landmarks["landmarks_1"].source_coordinates), 3)
            self.assertEqual(len(read_meta.landmarks["landmarks_2"].source_coordinates), 5)

    def test_roundtrip_mixed_transformation_types(self):
        """SpatialTransformationMetadata with different transformation types can be written and read back."""
        # Create different types of transformations
        rt = RigidTransformation(
            name="rigid_transform",
            rotation_angle=0.0,
            translation_vector=np.array([10.0, 20.0]),
        )

        st = SimilarityTransformation(
            name="similarity_transform",
            rotation_angle=np.pi / 4,
            translation_vector=np.array([5.0, 10.0]),
            scale=1.5,
        )

        # Create landmarks linked to the rigid transformation
        lm = Landmarks(
            name="test_landmarks",
            description="Landmarks for mixed types test",
            transformation=rt,
        )
        lm.add_row(
            source_coordinates=np.array([0.0, 0.0]),
            landmark_labels="Point1",
            confidence=0.9,
        )

        # Add everything to metadata
        meta = SpatialTransformationMetadata(name="mixed_types_meta")
        meta.add_spatial_transformations(spatial_transformations=rt)
        meta.add_spatial_transformations(spatial_transformations=st)
        meta.add_landmarks(landmarks=lm)
        self.nwbfile.add_lab_meta_data(meta)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_meta = read_nwbfile.lab_meta_data["mixed_types_meta"]

            # Verify all transformations are present
            self.assertEqual(len(read_meta.spatial_transformations), 2)

            # Verify specific types
            read_rt = read_meta.spatial_transformations["rigid_transform"]
            read_st = read_meta.spatial_transformations["similarity_transform"]

            self.assertIsInstance(read_rt, RigidTransformation)
            self.assertIsInstance(read_st, SimilarityTransformation)

            # Verify data preservation
            self.assertEqual(read_rt.rotation_angle, 0.0)
            self.assertEqual(read_st.scale, 1.5)

            # Verify landmarks and link
            self.assertEqual(len(read_meta.landmarks), 1)
            read_lm = read_meta.landmarks["test_landmarks"]
            self.assertIsNotNone(read_lm.transformation)
            self.assertEqual(read_lm.transformation.name, "rigid_transform")
