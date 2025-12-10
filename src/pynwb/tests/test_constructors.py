"""Test in-memory Python API constructors for ndx-spatial-transformation extension."""

import numpy as np

from pynwb.testing import TestCase

from ndx_spatial_transformation import (
    RigidTransformation,
    SimilarityTransformation,
    Landmarks,
    SpatialTransformationMetadata,
)

ROTATION_MATRIX_2D = np.array(
    [[-0.00032677112826650533, 0.9755835811025647], [-0.9755835811025648, -0.00032677112826617975]]
)
TRANSLATION_VECTOR_2D = [57.227564641852496, 615.2575908529723]
SCALE = 0.9755836358284588


class TestRigidTransformationConstructor(TestCase):
    """Unit tests for RigidTransformation constructor."""

    def setUp(self):
        """Set up example parameters for tests."""
        self.rotation_matrix = ROTATION_MATRIX_2D
        self.translation = TRANSLATION_VECTOR_2D
        self.center_of_rotation = np.array([0.0, 0.0])

    def test_constructor_basic(self):
        """RigidTransformation can be constructed with rotation matrix and translation vector."""
        rt = RigidTransformation(
            name="rigid_basic",
            rotation_matrix=self.rotation_matrix,
            translation_vector=self.translation,
        )

        self.assertEqual(rt.name, "rigid_basic")
        np.testing.assert_array_equal(rt.rotation_matrix, self.rotation_matrix)
        np.testing.assert_array_equal(rt.translation_vector, self.translation)

    def test_constructor_with_center_of_rotation(self):
        """RigidTransformation can be constructed with optional center_of_rotation."""
        rt = RigidTransformation(
            name="rigid_with_center",
            rotation_matrix=self.rotation_matrix,
            translation_vector=self.translation,
            center_of_rotation=self.center_of_rotation,
        )

        self.assertEqual(rt.name, "rigid_with_center")
        np.testing.assert_array_equal(rt.rotation_matrix, self.rotation_matrix)
        np.testing.assert_array_equal(rt.center_of_rotation, self.center_of_rotation)


class TestSimilarityTransformationConstructor(TestCase):
    """Unit tests for SimilarityTransformation constructor."""

    def setUp(self):
        """Set up example parameters for tests."""
        self.rotation_matrix = ROTATION_MATRIX_2D
        self.translation = TRANSLATION_VECTOR_2D
        self.scale = SCALE
        self.center_of_rotation = np.array([0.0, 0.0])

    def test_constructor_basic(self):
        """SimilarityTransformation can be constructed with rotation matrix, translation, and scale."""
        st = SimilarityTransformation(
            name="similarity_basic",
            rotation_matrix=self.rotation_matrix,
            translation_vector=self.translation,
            scale=self.scale,
        )

        self.assertEqual(st.name, "similarity_basic")
        np.testing.assert_array_equal(st.rotation_matrix, self.rotation_matrix)
        np.testing.assert_array_equal(st.translation_vector, self.translation)
        self.assertEqual(st.scale, self.scale)

    def test_constructor_inherits_from_rigid(self):
        """SimilarityTransformation inherits all RigidTransformation attributes."""
        st = SimilarityTransformation(
            name="similarity_with_center",
            rotation_matrix=self.rotation_matrix,
            translation_vector=self.translation,
            center_of_rotation=self.center_of_rotation,
            scale=self.scale,
        )

        np.testing.assert_array_equal(st.center_of_rotation, self.center_of_rotation)
        self.assertEqual(st.scale, self.scale)


class TestLandmarksConstructor(TestCase):
    """Unit tests for Landmarks constructor."""

    def setUp(self):
        """Set up example parameters for tests."""

        self.rotation_matrix = ROTATION_MATRIX_2D
        self.translation = TRANSLATION_VECTOR_2D
        self.source_coordinates = np.array([[0.0, 0.0], [5.0, 5.0]])
        self.target_coordinates = np.array([[10.0, 20.0], [15.0, 25.0]])
        self.center_of_rotation = np.array([0.0, 0.0])
        self.scale = SCALE

    def test_constructor_basic(self):
        """Landmarks can be constructed with minimal required data."""
        lm = Landmarks(
            name="landmarks_basic",
            description="Basic landmarks test",
        )
        for coord in self.source_coordinates:
            lm.add_row(source_coordinates=coord)

        self.assertEqual(lm.name, "landmarks_basic")
        self.assertEqual(len(lm.source_coordinates), len(self.source_coordinates))
        np.testing.assert_array_equal(lm.source_coordinates[0], self.source_coordinates[0])

    def test_constructor_with_target_coordinates(self):
        """Landmarks can include target_coordinates for registered landmarks."""
        lm = Landmarks(
            name="landmarks_with_target",
            description="Landmarks with target coordinates",
        )
        for source_coord, target_coord in zip(self.source_coordinates, self.target_coordinates):
            lm.add_row(
                source_coordinates=source_coord,
                target_coordinates=target_coord,
            )

        self.assertEqual(len(lm.source_coordinates), len(self.source_coordinates))
        np.testing.assert_array_equal(lm.source_coordinates[1], self.source_coordinates[1])
        self.assertEqual(len(lm.target_coordinates), len(self.target_coordinates))
        np.testing.assert_array_equal(lm.target_coordinates[1], self.target_coordinates[1])

    def test_constructor_with_transformation_link(self):
        """Landmarks can be linked to a SpatialTransformation."""
        rt = RigidTransformation(
            name="rigid_for_landmarks",
            rotation_matrix=self.rotation_matrix,
            translation_vector=self.translation,
        )
        lm = Landmarks(
            name="landmarks_linked",
            description="Landmarks with transformation link",
            transformation=rt,
        )
        lm.add_row(source_coordinates=self.source_coordinates)

        self.assertIsNotNone(lm.transformation)
        self.assertEqual(lm.transformation.name, "rigid_for_landmarks")

    def test_constructor_with_labels_and_confidence(self):
        """Landmarks can include landmark_labels and confidence scores."""
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

        self.assertEqual(len(lm.landmark_labels), 2)
        self.assertEqual(lm.landmark_labels[0], "Bregma")
        self.assertEqual(lm.landmark_labels[1], "Lambda")
        self.assertAlmostEqual(lm.confidence[0], 0.95)
        self.assertAlmostEqual(lm.confidence[1], 0.87)

    def test_constructor_with_source_image(self):
        """Landmarks can be linked to a source image."""
        from pynwb.image import GrayscaleImage

        source_img = GrayscaleImage(
            name="source_image",
            data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            description="Source image for landmarks",
        )

        lm = Landmarks(
            name="landmarks_with_source_image",
            description="Landmarks with source image link",
            source_image=source_img,
        )
        lm.add_row(source_coordinates=np.array([10.0, 20.0]))
        lm.add_row(source_coordinates=np.array([30.0, 40.0]))

        self.assertIsNotNone(lm.source_image)
        self.assertEqual(lm.source_image.name, "source_image")
        self.assertEqual(lm.source_image.data.shape, (100, 100))

    def test_constructor_with_target_image(self):
        """Landmarks can be linked to a target image."""
        from pynwb.image import GrayscaleImage

        target_img = GrayscaleImage(
            name="target_image",
            data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            description="Target image for landmarks",
        )

        lm = Landmarks(
            name="landmarks_with_target_image",
            description="Landmarks with target image link",
            target_image=target_img,
        )
        lm.add_row(source_coordinates=np.array([10.0, 20.0]))

        self.assertIsNotNone(lm.target_image)
        self.assertEqual(lm.target_image.name, "target_image")

    def test_constructor_with_both_images(self):
        """Landmarks can be linked to both source and target images."""
        from pynwb.image import GrayscaleImage

        source_img = GrayscaleImage(
            name="source_image",
            data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            description="Source image",
        )

        target_img = GrayscaleImage(
            name="target_image",
            data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            description="Target image",
        )

        lm = Landmarks(
            name="landmarks_with_both_images",
            description="Landmarks with both image links",
            source_image=source_img,
            target_image=target_img,
        )
        lm.add_row(
            source_coordinates=np.array([10.0, 20.0]),
            target_coordinates=np.array([15.0, 25.0]),
        )
        lm.add_row(
            source_coordinates=np.array([50.0, 60.0]),
            target_coordinates=np.array([55.0, 65.0]),
        )

        self.assertIsNotNone(lm.source_image)
        self.assertIsNotNone(lm.target_image)
        self.assertEqual(lm.source_image.name, "source_image")
        self.assertEqual(lm.target_image.name, "target_image")

    def test_constructor_with_transformation_and_images(self):
        """Landmarks can be linked to transformation and images simultaneously."""
        from pynwb.image import GrayscaleImage

        rt = RigidTransformation(
            name="rigid_transform",
            rotation_matrix=np.array(
                [[-0.00032677112826650533, 0.9755835811025647], [-0.9755835811025648, -0.00032677112826617975]]
            ),
            translation_vector=[57.227564641852496, 615.2575908529723],
        )

        source_img = GrayscaleImage(
            name="source_image",
            data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            description="Source image",
        )

        target_img = GrayscaleImage(
            name="target_image",
            data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            description="Target image",
        )

        lm = Landmarks(
            name="landmarks_complete_links",
            description="Landmarks with all links",
            transformation=rt,
            source_image=source_img,
            target_image=target_img,
        )
        lm.add_row(
            source_coordinates=np.array([10.0, 20.0]),
            target_coordinates=np.array([15.0, 25.0]),
            landmark_labels="Point1",
            confidence=0.95,
        )

        self.assertIsNotNone(lm.transformation)
        self.assertIsNotNone(lm.source_image)
        self.assertIsNotNone(lm.target_image)
        self.assertEqual(lm.transformation.name, "rigid_transform")
        self.assertEqual(lm.source_image.name, "source_image")
        self.assertEqual(lm.target_image.name, "target_image")


class TestSpatialTransformationMetadataConstructor(TestCase):
    """Unit tests for SpatialTransformationMetadata constructor."""

    def setUp(self):
        """Set up common test objects."""
        self.rt = RigidTransformation(
            name="rigid_1",
            rotation_matrix=ROTATION_MATRIX_2D,
            translation_vector=TRANSLATION_VECTOR_2D,
        )
        self.st = SimilarityTransformation(
            name="similarity_1",
            rotation_matrix=ROTATION_MATRIX_2D,
            translation_vector=TRANSLATION_VECTOR_2D,
            scale=SCALE,
        )
        self.lm = Landmarks(name="landmarks_1", description="Test landmarks")
        self.lm.add_row(source_coordinates=np.array([0.0, 0.0]))

    def test_constructor_empty(self):
        """SpatialTransformationMetadata can be constructed empty."""
        meta = SpatialTransformationMetadata(name="empty_meta")

        self.assertEqual(meta.name, "empty_meta")

    def test_add_single_transformation(self):
        """SpatialTransformationMetadata can hold a single transformation."""
        meta = SpatialTransformationMetadata(name="single_transform_meta")
        meta.add_spatial_transformations(spatial_transformations=self.rt)

        self.assertEqual(len(meta.spatial_transformations), 1)
        self.assertIn("rigid_1", meta.spatial_transformations)

    def test_add_multiple_transformations(self):
        """SpatialTransformationMetadata can hold multiple transformations."""
        meta = SpatialTransformationMetadata(name="multi_transform_meta")
        meta.add_spatial_transformations(spatial_transformations=self.rt)
        meta.add_spatial_transformations(spatial_transformations=self.st)

        self.assertEqual(len(meta.spatial_transformations), 2)
        self.assertIn("rigid_1", meta.spatial_transformations)
        self.assertIn("similarity_1", meta.spatial_transformations)
        # Verify types
        self.assertIsInstance(meta.spatial_transformations["rigid_1"], RigidTransformation)
        self.assertIsInstance(meta.spatial_transformations["similarity_1"], SimilarityTransformation)

    def test_add_single_landmarks(self):
        """SpatialTransformationMetadata can hold a single landmark set."""
        meta = SpatialTransformationMetadata(name="single_landmarks_meta")
        meta.add_landmarks(landmarks=self.lm)

        self.assertEqual(len(meta.landmarks), 1)
        self.assertIn("landmarks_1", meta.landmarks)

    def test_add_multiple_landmarks(self):
        """SpatialTransformationMetadata can hold multiple landmark sets."""
        meta = SpatialTransformationMetadata(name="multi_landmarks_meta")

        lm2 = Landmarks(name="landmarks_2", description="Second set")
        lm2.add_row(source_coordinates=np.array([1.0, 1.0]))

        meta.add_landmarks(landmarks=self.lm)
        meta.add_landmarks(landmarks=lm2)

        self.assertEqual(len(meta.landmarks), 2)
        self.assertIn("landmarks_1", meta.landmarks)
        self.assertIn("landmarks_2", meta.landmarks)

    def test_complete_metadata_container(self):
        """SpatialTransformationMetadata can hold both transformations and landmarks."""
        meta = SpatialTransformationMetadata(name="complete_meta")

        lm = Landmarks(
            name="landmarks",
            description="Test landmarks",
            transformation=self.rt,
        )
        lm.add_row(source_coordinates=np.array([0.0, 0.0]))

        meta.add_spatial_transformations(spatial_transformations=self.rt)
        meta.add_landmarks(landmarks=lm)

        self.assertEqual(len(meta.spatial_transformations), 1)
        self.assertEqual(len(meta.landmarks), 1)
