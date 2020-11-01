"""Test the CosineAnnealing class."""

import collections
import unittest
import tracking.domain_adaptation_schedules


Row = collections.namedtuple("Row", ["schedule", "io"])


class SchedulesTest(unittest.TestCase):
    """Test cases for the domain adaptation learning rate schedules."""

    def setUp(self):
        """
        Create the data table used for data driven testing.

        All the test methods use the same data table, so this can be done in the setUp() method.
        """
        self.test_data = [
            # Row(
            #     tracking.domain_adaptation_schedules.CosineAnnealing((0.2, 0.8), 200),
            #     [-0.8, -0.5, -0.2],
            # ),
            # Row(
            #     tracking.domain_adaptation_schedules.InverseCosineAnnealing(
            #         (0.2, 0.8), number_of_epochs=200
            #     ),
            #     [-0.2, -0.5, -0.8],
            # ),
            Row(
                tracking.domain_adaptation_schedules.PadaScheduler(
                    max_epochs=10, lambda_=1.0, alpha=10.0
                ),
                [(0, 0.0), (7, 0.998177), (14, 1.0)]
            ),
            # Row(
            #     tracking.domain_adaptation_schedules.GammaScheduler(
            #         (0.2, 0.8), 0.15, 200
            #     ),
            #     [-0.2, -0.20591, -0.8],
            # ),
            # Row(
            #     tracking.domain_adaptation_schedules.Linear((0.2, 0.8), 200.0),
            #     [-0.2, -0.5, -0.8],
            # ),
            # Row(
            #     tracking.domain_adaptation_schedules.Constant(0.75),
            #     [-0.75, -0.75, -0.75],
            # ),
        ]

    def test_schedule(self):
        """Validate the backward() method of each schedule."""
        for test in self.test_data:
            with self.subTest(schedule=type(test.schedule).__name__):
                for data_point in test.io:
                    with self.subTest(epoch=data_point[0]):
                        self.assertAlmostEqual(
                            test.schedule(data_point[0]), data_point[1], places=5
                        )
