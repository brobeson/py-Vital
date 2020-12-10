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
            Row(
                tracking.domain_adaptation_schedules.Constant(0.23),
                [(0, 0.23), (100, 0.23), (200, 0.23)],
            ),
            Row(
                tracking.domain_adaptation_schedules.IncreasingPada(
                    0.2, 0.8, 1.0, 10.0, 200
                ),
                [(0, 0.2), (100, 0.79197), (200, 0.79995)]
            ),
            Row(
                tracking.domain_adaptation_schedules.IncreasingLinear(0.2, 0.8, 200),
                [(0, 0.2), (100, 0.5), (200, 0.8)],
            ),
            Row(
                tracking.domain_adaptation_schedules.IncreasingCosineAnnealing(
                    0.2, 0.8, 200
                ),
                [(0, 0.2), (100, 0.5), (200, 0.8)],
            ),
            Row(
                tracking.domain_adaptation_schedules.IncreasingGamma(
                    0.2, 0.8, 0.15, 200
                ),
                [(0, 0.2), (100, 0.20591), (200, 0.8)],
            ),
            Row(
                tracking.domain_adaptation_schedules.DecreasingPada(
                    0.2, 0.8, 1.0, 10.0, 200
                ),
                [(0, 0.8), (100, 0.20803), (200, 0.200054)],
            ),
            Row(
                tracking.domain_adaptation_schedules.DecreasingLinear(0.2, 0.8, 200),
                [(0, 0.8), (100, 0.5), (150, 0.35), (200, 0.2)],
            ),
            Row(
                tracking.domain_adaptation_schedules.DecreasingCosineAnnealing(0.2, 0.8, 200),
                [(0, 0.8), (100, 0.5), (200, 0.2)],
            ),
            Row(
                tracking.domain_adaptation_schedules.DecreasingGamma(0.2, 0.8, 0.15, 200),
                [(0, 0.8), (100, 0.79409), (200, 0.2)],
            )
        ]

    def test_schedule(self):
        """Validate the schedule's learning rate."""
        for test in self.test_data:
            with self.subTest(schedule=type(test.schedule).__name__):
                for data_point in test.io:
                    with self.subTest(epoch=data_point[0]):
                        self.assertAlmostEqual(
                            test.schedule(data_point[0]), data_point[1], places=5
                        )
