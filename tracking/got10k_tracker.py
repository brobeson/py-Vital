"""Wrap tracking.vital.VitalTracker in a GOT-10k tracker."""

import got10k.trackers
import tracking.vital


class Gotk10Vital(got10k.trackers.Tracker):
    """A GOT-10k wrapper around Vital."""

    def __init__(self, name: str, configuration: dict):
        super().__init__(name, is_deterministic=False)
        self.tracker = tracking.vital.VitalTracker(configuration)

    def init(self, image, box):
        self.tracker.initialize(box, image)

    def update(self, image):
        return self.tracker.find_target(image)
