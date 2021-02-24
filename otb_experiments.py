"""Run OTB experiments."""

import glob
import json
import statistics
import os.path
import numpy
import PIL.Image
import modules.utils
import tracking.vital

# Load the image paths and the ground truth bounding boxes.
otb_path = os.path.expanduser("~/Videos/otb")
sequence_name = "Deer"
sequence_path = os.path.join(otb_path, sequence_name)
image_paths = glob.glob(os.path.join(sequence_path, "img", "*.jpg"))
image_paths.sort()
image = PIL.Image.open(image_paths[0]).convert("RGB")
with open(os.path.join(sequence_path, "groundtruth_rect.txt"), "r") as groundtruth_file:
    lines = groundtruth_file.readlines()
groundtruth_boxes = [
    numpy.array([int(number) for number in line.strip().split(",")]) for line in lines
]

# Initialize the tracker and track the target.
tracking.vital.set_random_seeds(0)
tracker = tracking.vital.VitalTracker(
    tracking.vital.load_configuration("tracking/options.yaml")
)
tracker.initialize(groundtruth_boxes[0], image)
boxes = []
for i, image_path in enumerate(image_paths[1:]):
    print("Tracking frame", i + 1, end="\r")
    boxes.append(tracker.find_target(PIL.Image.open(image_path).convert("RGB")))

# Check the overlap ratio.
assert len(boxes) == len(groundtruth_boxes) - 1
overlaps = [
    modules.utils.overlap_ratio(box, groundtruth)[0]
    for box, groundtruth in zip(boxes, groundtruth_boxes[1:])
]
print("Mean IOU:", statistics.mean(overlaps))

with open("results/Deer/result_new.json", "w") as json_file:
    json.dump(
        {
            "res": [box.round().tolist() for box in boxes],
            "type": "rect",
            "fps": 0.4723471048713186,
        },
        json_file,
        indent=2,
    )
