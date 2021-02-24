"""Run UAV123 experiments."""

import os.path
import traceback
import got10k.experiments
import slack_message
import tracking.got10k_tracker

reporter = slack_message.ExperimentReporter(
    source="Laptop", channel="learning-rate-research"
)
reporter.send_message("Starting MDNet UAV123 experiments.")

configurations = [
    {"schedule": "constant"},
    {"grl_direction": "decreasing", "schedule": "arccosine"},
    {"grl_direction": "decreasing", "schedule": "cosine_annealing"},
    {"grl_direction": "decreasing", "schedule": "exponential"},
    {"grl_direction": "decreasing", "schedule": "gamma"},
    {"grl_direction": "decreasing", "schedule": "linear"},
    {"grl_direction": "decreasing", "schedule": "pada"},
    {"grl_direction": "increasing", "schedule": "arccosine"},
    {"grl_direction": "increasing", "schedule": "cosine_annealing"},
    {"grl_direction": "increasing", "schedule": "exponential"},
    {"grl_direction": "increasing", "schedule": "gamma"},
    {"grl_direction": "increasing", "schedule": "linear"},
    {"grl_direction": "increasing", "schedule": "pada"},
]
experiment = got10k.experiments.ExperimentUAV123(os.path.expanduser("~/Videos/uav123"))
tracker_names = []
for configuration in configurations:
    if "grl_direction" in configuration:
        name = configuration["grl_direction"][0:3] + "_" + configuration["schedule"]
    else:
        name = configuration["schedule"]
    tracker_names.append(name)
    tracker = tracking.got10k_tracker.Gotk10Vital(name, configuration)
    reporter.send_message(f"Starting {name} schedule.")
    try:
        experiment.run(tracker)
        reporter.send_message(f"Finished {name} schedule.")
    except Exception as e:
        reporter.send_message(f"{name} schedule threw an exception:\n```{traceback.format_exc()}```")
        traceback.print_exc()
reporter.send_message("Starting results analysis.")
experiment.report(tracker_names)
reporter.send_message("Done analyzing results.")
