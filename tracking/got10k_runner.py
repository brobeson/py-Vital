"""Run experiments via the GOT10K tool."""

import os.path
import got10k.experiments
import tracking.got10k_vital


def _main():
    """The main entry point of the application."""
    # _run_vot_experiments()
    _run_uav_experiments()


def _run_vot_experiments() -> None:
    """Run VOT experiments."""
    experiment = got10k.experiments.ExperimentVOT(
        os.path.expanduser("~/Videos/vot-got"),
        version=2018,
        experiments="supervised",
        result_dir=os.path.abspath("./results"),
    )
    experiment.repetitions = 5
    tracker = tracking.got10k_vital.Vital(name="Vital")
    experiment.run(tracker)
    experiment.report(["Vital"])


def _run_uav_experiments() -> None:
    """Run UAV experimens."""
    experiment = got10k.experiments.ExperimentUAV123(
        os.path.abspath(os.path.expanduser("~/Videos/uav123")),
        result_dir=os.path.abspath("./results"),
    )
    tracker = tracking.got10k_vital.Vital(name="Vital")
    experiment.run(tracker)
    experiment.report(["Vital"])


if __name__ == "__main__":
    _main()
