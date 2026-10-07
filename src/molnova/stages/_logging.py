"""Signal real work to the supervisor without changing standalone stage output."""
import os

WORK_STARTED = "__MOLNOVA_WORK_STARTED__"


def work_started(message):
    if os.environ.get("MOLNOVA_SUPERVISED") == "1":
        print(WORK_STARTED, flush=True)
    print(message, flush=True)
