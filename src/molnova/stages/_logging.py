"""Signal real work to the supervisor without changing standalone stage output."""
import contextvars
import datetime
import functools
import os
import time

WORK_STARTED = "__MOLNOVA_WORK_STARTED__"
_timer = contextvars.ContextVar("molnova_stage_timer", default=None)


def work_started(message):
    timing = _timer.get()
    if timing is not None and timing["start"] is None:
        timing["start"] = time.monotonic()
    if os.environ.get("MOLNOVA_SUPERVISED") == "1":
        print(WORK_STARTED, flush=True)
    print(message, flush=True)


def format_elapsed(seconds):
    hours, remainder = divmod(round(max(0, seconds), 1), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours):02d}:{int(minutes):02d}:{seconds:04.1f}'


def job_elapsed(job):
    start = job.get('timeStarted') or job.get('timeCreated')
    end = job.get('statusUpdated')
    if not start or not end:
        return None
    try:
        return max(0, (datetime.datetime.fromisoformat(end.replace('Z', '+00:00')) -
                       datetime.datetime.fromisoformat(start.replace('Z', '+00:00'))).total_seconds())
    except (TypeError, ValueError):
        return None


def timed_stage(label):
    """Report active stage invocation duration, including failures; skip idle polls."""
    def decorate(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            timing = {'start': None}
            token = _timer.set(timing)
            outcome = 'completed'
            try:
                return func(*args, **kwargs)
            except BaseException:
                outcome = 'failed/interrupted'
                raise
            finally:
                _timer.reset(token)
                if timing['start'] is not None:
                    elapsed = format_elapsed(time.monotonic() - timing['start'])
                    print(f'{label} {outcome}; elapsed={elapsed}.', flush=True)
        return wrapped
    return decorate
