"""Cooperative cancellation shared by analysis and kinematics."""


class AnalysisCancelled(RuntimeError):
    """Raised when an analysis or kinematics operation is cancelled."""


def check_cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelled("Analysis was cancelled")
