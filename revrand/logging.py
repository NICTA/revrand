"""Specialised metrics logging."""

from abc import ABC, abstractmethod

class MetricsLogger:
    """
    Logger for metrics data (e.g. timing information, optimisation progress)
    """

    @abstractmethod
    def log_progress(self, params):
        """
        Log an optimisation step.

        This method takes arbitrary key value pairs that represent some
        indicators of optimisation progress.

        Examples
        --------
        >>> mlog.log_progress({'loss': 12.3, 'lambda': 0.5})

        Parameters
        ----------
        params: dict
            Key values to log.
        """
        pass


class NoOpLogger(MetricsLogger):
    """
    Metrics logger that does nothing.
    """

    def log_progress(self, params):
        pass


mlog = NoOpLogger() # Use no-op by default

def set_mlog(logger):
    """
    Set the metrics logger.

    Parameters
    ----------
    logger : MetricsLogger
        A metrics logger to use.
    """
    global mlog
    mlog = logger

def get_mlog():
    """
    Get the metrics logger.

    Returns
    -------
    object
        A MetricsLogger to use.
    """
    global mlog
    return mlog
