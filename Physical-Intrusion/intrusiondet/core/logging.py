"""Allow multiprocessing routines to use logging via queues.

 * listener_configurer configures a multiprocessing process or process in a pool to log to a file.
 * listener_process creates a single process to listen to a log queue.
   A single listener process is needed only.
 * build_logger is a simple routine that maybe phased out with time.
"""
import logging
import logging.handlers
import sys
import traceback
from multiprocessing.queues import Queue
from typing import Callable, Optional

LOG_LEVEL: int = logging.INFO

LOG_FORMAT: str = (
    "%(asctime)s::%(levelname)s::%(name)s::" "%(filename)s::%(lineno)d::%(message)s"
)

LOG_FILENAME: str = "logger.log"


def build_logger(
    log_level: Optional = None,
    log_format: Optional[str] = None,
    name: Optional[str] = None,
) -> logging.Logger:
    """Build a standard library logger instance

    :param log_level: Logging level string
    :param log_format: Logging format string
    :param name: Name of the logger
    :return: A logging instance
    """
    if log_level is None:
        log_level = LOG_LEVEL
    if log_format is None:
        log_format = LOG_FORMAT
    logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger(name)
    return logger


def listener_configurer(log_filename: str) -> None:
    """Configure a logging listener for multiprocessing to a log file.

    Recipe and notes from Python STD LIB cookbook.

    Because you'll want to define the logging configurations for listener and workers, the
    listener and worker process functions take a configurer parameter which is a callable
    for configuring logging for that process. These functions are also passed the queue,
    which they use for communication.

    In practice, you can configure the listener however you want, but note that in this
    simple example, the listener does not apply level or filter logic to received records.
    In practice, you would probably want to do this logic in the worker processes, to avoid
    sending events which would be filtered out between processes.

    :param log_filename: filename of the log file
    :return: None
    """
    root = logging.getLogger()
    handler = logging.handlers.RotatingFileHandler(
        log_filename,
        "a",
        backupCount=10,
        maxBytes=10*1024*1024
    )
    handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    root.addHandler(handler)


def listener_process(queue: Queue, configurer: Callable, *config_args) -> None:
    """Create a dedicated listener process for logging.

    Recipe and notes from Python standard library cookbook.

    To end the process, send a literal None.

    This is the listener process top-level loop: wait for logging events
    (LogRecords)on the queue and handle them, quit when you get a None for a
    LogRecord.

    :param queue: Logging queue
    :param configurer: Callable configure method with no arguments
    :param config_args: Arguments for the configurer
    :return: None
    """
    configurer(*config_args)
    while True:
        try:
            record = queue.get()
            if (
                record is None
            ):  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except (Exception,):
            print("Whoops! Problem:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
