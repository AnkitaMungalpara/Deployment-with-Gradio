from typing import Callable

from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:

    def wrap(*args, **kwargs):
        try:
            return task_func(*args, **kwargs)
        except Exception as e:
            log.exception("Exception occured during task execution")
            raise e
        finally:
            log.info("Task completed. check the logs folder for more details.")

    return wrap
