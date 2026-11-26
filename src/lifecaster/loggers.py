import getpass
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

from lightning.pytorch.loggers import TensorBoardLogger

try:
    from loguru import logger
except:
    print("Could not load loguru")


def configure_logger(log_file_path: Path) -> None:
    """
    Configures the logger by removing the default handler and adding file and console handlers.
    Sets up a global exception hook to log uncaught exceptions with their descriptions and prints
    the full traceback to the terminal.

    Args:
        log_file_path (Path): The full path to the log file.
    """

    # Remove the default handler.
    logger.remove()

    # Log to file with rotation and INFO level.
    logger.add(log_file_path, rotation="100 MB", level="INFO")

    # Log to console with INFO level.
    logger.add(sys.stdout, level="INFO")

    # Log the script start time and user
    user = getpass.getuser()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info(f"Script started by user '{user}' at {start_time_str}")

    def handle_exception(
        exc_type: type, exc_value: BaseException, exc_traceback
    ) -> None:
        """
        Logs uncaught exceptions as errors with descriptions and prints the full traceback.

        Args:
            exc_type (type): The exception class.
            exc_value (BaseException): The exception instance.
            exc_traceback (traceback): The traceback object.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.opt(exception=(exc_type, exc_value, exc_traceback)).error(
            "Uncaught exception occurred: {}", exc_value
        )
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    # Set the global exception hook.
    sys.excepthook = handle_exception


class RetryOrSkipTensorBoardLogger(TensorBoardLogger):
    def __init__(
        self,
        *args: Any,
        max_retries: int = 10,
        retry_delay: float = 5.0,
        max_skips: int = 5,
        **kwargs: Any,
    ):
        """
        A TensorBoard logger that retries on I/O errors and skips after repeated failures.

        Args:
            *args: Passed to TensorBoardLogger (root_dir, name, version, etc.).
            max_retries (int): How many times to retry on OSError.
            retry_delay (float): Seconds to wait between retries.
            max_skips (int): How many total skips allowed before raising.
            **kwargs: Forwarded to SummaryWriter.
        """
        super().__init__(*args, **kwargs)
        self.max_retries: int = max_retries
        self.retry_delay: float = retry_delay
        self.max_skips: int = max_skips
        self._skips: int = 0

    def _reset_experiment(self) -> None:
        """
        Tear down the current writer so that `self.experiment` will
        re-create a fresh SummaryWriter on next access.
        """
        try:
            self.experiment.close()
        except Exception:
            pass
        self._experiment = None  # type: ignore

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log scalar metrics with retry on OSError (or ValueError caused by OSError)
        and skip logic.

        Args:
            metrics (Dict[str, float]): Metric names to values.
            step (int): Global step.
        """
        if self._skips >= self.max_skips:
            raise RuntimeError(
                f"Exceeded max skips ({self.max_skips}); aborting logging."
            )
        original_metrics, original_step = metrics, step

        for attempt in range(1, self.max_retries + 1):
            try:
                super().log_metrics(original_metrics, original_step)
                return
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # unwrap Lightning's ValueError wrapper
                cause = getattr(e, "__cause__", None)
                is_io = isinstance(e, OSError) or isinstance(cause, OSError)
                if not is_io:
                    # some other problem (e.g. wrong dtype) — propagate immediately
                    raise
                # it's an I/O failure: retry or skip
                if attempt < self.max_retries:
                    print(
                        f"[TBLogger] I/O failure {attempt}/{self.max_retries}, retrying in {self.retry_delay}s…"
                    )
                    self._reset_experiment()
                    time.sleep(self.retry_delay)
                else:
                    self._skips += 1
                    print(
                        f"[TBLogger] Skipping log {self._skips}/{self.max_skips} after {self.max_retries} I/O failures."
                    )
                    self._reset_experiment()
                    return
