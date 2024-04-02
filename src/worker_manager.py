from threading import Thread
import time
from typing import Callable
from logging import Logger

from src.worker import Worker


class WorkerManager(Thread):
    """
    WorkerManager watches and manages the lifetime of the Worker to make sure
    the connection to the camera is healthy, as well as terminating it when necessary.
    """

    def __init__(
        self,
        worker: Worker,
        logger: Logger,
        reconfigure: Callable[[None], None],
    ) -> None:
        self.worker = worker
        self.logger = logger
        self.reconfigure = reconfigure

        super().__init__()

    def run(self) -> None:
        self.logger.debug("Starting worker manager.")
        if not self.worker.running:
            self.worker.start()
        while self.worker.running:
            self.logger.debug("Checking if worker must be reconfigured.")
            if self.worker.oak.device.isClosed():
                self.logger.debug("Camera is closed. Reconfiguring worker.")
                self.logger.info(
                    "we are about to call reconfigure inside workermanager"
                )
                self.reconfigure()
                self.worker.running = False
            time.sleep(3)

    def stop(self) -> None:
        self.logger.debug("Stopping worker manager.")
        self.worker.stop()
