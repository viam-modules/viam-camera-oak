from threading import Thread
import time
from typing import Callable
from logging import Logger

from src.worker.worker import Worker


class WorkerManager(Thread):
    """
    WorkerManager starts then watches Worker, making sure the connection
    to the OAK camera is healthy, reconfiguring the entire module if not.
    """

    def __init__(
        self,
        worker: Worker,
        logger: Logger,
    ) -> None:
        self.worker = worker
        self.logger = logger

        super().__init__()

    def run(self) -> None:
        self.logger.debug("Starting worker manager.")
        if self.worker.should_exec:
            self.worker.configure()
            self.worker.start()
        else:
            self.logger.warn("Worker already running!")

        while self.worker.should_exec:
            self.logger.debug("Checking if worker must be reconfigured.")
            if self.worker.oak and self.worker.oak.device.isClosed():
                self.logger.info("Camera is closed. Stopping and reconfiguring worker.")
                self.worker.reset()
                self.worker.configure()
                self.worker.start()
            time.sleep(3)
