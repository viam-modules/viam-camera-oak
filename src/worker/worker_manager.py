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
        else:
            self.logger.warn("worker already running!")

        while self.worker.running:
            self.logger.debug("Checking if worker must be reconfigured.")
            if self.worker.oak.device.isClosed():
                self.logger.debug("Camera is closed. Reconfiguring worker.")
                self.reconfigure()
                self.worker.running = False
            time.sleep(3)
