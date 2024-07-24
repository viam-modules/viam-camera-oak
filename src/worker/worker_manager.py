from threading import Thread
import time

from src.worker.worker import Worker

# TODO RSDK-8342: re-add the commented logging logic
# from viam.logging import getLogger

# LOGGER = getLogger("oak-manager-logger")


class WorkerManager(Thread):
    """
    WorkerManager starts then watches Worker, making sure the connection
    to the OAK camera is healthy, reconfiguring the entire module if not.
    """

    def __init__(
        self,
        worker: Worker,
    ) -> None:
        self.worker = worker

        super().__init__()

    def run(self) -> None:
        # LOGGER.debug("Starting worker manager.")
        if self.worker.should_exec:
            self.worker.configure()
            self.worker.start()
        else:
            # LOGGER.warn("Worker already running!")
            pass

        while self.worker.should_exec:
            # LOGGER.debug("Checking if worker must be reconfigured.")
            if self.worker.oak and self.worker.oak.device.isClosed():
                # LOGGER.info("Camera is closed. Stopping and reconfiguring worker.")
                self.worker.reset()
                self.worker.configure()
                self.worker.start()
            time.sleep(3)
