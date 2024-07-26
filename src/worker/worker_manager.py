import asyncio
from threading import Thread

from src.worker.worker import Worker
from viam.logging import getLogger


class WorkerManager(Thread):
    """
    WorkerManager starts then watches Worker, making sure the connection
    to the OAK camera is healthy, reconfiguring the entire module if not.
    """

    def __init__(
        self,
        worker: Worker,
    ) -> None:
        self.logger = getLogger("oak-manager-logger")
        self.worker = worker
        super().__init__()

    def run(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.create_task(self.check_health())
        loop.run_forever()

    async def check_health(self) -> None:
        self.logger.debug("Starting worker manager.")
        if self.worker.should_exec:
            await self.worker.configure()
            self.worker.start()
        else:
            self.logger.warn("Worker already running!")
            pass

        while self.worker.should_exec:
            self.logger.debug("Checking if worker must be restarted.")
            if self.worker.oak and self.worker.oak.device.isClosed():
                self.logger.info("Camera is closed. Stopping and restarting worker.")
                self.worker.reset()
                await self.worker.configure()
                self.worker.start()
            await asyncio.sleep(3)

    def stop(self):
        self.worker.stop()
