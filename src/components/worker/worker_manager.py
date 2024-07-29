import asyncio
from threading import Thread

from src.components.worker.worker import Worker
from viam.logging import getLogger


class WorkerManager(Thread):
    """
    WorkerManager starts then watches Worker, making sure the connection
    to the OAK camera is healthy, reconfiguring the entire module if not.
    """

    def __init__(self, worker: Worker) -> None:
        super().__init__()
        self.logger = getLogger("viam-oak-manager-logger")
        self.worker = worker
        self.loop = None
        self._stop_event = asyncio.Event()

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self.check_health())
        try:
            self.loop.run_forever()
        finally:
            self.loop.run_until_complete(self.shutdown())
            self.loop.close()

    async def check_health(self) -> None:
        self.logger.debug("Starting worker manager.")
        await self.worker.configure()
        self.worker.start()

        while not self._stop_event.is_set():
            self.logger.debug("Checking if worker must be restarted.")
            if (
                self.worker.oak
                and self.worker.oak.device
                and self.worker.oak.device.isClosed()
            ):
                self.logger.info("Camera is closed. Stopping and restarting worker.")
                self.worker.reset()
                await self.worker.configure()
                self.worker.start()
            await asyncio.sleep(3)

    def stop(self):
        self.loop.call_soon_threadsafe(self._stop_event.set)
        self.loop.call_soon_threadsafe(self.loop.stop)

    async def shutdown(self):
        self.worker.stop()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
