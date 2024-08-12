import asyncio
import threading

from src.components.worker.worker import Worker
from viam.logging import getLogger


class AtomicBoolean:
    """
    Implementation of an atomic boolean used to prevent races in
    restarting the worker from the worker manager.
    """

    boolean: bool
    lock: threading.Lock

    def __init__(self, initial_value=False):
        # Only access and modify value with lock acquired. Trusting you on this...
        self.boolean = initial_value
        self.lock = threading.Lock()

    def get(self) -> bool:
        with self.lock:
            return self.boolean

    def set(self, new_value) -> None:
        with self.lock:
            self.boolean = new_value


class WorkerManager(threading.Thread):
    """
    WorkerManager starts then watches Worker, making sure the connection
    to the OAK camera is healthy, reconfiguring the entire module if not.
    It also handles manual worker restart requests.
    """

    def __init__(self, worker: Worker) -> None:
        super().__init__()
        self.logger = worker.logger
        self.worker = worker
        self.loop = None
        self.restart_atomic_bool = AtomicBoolean()
        self._stop_event = asyncio.Event()

    def run(self):
        """
        .start() of thread.Thread calls this method to begin execution
        in different event loop.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self.check_health())
        try:
            self.loop.run_forever()
        finally:
            self.loop.run_until_complete(self.shutdown())
            self.loop.close()

    async def check_health(self) -> None:
        """
        Starts task to periodically check the health of the worker as well
        as respond to manual restart requests.
        """
        self.logger.debug("Starting worker manager.")
        self.worker.configure()
        await self.worker.start()

        while not self._stop_event.is_set():
            self.logger.debug("Checking if worker must be restarted.")
            if self.worker.device and self.worker.device.isClosed():
                with self.restart_atomic_bool.lock:
                    self.logger.info(
                        "Camera is closed. Stopping and restarting worker."
                    )
                    await self._restart_worker()
            if self.restart_atomic_bool.get():
                with self.restart_atomic_bool.lock:
                    self.logger.info(
                        "Handling restart request. Stopping and restarting worker."
                    )
                    await self._restart_worker()
                    self.restart_atomic_bool.boolean = False

            await asyncio.sleep(3)

    def stop(self):
        """
        Thread-safe stop method to set the event to stop the singular health checking task.
        """
        self.loop.call_soon_threadsafe(self._stop_event.set)
        self.loop.call_soon_threadsafe(self.loop.stop)

    async def shutdown(self):
        """
        Taking care of and closing straggler tasks.
        """
        self.worker.stop()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _restart_worker(self):
        """
        Tears down and fully reboots worker and its pipeline.
        """
        self.worker.reset()
        self.worker.configure()
        await self.worker.start()
