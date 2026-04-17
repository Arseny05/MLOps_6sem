from Orchestrator import ModelOrchestrator
import threading
import time
import traceback


class BackgroundChecker:
    def __init__(self, Orch: ModelOrchestrator, monitor_interval=300):
        self.Orch = Orch
        self.interval = monitor_interval
        self.stop_event = threading.Event()
        self.thread = None
    
    def _main_loop_(self):
        last_check = 0
        while not self.stop_event.is_set():
            now = time.time()
            try:
                if now - last_check >= self.interval:
                    self.run_checks()
                    last_check = now
            except Exception as e:
                print(f"Scheduler error: {e}\n")
                traceback.print_exc()
            
            self.stop_event.wait(1.0)
    
    def run_checks(self):
        print("Starting checks...\n")
        with self.Orch.lock:
            lst = list(self.Orch.MODELS.keys())
        for index in lst:
            try:
                print(self.Orch.check_for_drift(index))
            except Exception as e:
                print(f"Check failed for model with ID = {index}\n")
    
    def start(self):
        if self.thread is not None and self.thread.is_alive(): return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._main_loop_, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.stop_event.set()
        if self.thread is not None: self.thread.join(timeout=2)