import threading


def progress_bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0/0"

    filled = int(width * done / total)
    percent = done / total * 100
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total} {percent:.1f}%"


class ThreadSafeProgress:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._done = 0

    def reset(self, total: int) -> None:
        with self._lock:
            self._total = total
            self._done = 0

    def mark_done(self) -> tuple[int, int]:
        with self._lock:
            self._done += 1
            return self._done, self._total

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self._done, self._total
