import threading
from tqdm import tqdm


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
        self._bar: tqdm | None = None

    def reset(self, total: int, desc: str = "progress") -> None:
        with self._lock:
            if self._bar is not None:
                self._bar.close()
            self._total = total
            self._done = 0
            self._bar = tqdm(
                total=total,
                desc=desc,
                unit="batch",
                dynamic_ncols=True,
                leave=True,
            )

    def mark_done(self) -> tuple[int, int]:
        with self._lock:
            self._done += 1
            if self._bar is not None:
                self._bar.update(1)
                if self._done >= self._total:
                    self._bar.close()
                    self._bar = None
            return self._done, self._total

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self._done, self._total

    def close(self) -> None:
        with self._lock:
            if self._bar is not None:
                self._bar.close()
                self._bar = None
