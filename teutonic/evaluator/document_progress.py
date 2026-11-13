"""Document progress across reference and paired scoring, independent of worker layout."""

import threading


class DocumentProgress:
    def __init__(self, total, emit):
        self.total, self.emit = total, emit
        self.reference_done = self.paired_done = 0
        self.paired = False
        self.provisional = {}
        self.lock = threading.RLock()

    def __call__(self, event):
        with self.lock:
            event = dict(event)
            if event.get("phase") == "scoring":
                event["phase"] = "paired_scoring" if self.paired else "reference_scoring"
            event.update(self.provisional)
            event.update(
                done=self.paired_done if self.paired else self.reference_done,
                total=self.total,
                percent=50.0 * (self.reference_done + self.paired_done) / self.total,
            )
            self.emit(event)

    def reference_scored(self, count):
        with self.lock:
            self.reference_done = count
            self({"phase": "reference_scoring"})

    def begin_paired(self):
        with self.lock:
            self.paired = True
            self({"phase": "paired_prepare"})

    def paired_scored(self, count, provisional):
        with self.lock:
            self.paired_done = count
            self.provisional.update(provisional)
            self({"phase": "paired_scoring"})
