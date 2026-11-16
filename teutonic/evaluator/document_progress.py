"""Document progress across reference and paired scoring, independent of worker layout."""

import threading


class DocumentProgress:
    def __init__(self, total, emit, *, token_budget=None):
        self.total, self.emit = total, emit
        self.token_budget = token_budget
        self.sampling = token_budget is not None
        self.reference_tokens = 0
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
            percent = (min(49.9, 50.0 * self.reference_tokens / max(1, self.token_budget))
                       if self.sampling else
                       50.0 * (self.reference_done + self.paired_done) / max(1, self.total))
            event.update(
                done=self.paired_done if self.paired else self.reference_done,
                total=self.total,
                percent=percent,
            )
            self.emit(event)

    def selected(self, count):
        with self.lock:
            self.total = count

    def finish_sampling(self, count):
        with self.lock:
            self.total = count
            self.sampling = False

    def reference_scored(self, count, *, input_tokens=None):
        with self.lock:
            self.reference_done = count
            if input_tokens is not None:
                self.reference_tokens = input_tokens
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
