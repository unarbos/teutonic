import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from teutonic.evaluation.early_stopping import EarlyStoppingPolicy
from teutonic.validator.service import ValidatorScheduler


class ValidatorHealthTests(unittest.TestCase):
    def test_queue_claim_requires_token_weighted_early_stopping(self):
        for feature, available in (
            ("token-weighted-observed-quantile-v1", True),
            ("observed-quantile-v1", False),
            (None, False),
        ):
            with self.subTest(feature=feature):
                repository = Mock()
                repository.claim_next.return_value = None
                evaluator = Mock()
                evaluator.health = AsyncMock(return_value={
                    "status": "ok",
                    "request_features": {"challenger_futility_early_stopping": feature},
                })
                scheduler = ValidatorScheduler(
                    repository,
                    evaluator,
                    policy=SimpleNamespace(early_stopping=EarlyStoppingPolicy(enabled=True)),
                    preflight=AsyncMock(),
                )

                asyncio.run(scheduler.run_once())

                if available:
                    repository.claim_next.assert_called_once()
                else:
                    repository.claim_next.assert_not_called()
