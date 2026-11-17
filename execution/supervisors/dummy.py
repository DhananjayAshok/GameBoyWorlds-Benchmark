from __future__ import annotations

from typing import Any, Optional

from execution.report import ExecutorReport
from execution.supervisors.base import Supervisor


class DummySupervisor(Supervisor):
    """Pass the task straight to one executor run and surface nothing extra."""

    def _evaluate(self) -> Optional[dict]:
        self.call_executor(self._task)
        return None

    def process_executor_return(self, report: ExecutorReport) -> Any:
        return report
