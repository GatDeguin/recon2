import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Optional


class MetricsLogger:
    """Simple SQLite-based metrics logger."""

    def __init__(self, db_path: str = "metrics.db") -> None:
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self) -> None:
        with closing(self.conn.cursor()) as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    timestamp TEXT,
                    wer REAL,
                    nmm_acc REAL,
                    latency REAL,
                    fps REAL
                )
                """
            )
            self.conn.commit()

    def log(
        self,
        wer: Optional[float] = None,
        nmm_acc: Optional[float] = None,
        latency: Optional[float] = None,
        fps: Optional[float] = None,
    ) -> None:
        with closing(self.conn.cursor()) as cur:
            cur.execute(
                "INSERT INTO metrics (timestamp, wer, nmm_acc, latency, fps) VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.utcnow().isoformat(),
                    wer,
                    nmm_acc,
                    latency,
                    fps,
                ),
            )
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()
