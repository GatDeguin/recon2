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
                    cer REAL,
                    nmm_acc REAL,
                    class_acc TEXT,
                    latency REAL,
                    fps REAL
                )
                """
            )
            self.conn.commit()

    def log(
        self,
        wer: Optional[float] = None,
        cer: Optional[float] = None,
        nmm_acc: Optional[float] = None,
        class_acc: Optional[dict] = None,
        latency: Optional[float] = None,
        fps: Optional[float] = None,
    ) -> None:
        """Insert a new metrics row into the database."""
        import json

        with closing(self.conn.cursor()) as cur:
            cur.execute(
                """
                INSERT INTO metrics
                    (timestamp, wer, cer, nmm_acc, class_acc, latency, fps)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    wer,
                    cer,
                    nmm_acc,
                    json.dumps(class_acc) if class_acc is not None else None,
                    latency,
                    fps,
                ),
            )
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()
