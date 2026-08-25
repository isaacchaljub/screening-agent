"""Storage tests. Offline — SQLite on a tmp path, no network, no model calls."""

from __future__ import annotations

import sqlite3

import pytest

from screening_agent.models import CandidateProfile, Stage
from screening_agent.store import Store


def _columns(db_path, table: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _make_pre_m7_database(db_path) -> None:
    """A `conversations` table as it existed before M7 added the re-engagement columns — i.e. what
    any database created earlier still looks like on disk today."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE conversations ("
            " id VARCHAR NOT NULL PRIMARY KEY, created_at DATETIME NOT NULL,"
            " updated_at DATETIME NOT NULL, stage VARCHAR NOT NULL, outcome VARCHAR,"
            " disqualify_reason VARCHAR, language VARCHAR)"
        )
        conn.execute(
            "INSERT INTO conversations VALUES"
            " ('old-1', '2026-08-01 10:00:00', '2026-08-01 10:05:00', 'wrap_up',"
            "  'qualified', NULL, 'es')"
        )


def test_existing_database_missing_a_later_column_is_reconciled(tmp_path):
    # Regression: `Base.metadata.create_all()` creates missing *tables* and never alters an
    # existing one, so a database created before M7 kept its old schema silently and then failed
    # on the next insert with "table conversations has no column named last_candidate_activity".
    # Live-reproduced: it turned the first message of a containerised run into a 500.
    db_path = tmp_path / "drifted.db"
    _make_pre_m7_database(db_path)
    assert "last_candidate_activity" not in _columns(db_path, "conversations")

    Store(db_path=db_path)

    assert "last_candidate_activity" in _columns(db_path, "conversations")
    assert "nudge_count" in _columns(db_path, "conversations")


def test_reconciled_database_still_accepts_writes_and_keeps_its_old_rows(tmp_path):
    db_path = tmp_path / "drifted.db"
    _make_pre_m7_database(db_path)
    store = Store(db_path=db_path)

    store.create_conversation("new-1")
    store.record_turn(
        "new-1",
        candidate_message=None,
        agent_message="hola",
        profile=CandidateProfile(),
        stage=Stage.GREETING,
        outcome=None,
        disqualify_reason=None,
        language=None,
    )

    with sqlite3.connect(db_path) as conn:
        rows = dict(conn.execute("SELECT id, nudge_count FROM conversations"))
    assert rows == {"old-1": 0, "new-1": 0}  # pre-existing row kept, NOT NULL column backfilled


def test_unbackfillable_column_fails_loudly_rather_than_guessing(tmp_path):
    # The reconciler is deliberately not a migration system. A NOT NULL column with no scalar
    # default cannot be added to a table that already has rows, and inventing a value would be
    # worse than stopping — so it must raise something a human can act on.
    from sqlalchemy import Column, MetaData, String, Table, create_engine

    from screening_agent.store import _add_column_sql

    engine = create_engine(f"sqlite:///{tmp_path / 'x.db'}")
    table = Table("t", MetaData(), Column("needed", String, nullable=False))
    with pytest.raises(RuntimeError, match="cannot be backfilled"):
        _add_column_sql(table, table.c.needed, engine)


def test_export_json_lands_next_to_its_own_database(tmp_path):
    # A Store pointed at a custom db_path must not write into the real data/exports/.
    store = Store(db_path=tmp_path / "custom.db")
    store.create_conversation("c1")
    store.record_turn(
        "c1",
        candidate_message="hola",
        agent_message="¿cómo te llamas?",
        profile=CandidateProfile(full_name="Ana Ruiz"),
        stage=Stage.NAME,
        outcome=None,
        disqualify_reason=None,
        language=None,
    )
    path = store.export_json("c1", summary="test summary")
    assert path.parent == tmp_path / "exports"
    assert path.exists()
