"""SQLite storage (SQLAlchemy) for conversations, turns, and profiles, plus a per-conversation
JSON export. This module has no opinion about flow or validity — it just persists whatever
`engine.py` hands it, once per turn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, ForeignKey, Table, create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from sqlalchemy.schema import Column

from screening_agent.models import CandidateProfile, Language, Stage, Terminal

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "screening.db"


class Base(DeclarativeBase):
    pass


class ConversationRow(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(primary_key=True)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    stage: Mapped[str]
    outcome: Mapped[str | None]
    disqualify_reason: Mapped[str | None]
    language: Mapped[str | None]
    # Re-engagement (M7). Deliberately separate from `updated_at`, which also moves on a nudge
    # send — the ladder needs to measure time since the *candidate* last did something, not since
    # the conversation was last touched at all (a nudge touches it too, but doesn't cancel itself).
    last_candidate_activity: Mapped[datetime | None]
    nudge_count: Mapped[int] = mapped_column(default=0)

    turns: Mapped[list[TurnRow]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="TurnRow.turn_index"
    )
    profile: Mapped[ProfileRow] = relationship(
        back_populates="conversation", uselist=False, cascade="all, delete-orphan"
    )


class TurnRow(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    turn_index: Mapped[int]
    role: Mapped[str]
    content: Mapped[str]
    created_at: Mapped[datetime]

    conversation: Mapped[ConversationRow] = relationship(back_populates="turns")


class ProfileRow(Base):
    __tablename__ = "profiles"

    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), primary_key=True)
    full_name: Mapped[str | None]
    has_license: Mapped[bool | None]
    city_raw: Mapped[str | None]
    zone_id: Mapped[str | None]
    availability: Mapped[str | None]
    preferred_schedule: Mapped[str | None]
    experience_years: Mapped[float | None]
    experience_platforms: Mapped[list] = mapped_column(JSON, default=list)
    start_date: Mapped[str | None]
    starts_immediately: Mapped[bool] = mapped_column(default=False)

    conversation: Mapped[ConversationRow] = relationship(back_populates="profile")


def _apply_profile(row: ProfileRow, profile: CandidateProfile) -> None:
    row.full_name = profile.full_name
    row.has_license = profile.has_license
    row.city_raw = profile.city_raw
    row.zone_id = profile.zone_id
    row.availability = profile.availability.value if profile.availability else None
    row.preferred_schedule = (
        profile.preferred_schedule.value if profile.preferred_schedule else None
    )
    row.experience_years = profile.experience_years
    row.experience_platforms = list(profile.experience_platforms)
    row.start_date = profile.start_date.isoformat() if profile.start_date else None
    row.starts_immediately = profile.starts_immediately


def _profile_to_dict(row: ProfileRow) -> dict[str, Any]:
    return {
        "full_name": row.full_name,
        "has_license": row.has_license,
        "city_raw": row.city_raw,
        "zone_id": row.zone_id,
        "availability": row.availability,
        "preferred_schedule": row.preferred_schedule,
        "experience_years": row.experience_years,
        "experience_platforms": row.experience_platforms,
        "start_date": row.start_date,
        "starts_immediately": row.starts_immediately,
    }


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite has no native datetime type — SQLAlchemy round-trips a value written as UTC-aware
    back as *naive* (confirmed against the installed SQLAlchemy/SQLite combination). Every
    datetime this module writes is `datetime.now(UTC)`, so re-attaching UTC on read is correct by
    convention, not a guess — without it, `reengage/policy.py`'s `now - last_candidate_activity`
    raises (can't subtract offset-naive and offset-aware datetimes)."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    id: str
    stage: str
    outcome: str | None
    disqualify_reason: str | None
    language: str | None
    profile: dict[str, Any]
    transcript: list[dict[str, str]]


@dataclass(frozen=True, slots=True)
class ActiveConversation:
    id: str
    stage: str
    language: str | None
    zone_id: str | None
    last_candidate_activity: datetime | None
    nudge_count: int


def _add_column_sql(table: Table, column: Column, engine: Engine) -> str:
    """DDL to add one missing column, or a `RuntimeError` naming what a human has to do instead."""
    type_sql = column.type.compile(engine.dialect)
    if column.nullable:
        return f"ALTER TABLE {table.name} ADD COLUMN {column.name} {type_sql}"
    # A NOT NULL column can only be added to a table that already has rows if it comes with a
    # default to backfill them. SQLAlchemy's `default=` is applied Python-side on insert, so it
    # isn't in the DDL — but when it's a plain scalar we can honestly reuse it as the fill value.
    default = getattr(column.default, "arg", None)
    if column.default is not None and not callable(default):
        literal = repr(default) if isinstance(default, str) else str(int(default))
        return (
            f"ALTER TABLE {table.name} ADD COLUMN {column.name} {type_sql} "
            f"NOT NULL DEFAULT {literal}"
        )
    raise RuntimeError(
        f"{table.name}.{column.name} is NOT NULL with no scalar default, so it cannot be "
        f"backfilled onto the existing rows in this database automatically. Migrate it by hand, "
        f"or delete the database if the data is disposable."
    )


def _reconcile_schema(engine: Engine) -> None:
    """Add columns that the models declare but an existing database lacks.

    `Base.metadata.create_all()` creates *missing tables* and nothing else — it will not touch a
    table that already exists, however far its schema has drifted. That is a silent failure with a
    long fuse: M7 added `last_candidate_activity` and `nudge_count` to `conversations`, every fresh
    database got them, and every database created before M7 kept working right up until the next
    insert, which then died with `OperationalError: table conversations has no column named
    last_candidate_activity`. Found exactly that way — a pre-M7 `data/screening.db` turned the
    first message of a containerised run into a 500.

    **This is deliberately not a migration system.** It handles the one case it can handle
    honestly — a column that was added — and raises a clear error for anything it cannot (renames,
    type changes, drops, non-defaultable NOT NULL columns). It exists so a demo or pilot survives
    schema drift; a real deployment gets Alembic, and `docs/deployment.md` says so.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue  # create_all() just built it, in full
        present = {col["name"] for col in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            statement = _add_column_sql(table, column, engine)
            with engine.begin() as conn:
                conn.exec_driver_sql(statement)
            logger.warning(
                "schema drift: added missing column %s.%s to the existing database",
                table.name,
                column.name,
            )


class Store:
    def __init__(self, db_path: Path = DB_PATH, exports_dir: Path | None = None) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        _reconcile_schema(self.engine)
        # Defaults alongside the DB file, not the hardcoded module constant — a Store pointed
        # at a custom db_path (tests, a debug script) must not write into the real data/exports/.
        self.exports_dir = exports_dir or (db_path.parent / "exports")

    def create_conversation(self, conversation_id: str) -> None:
        now = datetime.now(UTC)
        with Session(self.engine) as session:
            session.add(
                ConversationRow(
                    id=conversation_id,
                    created_at=now,
                    updated_at=now,
                    stage=Stage.GREETING.value,
                    outcome=None,
                    disqualify_reason=None,
                    language=None,
                    last_candidate_activity=None,
                    nudge_count=0,
                )
            )
            session.add(ProfileRow(conversation_id=conversation_id, experience_platforms=[]))
            session.commit()

    def record_turn(
        self,
        conversation_id: str,
        *,
        candidate_message: str | None,
        agent_message: str,
        profile: CandidateProfile,
        stage: Stage,
        outcome: Terminal | None,
        disqualify_reason: str | None,
        language: Language | None,
        now: datetime | None = None,
    ) -> None:
        # Injectable for the same reason `reengage/scheduler.run_once`'s `now` is: the fast-clock
        # demo (reengage/demo.py) needs `last_candidate_activity` set to a specific instant, not
        # whichever wall-clock moment happens to be current when the demo script runs.
        now = now or datetime.now(UTC)
        with Session(self.engine) as session:
            conv = session.get(ConversationRow, conversation_id)
            if conv is None:
                raise KeyError(
                    f"no conversation {conversation_id!r} — call create_conversation first"
                )
            next_index = len(conv.turns)
            if candidate_message is not None:
                session.add(
                    TurnRow(
                        conversation_id=conversation_id,
                        turn_index=next_index,
                        role="candidate",
                        content=candidate_message,
                        created_at=now,
                    )
                )
                next_index += 1
                # Any reply cancels the nudge ladder (process-design.md §3) — reset both the
                # clock the ladder measures from and the count of rungs already sent.
                conv.last_candidate_activity = now
                conv.nudge_count = 0
            session.add(
                TurnRow(
                    conversation_id=conversation_id,
                    turn_index=next_index,
                    role="agent",
                    content=agent_message,
                    created_at=now,
                )
            )

            profile_row = session.get(ProfileRow, conversation_id)
            assert profile_row is not None  # created alongside the conversation
            _apply_profile(profile_row, profile)

            conv.stage = stage.value
            conv.outcome = outcome.value if outcome else None
            conv.disqualify_reason = disqualify_reason
            if language is not None:
                conv.language = language.value
            conv.updated_at = now
            session.commit()

    def record_nudge(
        self,
        conversation_id: str,
        *,
        message: str,
        nudge_index: int,
        outcome: Terminal | None = None,
    ) -> None:
        """A re-engagement nudge (M7) — an agent-only turn. Does *not* touch
        `last_candidate_activity`; a nudge is the opposite of candidate activity, and if it did
        reset that clock the ladder would push itself back every time it fired."""
        now = datetime.now(UTC)
        with Session(self.engine) as session:
            conv = session.get(ConversationRow, conversation_id)
            if conv is None:
                raise KeyError(
                    f"no conversation {conversation_id!r} — call create_conversation first"
                )
            session.add(
                TurnRow(
                    conversation_id=conversation_id,
                    turn_index=len(conv.turns),
                    role="agent",
                    content=message,
                    created_at=now,
                )
            )
            conv.nudge_count = nudge_index + 1
            if outcome is not None:
                conv.outcome = outcome.value
            conv.updated_at = now
            session.commit()

    def list_active(self) -> list[ActiveConversation]:
        """Non-terminal conversations, for the re-engagement scheduler (M7) to scan."""
        with Session(self.engine) as session:
            rows = session.query(ConversationRow).filter(ConversationRow.outcome.is_(None)).all()
            return [
                ActiveConversation(
                    id=row.id,
                    stage=row.stage,
                    language=row.language,
                    zone_id=row.profile.zone_id if row.profile else None,
                    last_candidate_activity=_as_utc(row.last_candidate_activity),
                    nudge_count=row.nudge_count,
                )
                for row in rows
            ]

    def get(self, conversation_id: str) -> ConversationRecord:
        with Session(self.engine) as session:
            conv = session.get(ConversationRow, conversation_id)
            if conv is None:
                raise KeyError(f"no conversation {conversation_id!r}")
            return ConversationRecord(
                id=conv.id,
                stage=conv.stage,
                outcome=conv.outcome,
                disqualify_reason=conv.disqualify_reason,
                language=conv.language,
                profile=_profile_to_dict(conv.profile),
                transcript=[{"role": t.role, "content": t.content} for t in conv.turns],
            )

    def export_json(self, conversation_id: str, *, summary: str | None) -> Path:
        record = self.get(conversation_id)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        path = self.exports_dir / f"{conversation_id}.json"
        payload = {
            "conversation_id": record.id,
            "stage": record.stage,
            "outcome": record.outcome,
            "disqualify_reason": record.disqualify_reason,
            "language": record.language,
            "profile": record.profile,
            "transcript": record.transcript,
            "summary": summary,
            "exported_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return path
