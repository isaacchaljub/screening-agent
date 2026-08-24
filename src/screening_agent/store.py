"""SQLite storage (SQLAlchemy) for conversations, turns, and profiles, plus a per-conversation
JSON export. This module has no opinion about flow or validity — it just persists whatever
`engine.py` hands it, once per turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, ForeignKey, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from screening_agent.models import CandidateProfile, Language, Stage, Terminal

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


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    id: str
    stage: str
    outcome: str | None
    disqualify_reason: str | None
    language: str | None
    profile: dict[str, Any]
    transcript: list[dict[str, str]]


class Store:
    def __init__(self, db_path: Path = DB_PATH, exports_dir: Path | None = None) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
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
    ) -> None:
        now = datetime.now(UTC)
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
