"""
db/models.py

Defines the database tables as Python classes, matching the ERD agreed
on earlier: jobs -> gauges -> keypoints / candidates / selections.

Each class below is a "model" -- a Python class whose attributes map
directly to columns in an actual database table. This file defines
STRUCTURE only (what tables exist, what columns they have, how they
relate to each other) -- it does not connect to any database itself.
That's db/session.py's job (next file).
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """
    Every model below inherits from this. SQLAlchemy uses it to keep
    track of "every table I know about" in one place -- this is what
    lets a single line elsewhere (Base.metadata.create_all(engine)) go
    create ALL the tables defined in this file at once, without listing
    them out individually.
    """
    pass


class Job(Base):
    """
    One row per uploaded photo, tracking it through the async pipeline.
    This is the table GET /jobs/{job_id} will read from -- its `status`
    column is literally the answer to "is this job done yet?".
    """
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    # One of: "queued", "stage1", "stage2", "qc_gate", "stage3", "done",
    # "failed", "rejected" -- matches the status values torch_worker.py /
    # paddle_worker.py already return (see their TODOs -- this is exactly
    # what those TODOs will write into).
    status: Mapped[str] = mapped_column(String, default="queued")

    # Freeform detail for WHY status is what it is -- e.g. "no_gauge_detected"
    # (Stage 1 fail), "low_keypoint_confidence, implausible_arc_span"
    # (QC gate reject). Added once it became clear GET /jobs/{job_id}
    # needs more than just a status word to be useful to a caller.
    detail: Mapped[str] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    # A job can produce more than one gauge attempt (e.g. if Stage 1
    # returns "retake" and a new photo is submitted under the same job) --
    # see the ERD note on why this is one-to-many rather than one-to-one.
    # order_by ensures job.gauges[-1] reliably means "the most recent
    # attempt" -- without it, SQLAlchemy doesn't guarantee any particular
    # order when loading a relationship.
    gauges: Mapped[list["Gauge"]] = relationship(back_populates="job", order_by="Gauge.created_at")


class Gauge(Base):
    """
    One row per Stage 1 detection attempt. Stores the bbox Stage 1 found
    and the QC gate's verdict -- this is what torch_worker.py's TODOs
    ("write bbox + ... to the gauges table") will fill in.
    """
    __tablename__ = "gauges"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"))

    image_url: Mapped[str] = mapped_column(String, nullable=True)

    bbox_x: Mapped[int] = mapped_column(Integer)
    bbox_y: Mapped[int] = mapped_column(Integer)
    bbox_w: Mapped[int] = mapped_column(Integer)
    bbox_h: Mapped[int] = mapped_column(Integer)
    coverage_ratio: Mapped[float] = mapped_column(Float, nullable=True)

    # "ok" / "reject" -- qc_gate.py's run_qc_gate() return value, stored
    # verbatim; qc_reasons holds the comma-joined reasons list when rejected.
    qc_status: Mapped[str] = mapped_column(String, nullable=True)
    qc_reasons: Mapped[str] = mapped_column(String, nullable=True)

    # Added specifically so Job.gauges can be ordered ("which attempt was
    # most recent") -- a Gauge row's own id (a UUID) carries no time
    # information, so without this there'd be no reliable way to tell
    # attempts apart when a job has more than one.
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    job: Mapped["Job"] = relationship(back_populates="gauges")
    keypoints: Mapped["Keypoints"] = relationship(back_populates="gauge", uselist=False)
    candidates: Mapped[list["Candidate"]] = relationship(back_populates="gauge")
    selection: Mapped["Selection"] = relationship(back_populates="gauge", uselist=False)


class Keypoints(Base):
    """
    One row per gauge -- the four keypoints (in ORIGINAL image coordinates,
    i.e. after key_points.map_points_to_original()) and their confidences.
    """
    __tablename__ = "keypoints"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    gauge_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gauges.id"))

    center_x: Mapped[float] = mapped_column(Float)
    center_y: Mapped[float] = mapped_column(Float)
    tip_x: Mapped[float] = mapped_column(Float)
    tip_y: Mapped[float] = mapped_column(Float)
    min_x: Mapped[float] = mapped_column(Float)
    min_y: Mapped[float] = mapped_column(Float)
    max_x: Mapped[float] = mapped_column(Float)
    max_y: Mapped[float] = mapped_column(Float)

    center_conf: Mapped[float] = mapped_column(Float)
    tip_conf: Mapped[float] = mapped_column(Float)
    min_conf: Mapped[float] = mapped_column(Float)
    max_conf: Mapped[float] = mapped_column(Float)

    gauge: Mapped["Gauge"] = relationship(back_populates="keypoints")


class Candidate(Base):
    """
    One row per OCR candidate. A single gauge produces up to 2 (position)
    x 3 (rank) = 6 rows here -- pipeline_stage3.py's pos_a_candidates and
    pos_b_candidates, flattened out one row per candidate instead of
    nested lists, since that's how a relational table works.
    """
    __tablename__ = "candidates"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    gauge_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gauges.id"))

    position: Mapped[str] = mapped_column(String)   # "A" (min-labeled) or "B" (max-labeled)
    rank: Mapped[int] = mapped_column(Integer)       # 1, 2, or 3 -- position within the top-N list

    text: Mapped[str] = mapped_column(String)
    parsed_value: Mapped[float] = mapped_column(Float)
    composite_score: Mapped[float] = mapped_column(Float, nullable=True)
    is_forced_zero: Mapped[bool] = mapped_column(Boolean, default=False)

    gauge: Mapped["Gauge"] = relationship(back_populates="candidates")


class Selection(Base):
    """
    One row per gauge -- the human's final min/max choice (with swap, if
    applied), the resulting computed value, and their correctness judgment.
    This is what POST /gauges/{gauge_id}/selection will write.
    """
    __tablename__ = "selections"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    gauge_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gauges.id"))

    chosen_min_value: Mapped[float] = mapped_column(Float)
    chosen_max_value: Mapped[float] = mapped_column(Float)
    swapped: Mapped[bool] = mapped_column(Boolean, default=False)
    computed_value: Mapped[float] = mapped_column(Float, nullable=True)

    # The real value the person read off the gauge by eye, filled in only
    # when judgment="incorrect" -- the raw material for an error
    # (actual_value - computed_value) computed later, on demand, rather
    # than stored here. Storing a precomputed error would go stale if
    # the value-calculation logic (pointer_math.angle_utils) ever changes.
    actual_value: Mapped[float] = mapped_column(Float, nullable=True)

    # "correct" / "incorrect" / "unclear" -- matches the interactive HTML
    # report's judgment options from the exploration phase.
    judgment: Mapped[str] = mapped_column(String, nullable=True)
    manual_correction: Mapped[str] = mapped_column(String, nullable=True)
    note: Mapped[str] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    gauge: Mapped["Gauge"] = relationship(back_populates="selection")