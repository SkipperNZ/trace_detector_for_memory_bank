"""Normalize trusted event metadata and construct only ancestor-visible inputs."""

from dataclasses import asdict, dataclass

from .io import digest, uniform

INPUT_VERSION = "causal-visible-v1"
METRIC_VERSION = "agent-dissatisfaction-v1"
RUBRIC = (
    "Does the target user message explicitly express dissatisfaction with the agent's "
    "previous action? yes: an explicit complaint, reproach or sarcasm directed at the agent; "
    "no: a calm correction, a new requirement, an external problem or a quoted complaint; "
    "unclear: the addressee or relevant context cannot be established. "
    "Absence of a complaint is not evidence of task success."
)


@dataclass(frozen=True)
class Event:
    tenant_id: str
    trace_id: str
    event_id: str
    parent_id: str | None
    branch_id: str
    sequence: int
    role: str
    visibility: str
    text: str
    status: str = "completed"
    language: str = "unknown"
    group_id: str | None = None

    def __post_init__(self):
        for field in ("tenant_id", "trace_id", "event_id", "branch_id", "language"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"{field} must be a nonempty string")
        for field in ("parent_id", "group_id"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field} must be null or a nonempty string")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be a nonnegative integer")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if self.role not in {"user", "assistant", "tool", "system"}:
            raise ValueError("Unsupported role")
        if self.visibility not in {"user_visible", "internal"}:
            raise ValueError("visibility must be user_visible or internal")
        if self.status not in {"completed", "partial", "cancelled"}:
            raise ValueError("Unsupported event status")

    @property
    def key(self):
        return self.tenant_id, self.trace_id, self.event_id


def normalize(rows: list[dict]) -> list[Event]:
    events: dict[tuple, Event] = {}
    groups: dict[tuple, set] = {}
    for row in rows:
        try:
            event = Event(**row)
        except TypeError as exc:
            raise ValueError(f"Invalid event schema: {exc}") from exc
        if event.key in events and events[event.key] != event:
            raise ValueError(f"Conflicting delivery for event {event.key}")
        events[event.key] = event
        groups.setdefault(event.key[:2], set()).add(event.group_id)
    if any(len(group) != 1 for group in groups.values()):
        raise ValueError("group_id must be consistent across every event in a trace")
    for event in events.values():
        parent = events.get((event.tenant_id, event.trace_id, event.parent_id))
        if parent is not None and parent.sequence >= event.sequence:
            raise ValueError(f"Non-causal parent ordering/cycle at {event.key}")
    return sorted(events.values(), key=lambda e: (e.tenant_id, e.trace_id, e.sequence, e.event_id))


def split_for(tenant: str, group: str, seed: int) -> str:
    draw = uniform(seed, "split-v1", [tenant, group])
    return "train" if draw < 0.7 else "calibration" if draw < 0.85 else "test"


def prepare(rows: list[dict], *, max_chars: int = 12000, seed: int = 42) -> list[dict]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    events = normalize(rows)
    lookup = {event.key: event for event in events}
    examples = []
    for target in events:
        if (target.role, target.visibility, target.status) != ("user", "user_visible", "completed"):
            continue
        ancestors = []
        current = target
        status = "complete"
        while current.parent_id is not None:
            parent = lookup.get((target.tenant_id, target.trace_id, current.parent_id))
            if parent is None:
                status = "missing_parent"
                break
            if (
                parent.visibility == "user_visible"
                and parent.status == "completed"
                and parent.role != "system"
            ):
                ancestors.append(parent)
            current = parent

        # Keep a contiguous suffix of visible ancestors. Never cut the target text.
        selected = []
        size = len(target.text)
        for ancestor in ancestors:
            if size + len(ancestor.text) > max_chars:
                break
            selected.append(ancestor)
            size += len(ancestor.text)
        omitted = len(ancestors) - len(selected)
        if status == "complete" and (omitted or size > max_chars):
            status = "truncated"
        context = [
            {"event_id": event.event_id, "role": event.role, "text": event.text}
            for event in reversed(selected)
        ]
        context.append({"event_id": target.event_id, "role": "user", "text": target.text})
        identity = {
            "tenant_id": target.tenant_id,
            "trace_id": target.trace_id,
            "branch_id": target.branch_id,
            "target_event_id": target.event_id,
            "metric_version": METRIC_VERSION,
            "input_version": INPUT_VERSION,
            "context": context,
            "context_status": status,
            "omitted_ancestors": omitted,
            "max_chars": max_chars,
        }
        examples.append(
            {
                **identity,
                "input_id": digest(identity),
                "language": target.language,
                "group_id": target.group_id or target.trace_id,
                "split": split_for(target.tenant_id, target.group_id or target.trace_id, seed),
                "split_seed": seed,
            }
        )
    return examples


def annotation_template(examples: list[dict]) -> list[dict]:
    # No scores, predicted labels or split information are shown to annotators.
    return [
        {
            "input_id": ex["input_id"],
            "metric_version": ex["metric_version"],
            "target_event_id": ex["target_event_id"],
            "context": ex["context"],
            "context_status": ex["context_status"],
            "rubric": RUBRIC,
            "label": None,
            "evidence_event_ids": [],
            "annotator": None,
            "label_source": "human",
        }
        for ex in examples
    ]


def normalized_rows(rows: list[dict]) -> list[dict]:
    return [asdict(event) for event in normalize(rows)]
