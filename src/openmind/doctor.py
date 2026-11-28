"""doctor — the store's health check. Suggests, never prescribes.

One binary question per run: what is the embedding architecture doing right
now? `embedding_health()` answers it with the same three counts
`semantic_search` gates on, so the doctor and the tool can never disagree
about what "healthy" means.

- pending — live sections awaiting a vector. Normal (mid-embed or an
  enqueue that never landed after a crash). Non-zero → suggest
  `--reconcile`, which re-enqueues exactly these ids.
- stale — vectors carrying a model other than the configured one. Model
  drift. Not reconcilable: the operator reembeds or clears
  `openrouter_model`. Printed loudly, fixed by a human.
- broken — rows violating the all-or-nothing embedding CHECK. Should be
  impossible; if it fires the architecture is sick and only a human can
  decide what the right state was.

Reconcile never runs by itself. The doctor diagnoses; the operator orders
the treatment. Exit code is 1 when anything is unhealthy, so a wake script
can branch on it.

Run: `python -m openmind.doctor [--reconcile]`
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from openmind import db, telemetry, worker
from openmind.embedding import embedding_health


def report(conn: Any) -> dict[str, int]:
    """The three counts plus the configured model, from the shared gate."""
    return embedding_health(conn)


def reconcile(conn: Any, limit: int = 1000) -> int:
    """Enqueue every pending section. Returns how many were queued."""
    ids = worker.pending_section_ids(conn, limit=limit)
    if not ids:
        return 0
    return len(ids) if worker.enqueue(ids) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="openmind.doctor",
        description="Diagnose the embedding architecture; suggest, never auto-run.",
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="enqueue every pending (un-embedded) section, then re-check",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="max sections to reconcile in one pass (default 1000)",
    )
    args = parser.parse_args(argv)

    telemetry.configure()
    with db.connection() as conn:
        health = report(conn)
        if args.reconcile:
            queued = reconcile(conn, limit=args.limit)
            telemetry.log_event("doctor.reconcile", queued=queued)
            print(f"reconcile: queued {queued} section(s) for embedding")
            health = report(conn)  # re-count after the enqueue

    pending, stale, broken, model = (
        health["pending"],
        health["stale"],
        health["broken"],
        health["model"],
    )
    print(f"embedding model: {model}")
    print(f"  pending (awaiting embed): {pending}")
    print(f"  stale   (wrong model):    {stale}")
    print(f"  broken (corrupt records): {broken}")

    unhealthy = False
    if broken:
        print(
            f"DOCTOR: {broken} section(s) have a corrupted embedding record — "
            "semantic search is unsafe; reconcile with your user before trusting any result."
        )
        unhealthy = True
    if stale:
        print(
            f"DOCTOR: {stale} section(s) are embedded with a model other than {model!r} — "
            "semantic search refuses until you reembed, clear [embedding] openrouter_model, "
            "or stop using semantic search."
        )
        unhealthy = True
    if pending:
        print(
            f"DOCTOR: {pending} section(s) have no vector yet — "
            "run `python -m openmind.doctor --reconcile` to queue them."
        )
        unhealthy = True
    if not unhealthy:
        print("DOCTOR: healthy — every live section is embedded with the configured model.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
