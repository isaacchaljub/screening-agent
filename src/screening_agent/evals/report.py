"""Renders `ScenarioResult`s into a markdown report — one row per scenario, one column per model
(M8's `--model` accepts more than one, forward-compatible with M9's future full bake-off), plus a
per-model summary table.

M9's per-model summary adds avg latency (wall-clock seconds across every model call in a
conversation, per `runner._instrument`) and avg cost (`runner.pricing`, USD per conversation) —
cost is `—` for any model with no entry in `pricing.PRICING_PER_MILLION_TOKENS` rather than a
guessed number."""

from __future__ import annotations

from datetime import UTC, datetime

from screening_agent.evals.runner import ROLES_MODE, ScenarioResult


def _cell(result: ScenarioResult | None) -> str:
    if result is None:
        return "—"
    if result.error:
        return f"⚠️ ERROR: {result.error[:60]}"
    mark = "✅" if result.outcome_match else "❌"
    return (
        f"{mark} {result.actual_outcome} · fields {result.field_match_ratio:.0%} · "
        f"len {result.length_compliance_ratio:.0%} · {result.turns_taken}t"
    )


def render_report(results: list[ScenarioResult]) -> str:
    models = sorted({r.model for r in results})
    scenario_names = sorted({r.scenario for r in results})
    by_key = {(r.scenario, r.model): r for r in results}

    lines = [
        "# Screening agent — eval report",
        "",
        f"Generated {datetime.now(UTC).isoformat()}",
        "",
        "Cell format: outcome match · field accuracy · message-length compliance · turns taken.",
        "",
        "| Scenario | " + " | ".join(models) + " |",
        "|---|" + "---|" * len(models),
    ]
    for name in scenario_names:
        row = [name] + [_cell(by_key.get((name, m))) for m in models]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Per-model summary", ""]
    lines.append(
        f"`{ROLES_MODE}` is the production role split (`registry.ROLES`: Haiku extracts, Sonnet "
        "composes), not one forced model — it is the only row whose cost is what a real "
        "conversation costs. Every other row forces both calls onto the single model named."
    )
    lines.append("")
    lines.append(
        "| Model | Outcome pass rate | Avg field accuracy | Avg length compliance | Avg turns | "
        "Avg latency/conv | Avg cost/conv |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for m in models:
        rows = [r for r in results if r.model == m]
        n = len(rows) or 1
        outcome_rate = sum(r.outcome_match for r in rows) / n
        field_avg = sum(r.field_match_ratio for r in rows) / n
        len_avg = sum(r.length_compliance_ratio for r in rows) / n
        turns_avg = sum(r.turns_taken for r in rows) / n
        latency_avg = sum(r.latency_seconds for r in rows) / n
        costs = [r.cost_usd for r in rows if r.cost_usd is not None]
        cost_cell = f"${sum(costs) / len(costs):.4f}" if costs else "—"
        served_by = sorted({name for r in rows for name in r.models_used})
        label = f"{m} ({' + '.join(served_by)})" if m == ROLES_MODE and served_by else m
        lines.append(
            f"| {label} | {outcome_rate:.0%} | {field_avg:.0%} | {len_avg:.0%} | {turns_avg:.1f} | "
            f"{latency_avg:.1f}s | {cost_cell} |"
        )

    mismatches = [r for r in results if r.mismatched_fields and not r.error]
    if mismatches:
        lines += ["", "## Field mismatches", ""]
        for r in mismatches:
            lines.append(f"- **{r.scenario}** ({r.model}): " + "; ".join(r.mismatched_fields))

    return "\n".join(lines)
