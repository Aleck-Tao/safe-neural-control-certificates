from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape


CONTROLLER_LABELS = {
    "raw_nn": "Raw NN",
    "input_clipped_nn": "Input-clipped NN",
    "nominal_only_projection": "Nominal projection",
    "robust_projection": "Robust projection",
    "linear_stable_baseline": "Linear baseline",
}


def _stable(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: _stable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_stable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        output = {}
        for key in fieldnames:
            value = row.get(key, "")
            output[key] = f"{value:.12g}" if isinstance(value, float) else value
        writer.writerow(output)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def render_report(
    summary: dict[str, Any], counterexamples: list[dict[str, Any]]
) -> str:
    margins = summary["analytic_certificate_margins"]
    lines = [
        "# Safe Neural Control Certificate Benchmark",
        "",
        "This report is generated from a deterministic scalar sampled-data experiment. "
        "The analytic certificate applies to the composed neural-policy plus robust-projection "
        "controller under the declared model assumptions; rollout statistics are regression evidence.",
        "",
        "## Analytic certificate",
        "",
        "| Quantity | Verified margin |",
        "|---|---:|",
        f"| `eta` | {margins['eta']:.6f} |",
        f"| `eta - disturbance_bound` | {margins['eta_minus_disturbance']:.6f} |",
        f"| Box forward-invariance | {margins['box_invariance']:.6f} |",
        f"| Input authority | {margins['input_authority']:.6f} |",
        "| Uncertainty-interval width (next-state units) | "
        f"{margins['uncertainty_interval_width_next_state']:.6f} |",
        "",
        "All margins are recomputed independently by `safecontrol verify`; the finite grid is not used as proof.",
        "",
        "## Closed-loop comparison",
        "",
        "| Controller | Runs | Box safe | Practical contraction | Input bounded | Target tube | Mean intervention | Mean cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["controllers"]:
        lines.append(
            "| {label} | {runs} | {safe:.3f} | {contract:.3f} | {inputs:.3f} | "
            "{tube:.3f} | {intervention:.3f} | {cost:.3f} |".format(
                label=CONTROLLER_LABELS[item["controller_id"]],
                runs=item["runs"],
                safe=item["box_safe_rate"],
                contract=item["practical_contraction_rate"],
                inputs=item["input_bounds_rate"],
                tube=item["target_tube_rate"],
                intervention=item["mean_intervention_rate"],
                cost=item["mean_quadratic_cost"],
            )
        )
    lines.extend(
        [
            "",
            "## Deliberate negative controls",
            "",
            "| Case | Controller | Box violation | Contraction violation | Input violation |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in counterexamples:
        if item["counterexample_id"] == "extreme_finite_neural_output":
            continue
        lines.append(
            "| {case} | {controller} | {box:.6f} | {contraction:.6f} | {input:.6f} |".format(
                case=item["counterexample_id"],
                controller=CONTROLLER_LABELS[item["controller_id"]],
                box=item["box_violation"],
                contraction=item["contraction_violation"],
                input=item["input_violation"],
            )
        )
    lines.extend(
        [
            "",
            "The negative controls are intentional: input clipping alone does not enforce the state constraint, "
            "and a projection built only for the midpoint model can miss the robust contraction bound.",
            "",
            "## Interpretation boundary",
            "",
            "- The plant is scalar, discrete-time, synthetic, and exactly specified.",
            "- The guarantee assumes `a` lies in the configured interval and the disturbance bound holds at every step.",
            "- The neural policy alone is not certified. The certificate belongs to the exact robust projection composition.",
            "- This release makes no MIMO, continuous-time, hardware, actuator, airworthiness, or physical-safety claim.",
            "- A successful rollout does not replace the analytic proof; a failed negative control is not a product failure.",
            "",
        ]
    )
    return "\n".join(lines)


def render_dashboard(summary: dict[str, Any]) -> str:
    width, height = 1200, 700
    left, top, chart_w, chart_h = 90, 180, 1020, 330
    controllers = summary["controllers"]
    group_w = chart_w / len(controllers)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1200" height="700" fill="#f7fafc"/>',
        '<text x="70" y="65" font-family="Arial,sans-serif" font-size="30" font-weight="700" fill="#17324d">Safe neural control certificate benchmark</text>',
        '<text x="70" y="103" font-family="Arial,sans-serif" font-size="17" fill="#49657d">Closed-loop regression rates; analytic margins shown below</text>',
    ]
    for tick in range(6):
        value = tick / 5
        y = top + chart_h * (1.0 - value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#d7e1e8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 15}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="14" fill="#5f7485">{value:.1f}</text>'
        )
    for index, item in enumerate(controllers):
        center = left + group_w * (index + 0.5)
        safe_h = chart_h * item["box_safe_rate"]
        contraction_h = chart_h * item["practical_contraction_rate"]
        parts.extend(
            [
                f'<rect x="{center - 42:.1f}" y="{top + chart_h - safe_h:.1f}" width="34" height="{safe_h:.1f}" fill="#1976a3" rx="3"/>',
                f'<rect x="{center + 8:.1f}" y="{top + chart_h - contraction_h:.1f}" width="34" height="{contraction_h:.1f}" fill="#e48b2a" rx="3"/>',
                f'<text x="{center:.1f}" y="{top + chart_h + 30}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#263f52">{escape(CONTROLLER_LABELS[item["controller_id"]])}</text>',
            ]
        )
    parts.extend(
        [
            '<rect x="780" y="76" width="16" height="16" fill="#1976a3"/><text x="804" y="90" font-family="Arial,sans-serif" font-size="14" fill="#263f52">Box-safe rate</text>',
            '<rect x="930" y="76" width="16" height="16" fill="#e48b2a"/><text x="954" y="90" font-family="Arial,sans-serif" font-size="14" fill="#263f52">Contraction rate</text>',
        ]
    )
    margin_items = [
        ("Disturbance absorption", summary["analytic_certificate_margins"]["eta_minus_disturbance"]),
        ("Box invariance", summary["analytic_certificate_margins"]["box_invariance"]),
        ("Input authority", summary["analytic_certificate_margins"]["input_authority"]),
        ("Uncertainty width", summary["analytic_certificate_margins"]["uncertainty_interval_width_next_state"]),
    ]
    card_w = 250
    for index, (label, value) in enumerate(margin_items):
        x = 70 + index * 280
        parts.extend(
            [
                f'<rect x="{x}" y="565" width="{card_w}" height="92" fill="#ffffff" stroke="#cbd8e1" rx="8"/>',
                f'<text x="{x + 18}" y="598" font-family="Arial,sans-serif" font-size="14" fill="#5b7183">{escape(label)}</text>',
                f'<text x="{x + 18}" y="635" font-family="Arial,sans-serif" font-size="26" font-weight="700" fill="#17324d">{value:.6f}</text>',
            ]
        )
    parts.append("</svg>\n")
    return "".join(parts)


def write_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8", newline="\n")
