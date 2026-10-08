"""Turns a diagnosis (disease + severity) into a concrete, prioritised treatment plan
and a recovery prognosis."""

INCURABLE_TYPES = {"viral"}
INCURABLE_IDS = {"Orange___Haunglongbing_(Citrus_greening)", "Grape___Esca_(Black_Measles)"}

SEVERITY_MULTIPLIER = {"none": 0.0, "mild": 0.8, "moderate": 1.2, "severe": 1.8}
BASE_RECOVERY_CHANCE = {"mild": 0.92, "moderate": 0.75, "severe": 0.5}
SPREAD_WEIGHT = {"none": 0, "low": 0, "medium": 1, "high": 2}
URGENCY = ["low", "medium", "high", "critical"]


def is_curable(disease):
    return disease["type"] not in INCURABLE_TYPES and disease["id"] not in INCURABLE_IDS


def build_plan(disease, severity, confidence, preference="integrated"):
    """preference: 'organic', 'chemical' or 'integrated' (default)."""
    level = severity["level"]

    if disease["healthy"]:
        return {
            "urgency": "low",
            "curable": True,
            "approach": "preventive",
            "summary": "No disease detected. Keep up good growing practices.",
            "steps": [{"phase": "Ongoing", "actions": disease["prevention"]}],
            "prognosis": {"recovery_chance": 1.0, "estimated_recovery_days": 0},
            "warnings": _warnings(confidence),
        }

    curable = is_curable(disease)
    t = disease["treatment"]
    urgency_idx = {"mild": 0, "moderate": 1, "severe": 2}.get(level, 0) + (
        1 if SPREAD_WEIGHT.get(disease["spread_risk"], 0) >= 2 else 0
    )
    if not curable:
        urgency_idx = max(urgency_idx, 2)
    urgency = URGENCY[min(urgency_idx, 3)]

    if not curable:
        approach = "containment"
        immediate = ["Isolate the plant and avoid touching healthy plants after handling it"]
        immediate += [a for a in t["organic"] + t["chemical"] if "remove" in a.lower() or "destroy" in a.lower()]
        control = [a for a in t["organic"] + t["chemical"] if a not in immediate and "no " not in a.lower()[:3]]
        steps = [
            {"phase": "Today", "actions": _dedupe(immediate)},
            {"phase": "Days 1-14", "actions": _dedupe(control + t["cultural"])},
            {"phase": "Ongoing", "actions": disease["prevention"]},
        ]
        summary = (f"{disease['name']} cannot be cured once a plant is infected. "
                   "Focus on removing infected plants and stopping the spread to healthy ones.")
        prognosis = {"recovery_chance": 0.0, "estimated_recovery_days": None}
    else:
        approach = _choose_approach(level, preference)
        primary = _primary_treatments(t, approach)
        steps = [
            {"phase": "Today", "actions": _dedupe(_immediate_actions(level) + primary[:1])},
            {"phase": "Days 1-7", "actions": _dedupe(primary[1:] + t["cultural"])},
            {"phase": "Weeks 2-3", "actions": [
                "Re-inspect leaves every 3-4 days and photograph progress",
                "Repeat treatment at the label interval if new lesions appear",
                "Rotate fungicide / bactericide groups to prevent resistance" if approach != "organic"
                else "Re-apply organic sprays after rain",
            ]},
            {"phase": "Ongoing", "actions": disease["prevention"]},
        ]
        base_days = disease.get("recovery_days") or 21
        prognosis = {
            "recovery_chance": BASE_RECOVERY_CHANCE.get(level, 0.8),
            "estimated_recovery_days": round(base_days * SEVERITY_MULTIPLIER.get(level, 1.0)),
        }
        summary = _summary(disease, level, approach)

    return {
        "urgency": urgency,
        "curable": curable,
        "approach": approach,
        "summary": summary,
        "steps": [s for s in steps if s["actions"]],
        "prognosis": prognosis,
        "warnings": _warnings(confidence, approach),
    }


def _choose_approach(level, preference):
    if preference in ("organic", "chemical"):
        return preference
    return "organic" if level == "mild" else "integrated"


def _primary_treatments(t, approach):
    if approach == "organic":
        return t["organic"]
    if approach == "chemical":
        return t["chemical"] + t["organic"][:1]
    return t["organic"][:2] + t["chemical"]


def _immediate_actions(level):
    actions = ["Remove visibly infected leaves and dispose of them away from the garden (do not compost)"]
    if level == "severe":
        actions.append("Consider removing the whole plant if more than half the foliage is affected")
    actions.append("Disinfect pruning tools with 70% alcohol or 10% bleach after use")
    return actions


def _summary(disease, level, approach):
    tone = {
        "mild": "Caught early — this is very treatable.",
        "moderate": "The infection is established; act promptly to stop it spreading.",
        "severe": "The infection is advanced; aggressive treatment is needed to save the plant.",
    }.get(level, "")
    method = {"organic": "organic", "chemical": "chemical", "integrated": "integrated (organic + chemical)"}[approach]
    where = f" on {disease['crop'].lower()}" if disease["crop"] != "Unknown" else ""
    return f"{disease['name']}{where} detected. {tone} Recommended: {method} treatment."


def _warnings(confidence, approach=None):
    warnings = []
    if confidence < 0.6:
        warnings.append("Low prediction confidence. Take a clearer photo of a single leaf in good light, "
                        "or confirm with a local agricultural extension service.")
    if approach in ("chemical", "integrated"):
        warnings.append("Always follow the product label, wear protective equipment and observe "
                        "pre-harvest intervals. Check which products are approved in your region.")
    return warnings


def _dedupe(items):
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out
