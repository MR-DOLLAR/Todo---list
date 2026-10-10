"""Turns a diagnosis (disease + severity) into a concrete, prioritised treatment plan
and a recovery prognosis."""

import re

INCURABLE_TYPES = {"viral"}
INCURABLE_IDS = {"Orange___Haunglongbing_(Citrus_greening)", "Grape___Esca_(Black_Measles)"}

SEVERITY_MULTIPLIER = {"none": 0.0, "mild": 0.8, "moderate": 1.2, "severe": 1.8}
BASE_RECOVERY_CHANCE = {"mild": 0.92, "moderate": 0.75, "severe": 0.5}
SPREAD_WEIGHT = {"none": 0, "low": 0, "medium": 1, "high": 2}
URGENCY = ["low", "medium", "high", "critical"]

# Actions that are only appropriate once the diagnosis is confirmed: any product
# (spray, fungicide, oil, ...) or removing whole plants/trees.
_PRODUCT = re.compile(
    r"copper|sulfur|sulphur|bordeaux|fungicid|bactericid|insecticid|miticid|bicarbonate|\boil\b|neem|spray|"
    r"chlorothalonil|mancozeb|captan|imidacloprid|thiamethoxam|cyantraniliprole|abamectin|systemic|\bmilk\b|"
    r"bacillus|trichoderma|acibenzolar|oxytetracycline|myclobutanil|azoxystrobin|propiconazole|soap|urea|"
    r"predatory|release|kaolin|sealant|paste|epsom|chelated|fertili",
    re.IGNORECASE)
_DRASTIC = re.compile(r"\b(plants?|trees?|vines?|junipers?|cull)\b|whole plant", re.IGNORECASE)


def _safe(actions):
    """Steps that are fine to take before a diagnosis is confirmed."""
    def drastic(a):
        low = a.lower()
        return (any(w in low for w in ("destroy", "kill", "burn", "trunk renewal"))
                or ("remov" in low and _DRASTIC.search(a)))

    return [a for a in actions if not _PRODUCT.search(a) and not drastic(a)]


def is_curable(disease):
    return disease["type"] not in INCURABLE_TYPES and disease["id"] not in INCURABLE_IDS


def build_plan(disease, severity, confidence, preference="integrated", uncertain=False):
    """preference: 'organic', 'chemical' or 'integrated' (default).

    uncertain: the diagnosis is below the model's confidence threshold, so the
    plan sticks to safe, non-chemical steps until the diagnosis is confirmed.
    """
    level = severity["level"]

    if disease["healthy"]:
        return {
            "urgency": "low",
            "curable": True,
            "approach": "preventive",
            "summary": ("Probably healthy, but the model isn't sure. Look closely for spots, powder or "
                        "discolouration and retake the photo if you see any."
                        if uncertain else "No disease detected. Keep up good growing practices."),
            "steps": [{"phase": "Ongoing", "actions": disease["prevention"]}],
            "prognosis": {"recovery_chance": 1.0, "estimated_recovery_days": 0},
            "warnings": _warnings(confidence, uncertain=uncertain),
        }

    curable = is_curable(disease)
    t = disease["treatment"]
    urgency_idx = {"mild": 0, "moderate": 1, "severe": 2}.get(level, 0) + (
        1 if SPREAD_WEIGHT.get(disease["spread_risk"], 0) >= 2 else 0
    )
    if not curable:
        urgency_idx = max(urgency_idx, 2)
    urgency = URGENCY[min(urgency_idx, 3)]

    if uncertain:
        # Unconfirmed: no products, no removing plants, no alarming urgency or
        # prognosis until the diagnosis is confirmed.
        approach = "confirm first"
        urgency = URGENCY[min(urgency_idx, 1)]
        steps = [
            {"phase": "Today", "actions": [
                "Confirm the diagnosis: retake a clear close-up of one leaf in daylight and select the crop, "
                "or show the plant to a local agricultural extension service or plant clinic",
                "Remove only clearly spotted leaves and put them in the bin (not the compost)",
                "Keep the plant apart from healthy plants and wash your hands and tools after handling it",
            ]},
            {"phase": "Until confirmed", "actions": _dedupe(_safe(t["cultural"] + t["organic"]))},
            {"phase": "Ongoing", "actions": _safe(disease["prevention"])},
        ]
        curability = "" if curable else " This disease can't be cured, so confirm it before removing any plants."
        summary = (f"Possibly {disease['name']} — not confirmed. Until the diagnosis is confirmed, take only the "
                   f"safe steps below: no sprays and no removing plants.{curability}")
        prognosis = {"recovery_chance": None, "estimated_recovery_days": None}
    elif not curable:
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
        "warnings": _warnings(confidence, approach, uncertain),
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


def _warnings(confidence, approach=None, uncertain=False):
    warnings = []
    if uncertain:
        warnings.append(f"The model is not sure about this diagnosis ({confidence:.0%} confidence). Take a "
                        "close-up of a single leaf in daylight, select the crop, or confirm with a local "
                        "agricultural extension service before treating.")
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
