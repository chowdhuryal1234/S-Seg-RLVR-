"""Strict candidate recognition and drawn-geometry readback diagnostics.

The two tasks use the same marked image. Recognition is candidate-assisted,
not free nucleus detection. Geometry scores the drawn squares, not nucleus
boundaries. These are CPU diagnostics, with no SAM, mask PQ or training rewards.
Invalid whole answers accept no entries; raw entries remain available for
forensic inspection without clipping, coordinate repair or selective salvage.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import re


CANDIDATE_IDS = tuple("ABCDEF")
BOX_IOU_THRESHOLD = 0.5
POINT_L2_LIMIT = 7.0
CRITERION = {"all_six_unique_ids_required": True, "format_valid_required": True,
             "box_iou_strictly_greater_than": BOX_IOU_THRESHOLD,
             "point_l2_processed_at_most": POINT_L2_LIMIT,
             "scope": "Predeclared engineering check, not biological segmentation performance."}


def _manifest(manifest):
    candidates = manifest["candidates"]
    if (not isinstance(candidates, list) or len(candidates) != 6
            or {item["id"] for item in candidates} != set(CANDIDATE_IDS)
            or any(type(item["is_nucleus"]) is not bool for item in candidates)):
        raise ValueError("Manifest must contain exactly A-F with boolean nucleus labels")
    if manifest["original_size"] != [128, 128] or manifest["processed_size"] != [224, 224]:
        raise ValueError("This predeclared diagnostic uses native128 and processed224")
    for candidate in candidates:
        for original, processed, size in (("box_original", "box_processed", 4), ("point_original", "point_processed", 2)):
            if not _numbers(candidate[original], size) or not _numbers(candidate[processed], size):
                raise ValueError("Manifest geometry must contain finite numbers")
            if any(not math.isclose(a * 1.75, b, rel_tol=1e-12, abs_tol=1e-12)
                   for a, b in zip(candidate[original], candidate[processed])):
                raise ValueError("Manifest must preserve the declared 224/128 scaling")
    return {candidate["id"]: candidate for candidate in candidates}


def _numbers(value, length):
    try:
        return (isinstance(value, list) and len(value) == length
                and all(type(v) in (int, float) and math.isfinite(v) for v in value))
    except (OverflowError, TypeError):
        return False


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _decode(text, key):
    output = {"format_valid": False, "parse_error": None, "errors": [],
              "json_decoded": False, "raw_payload": None, "raw_entries": [], "entry_checks": []}
    if not isinstance(text, str) or len(text) > 65536:
        output["errors"].append("Expected completion text of at most 65536 characters")
        return output
    match = re.fullmatch(r"<answer>(.*?)</answer>", text.strip(), re.DOTALL)
    if match is None:
        output["errors"].append("Expected exactly one <answer>JSON</answer> block without prose")
        return output
    try:
        payload = json.loads(match.group(1), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as error:
        output["errors"].append(str(error))
        return output
    output.update(json_decoded=True, raw_payload=payload)
    if not isinstance(payload, dict) or set(payload) != {key}:
        output["errors"].append(f"Answer must contain exactly the {key} key")
    entries = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        output["errors"].append(f"{key} must be an array")
    else:
        output["raw_entries"] = entries
    return output


def _finish(parsed):
    parsed["format_valid"] = not parsed["errors"]
    parsed["parse_error"] = "; ".join(parsed["errors"]) if parsed["errors"] else None
    return parsed


def analyze_recognition(text, manifest):
    """Strict {nucleus_regions:[IDs]} parsing plus candidate-level TP/FP/FN."""
    candidates = _manifest(manifest)
    output = _decode(text, "nucleus_regions")
    seen = set()
    for index, entry in enumerate(output["raw_entries"]):
        errors = []
        if not isinstance(entry, str) or entry not in candidates:
            errors.append("Candidate ID must be one of A-F")
        elif entry in seen:
            errors.append("Duplicate candidate ID")
        if isinstance(entry, str):
            seen.add(entry)
        output["entry_checks"].append({"index": index, "id": entry if isinstance(entry, str) else None,
                                        "raw": entry, "valid": not errors, "errors": errors})
        output["errors"].extend(f"Entry {index + 1}: {error}" for error in errors)
    _finish(output)
    accepted = list(output["raw_entries"]) if output["format_valid"] else []
    expected = [key for key in CANDIDATE_IDS if candidates[key]["is_nucleus"]]
    selected, positive = set(accepted), set(expected)
    tp, fp, fn = len(selected & positive), len(selected - positive), len(positive - selected)
    tn = len(candidates) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else float(not positive)
    recall = tp / (tp + fn) if tp + fn else 1.
    output.update(task="candidate_recognition", accepted_ids=accepted, expected_ids=expected,
                  tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                  accuracy=(tp + tn) / len(candidates), f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
                  exact_set_correct=output["format_valid"] and selected == positive,
                  per_candidate=[{"id": key, "expected_positive": key in positive,
                                  "predicted_positive": key in selected, "correct": (key in positive) == (key in selected)} for key in CANDIDATE_IDS],
                  scope="Candidate-assisted recognition of center membership; formal metrics use accepted IDs only. Invalid whole answers accept none; raw entries are not repaired or salvaged.")
    return output


def _entry_errors(entry):
    errors = []
    if not isinstance(entry, dict):
        return ["Region must be an object"]
    if set(entry) != {"id", "box", "point"}:
        errors.append("Region must contain exactly id, box and point")
    if not isinstance(entry.get("id"), str) or entry["id"] not in CANDIDATE_IDS:
        errors.append("Region ID must be one of A-F")
    box, point = entry.get("box"), entry.get("point")
    if not _numbers(box, 4):
        errors.append("Box must contain four finite numbers, not booleans")
    elif not (0 <= box[0] < box[2] <= 224 and 0 <= box[1] < box[3] <= 224):
        errors.append("Box must have positive area and lie inside processed224")
    if not _numbers(point, 2):
        errors.append("Point must contain two finite numbers, not booleans")
    else:
        if not (0 <= point[0] < 224 and 0 <= point[1] < 224):
            errors.append("Point must lie inside processed224")
        if _numbers(box, 4) and not (box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]):
            errors.append("Point must lie inside its half-open box")
    return errors


def _iou(box, reference):
    intersection = max(0., min(box[2], reference[2]) - max(box[0], reference[0])) * max(
        0., min(box[3], reference[3]) - max(box[1], reference[1]))
    return intersection / ((box[2] - box[0]) * (box[3] - box[1])
                           + (reference[2] - reference[0]) * (reference[3] - reference[1]) - intersection)


def analyze_geometry(text, manifest):
    """Strict complete A-F readback, plus separate unchanged-raw geometry errors.

    Bounds errors invalidate the whole answer. A finite, positive-area raw box
    can still receive a forensic IoU using its full area outside the image;
    such an entry remains unaccepted. No nucleus label enters geometry scoring.
    """
    candidates = _manifest(manifest)
    output = _decode(text, "regions")
    counts = Counter(entry.get("id") for entry in output["raw_entries"]
                     if isinstance(entry, dict) and isinstance(entry.get("id"), str))
    duplicate_ids = sorted(key for key, count in counts.items() if count > 1)
    missing_ids = [key for key in CANDIDATE_IDS if key not in counts]
    if len(output["raw_entries"]) != 6:
        output["errors"].append("Geometry answer must contain exactly six regions")
    if missing_ids:
        output["errors"].append("Missing region IDs: " + ",".join(missing_ids))
    if duplicate_ids:
        output["errors"].append("Duplicate region IDs: " + ",".join(duplicate_ids))
    for index, entry in enumerate(output["raw_entries"]):
        errors = _entry_errors(entry)
        key = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(key, str) and key in duplicate_ids:
            errors.append("Duplicate region ID")
        output["entry_checks"].append({"index": index, "id": key, "raw": entry,
                                        "valid": not errors, "errors": errors})
        output["errors"].extend(f"Entry {index + 1}: {error}" for error in errors)
    _finish(output)
    per_region = []
    for check in output["entry_checks"]:
        entry, key = check["raw"], check["id"]
        expected = candidates.get(key) if isinstance(key, str) else None
        box = entry.get("box") if isinstance(entry, dict) else None
        point = entry.get("point") if isinstance(entry, dict) else None
        box_available = expected is not None and _numbers(box, 4) and box[2] > box[0] and box[3] > box[1]
        point_available = expected is not None and _numbers(point, 2)
        edges = [float(a - b) for a, b in zip(box, expected["box_processed"])] if box_available else None
        point_error = [float(a - b) for a, b in zip(point, expected["point_processed"])] if point_available else None
        iou = float(_iou(box, expected["box_processed"])) if box_available else None
        l2 = math.hypot(*point_error) if point_available else None
        per_region.append({"entry_index": check["index"], "id": key, "raw_box": box, "raw_point": point,
                           "expected_box_processed": expected["box_processed"] if expected else None,
                           "expected_point_processed": expected["point_processed"] if expected else None,
                           "per_entry_contract_valid": check["valid"], "contract_errors": check["errors"],
                           "accepted_by_strict_parser": output["format_valid"],
                           "box_metrics_available": bool(box_available), "point_metrics_available": bool(point_available),
                           "edge_errors_processed_xyxy": edges,
                           "box_center_error_processed_xy": [(edges[0] + edges[2]) / 2, (edges[1] + edges[3]) / 2] if edges else None,
                           "box_iou": iou, "point_error_processed_xy": point_error,
                           "point_error_original_xy": [v / 1.75 for v in point_error] if point_error else None,
                           "point_l2_processed": l2, "point_l2_original": l2 / 1.75 if l2 is not None else None,
                           "engineering_entry_pass": check["valid"] and iou is not None and l2 is not None and iou > BOX_IOU_THRESHOLD and l2 <= POINT_L2_LIMIT})
    box_values = [entry["box_iou"] for entry in per_region if entry["box_metrics_available"]]
    point_values = [entry["point_l2_processed"] for entry in per_region if entry["point_metrics_available"]]
    output.update(task="drawn_geometry_readback", accepted_regions=output["raw_entries"] if output["format_valid"] else [],
                  accepted_region_count=len(output["raw_entries"]) if output["format_valid"] else 0,
                  missing_ids=missing_ids, duplicate_ids=duplicate_ids, per_region=per_region,
                  raw_box_metric_entry_count=len(box_values), raw_point_metric_entry_count=len(point_values),
                  mean_box_iou=sum(box_values) / len(box_values) if box_values else None,
                  mean_point_l2_processed=sum(point_values) / len(point_values) if point_values else None,
                  max_point_l2_processed=max(point_values) if point_values else None,
                  engineering_pass=output["format_valid"] and len(per_region) == 6 and all(entry["engineering_entry_pass"] for entry in per_region),
                  criterion=dict(CRITERION),
                  scope="Readback of six drawn square bounds and centers; no nucleus decision, SAM or mask PQ. Continuous per-region errors summarize unmodified raw entries, including rejected answers, and do not imply formal acceptance.")
    return output
