from __future__ import annotations


def parse_layer_selection(
    spec: str | None,
    *,
    total_layers: int | None = None,
    all_as_none: bool = False,
    allow_none_token: bool = False,
) -> list[int] | None:
    stripped = "" if spec is None else str(spec).strip()
    if stripped == "" or stripped.lower() == "all":
        if not all_as_none and total_layers is None:
            raise ValueError("Ambiguous layer selection: 'all' requested but total_layers is not provided.")
        return None if all_as_none else list(range(int(total_layers or 0)))
    if allow_none_token and stripped.lower() in {"none", "off", "disable", "disabled"}:
        return []

    selected: set[int] = set()
    for part in stripped.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {token}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))

    if total_layers is not None:
        for idx in selected:
            if idx < 0 or idx >= int(total_layers):
                raise ValueError(f"Layer index out of bounds: {idx}. Valid range is 0 to {int(total_layers) - 1}.")

    return sorted(selected)
