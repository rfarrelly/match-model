from __future__ import annotations


def previous_season(season: str) -> str:
    """
    Converts:
    2425 -> 2324
    2324 -> 2223
    """
    if len(season) != 4 or not season.isdigit():
        raise ValueError(f"Unsupported season format: {season}")

    start = int(season[:2])
    end = int(season[2:])

    prev_start = start - 1
    prev_end = end - 1

    return f"{prev_start:02d}{prev_end:02d}"
