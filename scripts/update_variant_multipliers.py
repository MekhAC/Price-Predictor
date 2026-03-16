import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VARIANT_CSV = ROOT / "models" / "dm_variants.csv"

# ---------------------------------------------------------------------------
# Helpers for token / regex based matching
# ---------------------------------------------------------------------------

def _clean_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value.upper()).strip()
    return text


def _normalize_for_tokens(value: str) -> str:
    text = re.sub(r"[^A-Z0-9+&/ ]", " ", value.upper())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compile_regex(pattern: str) -> re.Pattern:
    return re.compile(pattern, flags=re.IGNORECASE)


def _compile_token(token: str) -> re.Pattern:
    token = token.strip()
    if not token:
        raise ValueError("Empty token")

    if token.startswith("REGEX:"):
        return _compile_regex(token[6:])

    # Allow flexible whitespace between words
    pattern = re.escape(token)
    pattern = pattern.replace(r"\ ", r"\s+")

    if token.endswith("+"):
        base = token[:-1].strip()
        if base:
            pattern = r"\b" + re.escape(base) + r"\+(?!\w)"
        else:
            pattern = r"\+(?!\w)"
    else:
        pattern = r"\b" + pattern + r"\b"
    return re.compile(pattern, flags=re.IGNORECASE)


class LevelMatcher:
    """Return an ordinal score (higher is better) based on matching tokens."""

    def __init__(
        self,
        levels: List[Iterable[str]],
        use_regex: bool = False,
    ) -> None:
        self.pattern_levels: List[List[re.Pattern]] = []
        for level_tokens in levels:
            compiled = []
            for token in level_tokens:
                if use_regex:
                    compiled.append(_compile_regex(token))
                else:
                    compiled.append(_compile_token(token))
            self.pattern_levels.append(compiled)

    def score(self, text: str) -> Optional[int]:
        if not text:
            return None

        best_idx: Optional[int] = None
        for idx, patterns in enumerate(self.pattern_levels):
            for pat in patterns:
                if pat.search(text):
                    best_idx = idx
        if best_idx is None:
            return None
        levels = len(self.pattern_levels)
        if levels <= 1:
            return 3

        ratio = best_idx / (levels - 1)
        mapped = 1 + int(round(ratio * 4))
        return max(1, min(5, mapped))


# ---------------------------------------------------------------------------
# Ordered ladders for specific brands / models
# ---------------------------------------------------------------------------

BRAND_LEVELS = {
    "MARUTI SUZUKI": LevelMatcher(
        [
            ["STD", "STANDARD", "BASE", "TOUR", "SIGMA"],
            ["LXI", "LDI"],
            ["VXI", "VDI", "DELTA"],
            ["ZXI", "ZDI", "ZETA"],
            ["ZXI+", "ZDI+", "ZXI PLUS", "ZDI PLUS", "ALPHA", "ALPHA PLUS"],
        ]
    ),
    "HYUNDAI": LevelMatcher(
        [
            ["D-LITE", "D LITE", "ERA"],
            ["MAGNA", "GL", "GLE", "GLS"],
            ["SPORTZ", "SX EXECUTIVE", "S PLUS"],
            ["ASTA", "SX"],
            ["SX O", "ASTA O", "SIGNATURE", "SIGNATURE O"],
        ]
    ),
    "TATA": LevelMatcher(
        [
            ["XE", "SMART", "PURE"],
            ["XM", "XM+", "XM PLUS", "ADVENTURE"],
            ["XT", "XZ"],
            ["XZ+", "XZ PLUS", "XZA", "XZA+"],
            ["XZA PLUS", "XZ PLUS O", "XZA PLUS O"],
        ]
    ),
    "MAHINDRA": LevelMatcher(
        [
            ["W4", "B4", "N4", "MX", "Z2"],
            ["W6", "B6", "N8", "AX3", "Z4"],
            ["W8", "AX5", "Z6"],
            ["W10", "W11", "S11", "AX7", "Z8"],
            ["AX7 L", "Z8 L"],
        ]
    ),
    "KIA": LevelMatcher(
        [
            ["HTE", "PREMIUM"],
            ["HTK", "PRESTIGE"],
            ["HTK+", "PRESTIGE PLUS"],
            ["HTX", "LUXURY"],
            ["HTX+", "GTX", "LUXURY PLUS", "GTX+", "X LINE"],
        ]
    ),
    "RENAULT": LevelMatcher(
        [
            ["RXE"],
            ["RXL"],
            ["RXT"],
            ["RXZ", "CLIMBER"],
        ]
    ),
    "NISSAN": LevelMatcher(
        [
            ["XE"],
            ["XL"],
            ["XV"],
            ["XV PREMIUM", "XV PREMIUM (O)"],
        ]
    ),
    "VOLKSWAGEN": LevelMatcher(
        [
            ["TRENDLINE"],
            ["COMFORTLINE"],
            ["HIGHLINE"],
            ["HIGHLINE PLUS", "GT"],
            ["GT PLUS", "GTX PLUS", "TOPLINE"],
        ]
    ),
    "SKODA": LevelMatcher(
        [
            ["ACTIVE"],
            ["AMBITION"],
            ["STYLE"],
            ["STYLE PLUS", "MONTE CARLO"],
            ["L&K", "LAURIN"],
        ]
    ),
    "MG": LevelMatcher(
        [
            ["STYLE"],
            ["SUPER"],
            ["SMART"],
            ["SHARP"],
            ["SAVVY", "SAVVY PRO", "SHARP PRO", "EXCLUSIVE"],
        ]
    ),
    "FORD": LevelMatcher(
        [
            ["AMBIENTE", "BASE"],
            ["TREND"],
            ["TITANIUM"],
            ["TITANIUM+", "SIGNATURE"],
            ["S", "SPORT"],
        ]
    ),
    "JEEP": LevelMatcher(
        [
            ["SPORT"],
            ["LONGITUDE"],
            ["LIMITED"],
            ["LIMITED O", "MODEL S", "S"],
        ]
    ),
}

MODEL_REGEX_LEVELS = {
    "Hyundai | Creta": LevelMatcher(
        [
            [r"\bE\b"],
            [r"\bEX\b"],
            [r"\bS\b"],
            [r"\bSX\b"],
            [r"\bSX\s*\(O\)"],
        ],
        use_regex=True,
    ),
    "Hyundai | Creta N Line": LevelMatcher(
        [
            [r"^N8\b"],
            [r"^N10\b"],
        ],
        use_regex=True,
    ),
    "Hyundai | Venue": LevelMatcher(
        [
            [r"\bE\b"],
            [r"\bS\b"],
            [r"\bS\+\b"],
            [r"\bSX\b"],
            [r"\bSX\s*\(O\)"],
        ],
        use_regex=True,
    ),
    "Hyundai | Venue N Line": LevelMatcher(
        [
            [r"\bN6\b"],
            [r"\bN8\b"],
        ],
        use_regex=True,
    ),
    "Hyundai | Verna": LevelMatcher(
        [
            [r"\bE\b"],
            [r"\bEX\b"],
            [r"\bS\b"],
            [r"\bSX\b"],
            [r"\bSX\s*\(O\)"],
        ],
        use_regex=True,
    ),
    "Hyundai | Exter": LevelMatcher(
        [
            [r"\bS\b"],
            [r"\bS\s*\(O\)\b"],
            [r"\bSX\b"],
            [r"\bSX\s*\(O\)\b"],
        ],
        use_regex=True,
    ),
    "Hyundai | Aura": LevelMatcher(
        [
            [r"\bE\b"],
            [r"\bS\b"],
            [r"\bSX\b"],
            [r"\bSX\s*\(O\)\b"],
        ],
        use_regex=True,
    ),
    "Hyundai | Grand i10 Nios": LevelMatcher(
        [
            [r"^ERA\b"],
            [r"^MAGNA\b"],
            [r"^SPORTZ\b"],
            [r"^ASTA\b"],
        ],
        use_regex=True,
    ),
    "Hyundai | Grand i10": LevelMatcher(
        [
            [r"^ERA\b"],
            [r"^MAGNA\b"],
            [r"^SPORTZ\b"],
            [r"^ASTA\b"],
        ],
        use_regex=True,
    ),
    "Hyundai | Elite i20": LevelMatcher(
        [
            [r"^ERA\b"],
            [r"^MAGNA\b"],
            [r"^SPORTZ\b"],
            [r"^ASTA\b"],
        ],
        use_regex=True,
    ),
    "Hyundai | NEW I20": LevelMatcher(
        [
            [r"^MAGNA\b"],
            [r"^SPORTZ\b"],
            [r"^ASTA\b"],
            [r"^ASTA\s*\(O\)\b"],
        ],
        use_regex=True,
    ),
    "Hyundai | NEW I20 N LINE": LevelMatcher(
        [
            [r"^N6\b"],
            [r"^N8\b"],
        ],
        use_regex=True,
    ),
    "Toyota | Glanza": LevelMatcher(
        [
            [r"\bE\b"],
            [r"\bS\b"],
            [r"\bG\b"],
            [r"\bV\b"],
        ],
        use_regex=True,
    ),
    "Toyota | Urban Cruiser": LevelMatcher(
        [
            [r"^MID"],
            [r"^HIGH"],
            [r"^PREMIUM"],
        ],
        use_regex=True,
    ),
    "Toyota | Urban Cruiser Hyryder": LevelMatcher(
        [
            [r"\bE\b", r"\bS\b"],
            [r"\bG\b"],
            [r"\bV\b"],
        ],
        use_regex=True,
    ),
    "Toyota | YARIS": LevelMatcher(
        [
            [r"\bJ\b"],
            [r"\bG\b"],
            [r"\bV\b"],
            [r"\bVX\b"],
        ],
        use_regex=True,
    ),
    "Toyota | Corolla Altis": LevelMatcher(
        [
            [r"\bJ\b"],
            [r"\bG\b"],
            [r"\bGL\b"],
            [r"\bVL\b", r"\bVX\b"],
        ],
        use_regex=True,
    ),
    "Honda | Amaze": LevelMatcher(
        [
            [r"\bE\b"],
            [r"\bS\b"],
            [r"\bV\b"],
            [r"\bVX\b"],
            [r"\bZX\b"],
        ],
        use_regex=True,
    ),
    "Honda | City": LevelMatcher(
        [
            [r"\bS\b"],
            [r"\bV\b"],
            [r"\bVX\b"],
            [r"\bZX\b"],
        ],
        use_regex=True,
    ),
    "Honda | Jazz": LevelMatcher(
        [
            [r"\bS\b"],
            [r"\bSV\b"],
            [r"\bV\b"],
            [r"\bVX\b"],
            [r"\bZX\b"],
        ],
        use_regex=True,
    ),
    "Honda | WR-V": LevelMatcher(
        [
            [r"\bS\b"],
            [r"\bSV\b"],
            [r"\bVX\b"],
        ],
        use_regex=True,
    ),
    "Tata | PUNCH": LevelMatcher(
        [
            ["PURE"],
            ["ADVENTURE"],
            ["ACCOMPLISHED"],
            ["CREATIVE"],
        ]
    ),
    "Tata | Punch": LevelMatcher(
        [
            ["PURE"],
            ["ADVENTURE"],
            ["ACCOMPLISHED"],
            ["CREATIVE"],
        ]
    ),
    "Tata | Nexon": LevelMatcher(
        [
            ["XE", "SMART"],
            ["XM", "PURE"],
            ["XT", "CREATIVE"],
            ["XZ", "XZ+", "FEARLESS"],
            ["XZ+ O", "FEARLESS+", "EMPOWERED"],
        ]
    ),
    "Tata | Nexon EV": LevelMatcher(
        [
            ["XM", "XR"],
            ["XZ", "MR"],
            ["XZ+", "LR"],
            ["XZ+ LUX", "EMPOWERED"],
        ]
    ),
    "Tata | Altroz": LevelMatcher(
        [
            ["XE"],
            ["XM"],
            ["XT"],
            ["XZ"],
            ["XZ+", "XZ PLUS"],
        ]
    ),
    "Tata | Tiago": LevelMatcher(
        [
            ["XE"],
            ["XM"],
            ["XT"],
            ["XZ"],
            ["XZ+", "XZ PLUS"],
        ]
    ),
    "Tata | Tigor": LevelMatcher(
        [
            ["XE"],
            ["XM"],
            ["XZ"],
            ["XZ+", "XZ PLUS"],
        ]
    ),
    "Mahindra | XUV700": LevelMatcher(
        [
            ["MX"],
            ["AX 3", "AX3"],
            ["AX 5", "AX5"],
            ["AX 7", "AX7"],
            ["AX 7 L", "AX7 L"],
        ]
    ),
    "Mahindra | Scorpio-N": LevelMatcher(
        [
            ["Z2"],
            ["Z4"],
            ["Z6"],
            ["Z8"],
            ["Z8 L"],
        ]
    ),
    "Mahindra | Scorpio N": LevelMatcher(
        [
            ["Z2"],
            ["Z4"],
            ["Z6"],
            ["Z8"],
            ["Z8 L"],
        ]
    ),
    "Mahindra | Bolero": LevelMatcher(
        [
            ["B4", "LX"],
            ["B6"],
            ["B6 O", "ZLX"],
        ]
    ),
    "Mahindra | Bolero Neo": LevelMatcher(
        [
            ["N4"],
            ["N8"],
            ["N10"],
            ["N10 O"],
        ]
    ),
    "Mahindra | XUV300": LevelMatcher(
        [
            ["W4"],
            ["W6"],
            ["W8"],
            ["W8 O"],
        ]
    ),
    "Mahindra | XUV500": LevelMatcher(
        [
            ["W4"],
            ["W6"],
            ["W8"],
            ["W10", "W11"],
        ]
    ),
    "Toyota | Innova Crysta": LevelMatcher(
        [
            ["G"],
            ["GX"],
            ["VX"],
            ["ZX"],
        ]
    ),
    "Toyota | Innova": LevelMatcher(
        [
            ["E"],
            ["G", "GX"],
            ["V", "VX"],
            ["Z", "ZX"],
        ]
    ),
}

# ---------------------------------------------------------------------------
# General token-based adjustments
# ---------------------------------------------------------------------------

TOKEN_TIERS = {
    5: [
        "ALPHA",
        "ALPHA PLUS",
        "ALPHA SMART HYBRID",
        "ZXI PLUS",
        "ZDI PLUS",
        "ZXI+",
        "ZDI+",
        "XZ PLUS",
        "XZA PLUS",
        "XZ+",
        "XZA+",
        "SX O",
        "SIGNATURE O",
        "SIGNATURE PLUS",
        "PLATINUM O",
        "LIMITED O",
        "LIMITED PLUS",
        "LUXURY PLUS",
        "MODEL S",
        "SAVVY PRO",
        "SAVVY",
        "SHARP PRO",
        "EXCLUSIVE",
        "BLACKSTORM",
        "Z8 L",
        "AX7 L",
        "AX 7 L",
        "LUXURY PACK",
        "L&K",
        "LAURIN",
        "AMG",
        "M SPORT",
        "X LINE",
        "GTX PLUS",
        "GTX+",
        "GT PLUS",
        "GT+",
        "TOPLINE",
        "FEARLESS+",
        "FEARLESS PLUS",
        "EMPOWERED PLUS",
        "XZ PLUS O",
        "XZA PLUS O",
        "TECHNOLOGY",
    ],
    4: [
        "ZXI",
        "ZDI",
        "ZETA",
        "XZ",
        "XZA",
        "GTX",
        "GT LINE",
        "SIGNATURE",
        "PLATINUM",
        "LIMITED",
        "LUXURY",
        "SHARP",
        "CREATIVE",
        "ACCOMPLISHED",
        "TITANIUM",
        "HIGHLINE",
        "HTX",
        "PRESTIGE",
        "VX",
        "ZX",
        "MATRIX",
        "S LINE",
        "SLINE",
        "SPORT LINE",
        "LUXURY LINE",
        "PREMIUM PLUS",
    ],
    2: [
        "LXI",
        "LDI",
        "SIGMA",
        "XE",
        "XM",
        "RXL",
        "MAGNA",
        "AMBIENTE",
    ],
    1: [
        "STD",
        "STANDARD",
        "BASE",
        "TOUR",
        "FLEET",
        "COMMERCIAL",
        "RXE",
        "W4",
        "B4",
        "MX",
    ],
}

TOKEN_PATTERNS = {
    score: [_compile_token(token) for token in tokens]
    for score, tokens in TOKEN_TIERS.items()
}

# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


def _initial_score(brand: str, parent: str, raw_name: str, norm_name: str) -> int:
    score: Optional[int] = None

    matcher = MODEL_REGEX_LEVELS.get(parent)
    if matcher is None:
        matcher = MODEL_REGEX_LEVELS.get(_clean_text(parent))

    if matcher:
        score = matcher.score(raw_name)
        if score is None:
            score = matcher.score(norm_name)

    if score is None:
        brand_matcher = BRAND_LEVELS.get(brand)
        if brand_matcher:
            score = brand_matcher.score(norm_name)

    return score if score is not None else 3


def adjust_with_tokens(score: int, norm_name: str) -> int:
    current = score
    # Entry-level overrides
    for pattern in TOKEN_PATTERNS[1]:
        if pattern.search(norm_name):
            return 1

    # Lower trims
    for pattern in TOKEN_PATTERNS[2]:
        if pattern.search(norm_name):
            current = min(current, 2)

    # Upper trims
    for pattern in TOKEN_PATTERNS[4]:
        if pattern.search(norm_name):
            current = max(current, 4)

    for pattern in TOKEN_PATTERNS[5]:
        if pattern.search(norm_name):
            current = max(current, 5)

    return max(1, min(5, current))


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    if not VARIANT_CSV.exists():
        raise FileNotFoundError(f"Could not find {VARIANT_CSV}")

    df = pd.read_csv(VARIANT_CSV)
    if df.empty:
        raise ValueError("dm_variants.csv is empty")

    brand_cache = {}
    raw_scores = []

    for _, row in df.iterrows():
        parent = row["parent"]
        brand = brand_cache.get(parent)
        if brand is None:
            brand = parent.split("|")[0].strip().upper()
            brand_cache[parent] = brand

        name = str(row["name"])
        raw_name = _clean_text(name)
        norm_name = _normalize_for_tokens(name)

        base_score = _initial_score(brand, parent, raw_name, norm_name)
        adjusted = adjust_with_tokens(base_score, norm_name)
        raw_scores.append(adjusted)

    df["_raw_score"] = raw_scores
    df["_score"] = df["_raw_score"]

    multiplier_map = {
        5: 1.07,
        4: 1.04,
        3: 1.00,
        2: 0.97,
        1: 0.94,
    }

    df["multiplier"] = df["_score"].map(multiplier_map)
    df["multiplier"] = df["multiplier"].round(2)

    df = df.drop(columns=["_raw_score", "_score"])

    df.to_csv(VARIANT_CSV, index=False)
    print(f"Updated multipliers for {len(df)} variants -> {VARIANT_CSV}")


if __name__ == "__main__":
    main()
