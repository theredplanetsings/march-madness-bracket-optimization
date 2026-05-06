"""Normalize team names to canonical (mostly-KenPom) form.

KenPom names also drift across years (e.g. "FGCU" vs "Florida Gulf Coast",
"LIU" vs "LIU Brooklyn", "Louisiana" vs "Louisiana Lafayette"), so we
normalize *both* the tournament data and KenPom data to a single canonical
key per team. The canonical form chosen is the most common KenPom rendering.

Use `canonicalize(name)` on every team name from any source before joining.
"""
from __future__ import annotations
import re

# Canonical mapping. Key is any encountered variant, value is canonical form.
ALIASES: dict[str, str] = {
    # ---- KenPom internal drift across years (canonicalize both sides) ----
    "FGCU": "Florida Gulf Coast",
    "LIU Brooklyn": "LIU",
    "Long Island University": "LIU",
    "Long Island": "LIU",
    "Louisiana Lafayette": "Louisiana",
    "UL Lafayette": "Louisiana",
    "ULL": "Louisiana",
    "Arkansas Little Rock": "Little Rock",
    "UALR": "Little Rock",
    "Charleston": "College of Charleston",
    "Charleston (SC)": "College of Charleston",
    "McNeese St.": "McNeese",
    "SIUE": "SIU Edwardsville",
    "SIU-Edwardsville": "SIU Edwardsville",
    "Sam Houston St.": "Sam Houston",
    "California Baptist": "Cal Baptist",
    # ---- External (sports-reference, ESPN, etc.) -> canonical ----
    "UConn": "Connecticut",
    "UNC": "North Carolina",
    "NC State": "N.C. St.",
    "NC St.": "N.C. St.",
    "North Carolina St.": "N.C. St.",
    "North Carolina State": "N.C. St.",
    "Pitt": "Pittsburgh",
    "Ole Miss": "Mississippi",
    "Brigham Young": "BYU",
    "Arkansas-Pine Bluff": "Arkansas Pine Bluff",
    "Gardner-Webb": "Gardner Webb",
    "Grambling": "Grambling St.",
    "Omaha": "Nebraska Omaha",
    "Pennsylvania": "Penn",
    "Southern California": "USC",
    "St. John's (NY)": "St. John's",
    "Saint John's": "St. John's",
    "Saint John's (NY)": "St. John's",
    "Massachusetts": "Massachusetts",
    "UMass": "Massachusetts",
    "Saint Mary's (CA)": "Saint Mary's",
    "St. Mary's": "Saint Mary's",
    "St. Joseph's": "Saint Joseph's",
    "Saint Bonaventure": "St. Bonaventure",
    "Saint Louis": "Saint Louis",
    "St. Louis": "Saint Louis",
    "St. Peter's": "Saint Peter's",
    "Saint Francis (PA)": "Saint Francis PA",
    "St. Francis (PA)": "Saint Francis PA",
    "St. Francis (NY)": "St. Francis NY",
    "Saint Francis (NY)": "St. Francis NY",
    "Cal State Fullerton": "Cal St. Fullerton",
    "Cal State Bakersfield": "Cal St. Bakersfield",
    "Cal State Northridge": "Cal St. Northridge",
    "Loyola (IL)": "Loyola Chicago",
    "Loyola-Chicago": "Loyola Chicago",
    "Loyola (MD)": "Loyola MD",
    "Detroit Mercy": "Detroit",
    "Texas-San Antonio": "UTSA",
    "Texas-Arlington": "UT Arlington",
    "Texas Rio Grande Valley": "UT Rio Grande Valley",
    "UTRGV": "UT Rio Grande Valley",
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "Miami": "Miami FL",
    "Bethune-Cookman": "Bethune Cookman",
    "Albany (NY)": "Albany",
    "UMass Lowell": "UMass Lowell",
    "Maryland-Baltimore County": "UMBC",
    "Maryland-Eastern Shore": "Maryland Eastern Shore",
    "Texas A&M-Corpus Christi": "Texas A&M Corpus Chris",
    "Texas A&M-CC": "Texas A&M Corpus Chris",
    "Texas A&M Corpus Christi": "Texas A&M Corpus Chris",
    "Central Florida": "UCF",
    "ETSU": "East Tennessee St.",
    "East Tennessee State": "East Tennessee St.",
    "Cal-Irvine": "UC Irvine",
    "UC-Irvine": "UC Irvine",
    "Cal-Santa Barbara": "UC Santa Barbara",
    "UC-Santa Barbara": "UC Santa Barbara",
    "Cal-Davis": "UC Davis",
    "UC-Davis": "UC Davis",
    "Cal-Riverside": "UC Riverside",
    "UC-Riverside": "UC Riverside",
    "Cal-San Diego": "UC San Diego",
    "UC-San Diego": "UC San Diego",
    "Mt. St. Mary's": "Mount St. Mary's",
    "FDU": "Fairleigh Dickinson",
    "F.D.U.": "Fairleigh Dickinson",
    "WKU": "Western Kentucky",
    "Hawai'i": "Hawaii",
    "Long Beach State": "Long Beach St.",
    "Queens (NC)": "Queens",
}


def _state_to_st(name: str) -> str:
    """Replace trailing or embedded ' State' with ' St.' to match KenPom."""
    return re.sub(r"\bState\b(?=$|[^a-zA-Z])", "St.", name).strip()


def canonicalize(name: str) -> str:
    """Map any team name (KenPom, tourney, ESPN) to canonical form."""
    if name is None:
        return name
    n = str(name).strip()
    if n in ALIASES:
        return ALIASES[n]
    transformed = _state_to_st(n)
    if transformed in ALIASES:
        return ALIASES[transformed]
    return transformed


# Backwards-compatible alias
normalize = canonicalize
