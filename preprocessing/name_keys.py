"""
name_keys.py
------------
One shared definition of how district / state names are turned into match keys,
so the boundary files, accessibility CSV, satellite patch file names, VGG
features and NFHS-5 ground truth are all joined the same way.

NOTE: area_id restarts at 1 inside every state in the boundary files, so it is
NOT a unique key. Always join on (state key, district key).
"""

import re

# FAO GAUL (2015) state names -> NFHS-5 state names (both as clean keys)
STATE_KEY_ALIASES = {
    "orissa": "odisha",
    "andamanandnicobar": "andamannicobarislands",
    "delhi": "nctdelhi",
    "damananddiu": "dadranagarhavelidamandiu",
    "dadraandnagarhaveli": "dadranagarhavelidamandiu",
    "jammuandkashmir": "jammukashmir",
}

# GAUL 2015 still lists the districts of Telangana (created 2014) under Andhra
# Pradesh. This is the ONLY allowed cross-state fallback in matching.
STATE_FALLBACKS = {"andhrapradesh": ["telangana"]}

# Words that distinguish different districts even when the rest of the name is
# identical (e.g. Bangalore Urban vs Bangalore Rural). Fuzzy matching must
# never treat these as spelling noise.
DISTINGUISHING_WORDS = {
    "urban", "rural", "north", "south", "east", "west", "central",
    "upper", "lower", "new", "old", "suburban",
}


def clean_key(value) -> str:
    """Lower-case and drop everything that is not a letter or digit."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def state_key(value) -> str:
    k = clean_key(value)
    return STATE_KEY_ALIASES.get(k, k)


def distinguishing_tokens(value) -> set:
    return {t for t in re.findall(r"[a-z]+", str(value).lower()) if t in DISTINGUISHING_WORDS}
