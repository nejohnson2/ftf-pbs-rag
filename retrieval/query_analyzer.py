"""Query analyzer: extract structured entities from a user question.

Uses rule-based patterns against a known vocabulary (20 countries, phase
keywords, survey-type keywords, year patterns). No NLP model required.

Returns a QueryEntities object used by hybrid_retriever to pre-filter
the vector and BM25 searches by metadata.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# ── Known vocabulary ──────────────────────────────────────────────────────────

COUNTRIES = {
    "Bangladesh", "Cambodia", "Ethiopia", "Ghana", "Guatemala", "Haiti",
    "Honduras", "Kenya", "Liberia", "Malawi", "Mali", "Mozambique",
    "Nepal", "Nigeria", "Rwanda", "Senegal", "Tajikistan", "Tanzania",
    "Uganda", "Zambia",
}

# Aliases / partial matches that uniquely map to a canonical country
COUNTRY_ALIASES: dict[str, str] = {
    "tajik": "Tajikistan",
    "guatemalan": "Guatemala",
    "tanzanian": "Tanzania",
    "ugandan": "Uganda",
    "kenyan": "Kenya",
    "malawian": "Malawi",
    "zambian": "Zambia",
    "ghanaian": "Ghana",
    "liberian": "Liberia",
    "haitian": "Haiti",
    "honduran": "Honduras",
    "nepali": "Nepal",
    "nepalese": "Nepal",
    "rwandan": "Rwanda",
    "senegalese": "Senegal",
    "nigerian": "Nigeria",
    "malian": "Mali",
    "mozambican": "Mozambique",
    "ethiopian": "Ethiopia",
    "cambodian": "Cambodia",
    "bangladeshi": "Bangladesh",
}

SURVEY_TYPE_KEYWORDS: dict[str, str] = {
    "baseline": "baseline",
    "midline": "midline",
    "endline": "endline",
    "interim": "interim",
    "end-line": "endline",
    "end line": "endline",
}

PHASE_PATTERNS = [
    (re.compile(r"\bphase\s+(?:i{1,3}|iv)\b", re.IGNORECASE), lambda m: _roman_to_int(m.group(0))),
    (re.compile(r"\bphase\s+([1-4])\b", re.IGNORECASE), lambda m: int(re.search(r"\d", m.group(0)).group())),
    (re.compile(r"\bp([1-4])\b", re.IGNORECASE), lambda m: int(re.search(r"\d", m.group(0)).group())),
]

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _roman_to_int(text: str) -> int:
    numeral = text.lower().split()[-1]
    mapping = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
    return mapping.get(numeral, 1)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class QueryEntities:
    countries: list[str] = field(default_factory=list)
    phases: list[int] = field(default_factory=list)
    survey_types: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)

    def has_filters(self) -> bool:
        return bool(self.countries or self.phases or self.survey_types or self.years)

    def to_metadata_filter(self) -> Optional[dict]:
        """Build a pgvector / BM25 compatible filter dict."""
        filters = {}

        if len(self.countries) == 1:
            filters["country"] = self.countries[0]
        elif len(self.countries) > 1:
            filters["country"] = {"$in": self.countries}

        if len(self.phases) == 1:
            filters["phase"] = self.phases[0]
        elif len(self.phases) > 1:
            filters["phase"] = {"$in": self.phases}

        if len(self.survey_types) == 1:
            filters["survey_type"] = self.survey_types[0]
        elif len(self.survey_types) > 1:
            filters["survey_type"] = {"$in": self.survey_types}

        return filters if filters else None


# ── Analyzer ──────────────────────────────────────────────────────────────────

def analyze(query: str) -> QueryEntities:
    """Extract structured entities from a free-text query."""
    entities = QueryEntities()
    lower = query.lower()

    # Countries
    for country in COUNTRIES:
        if country.lower() in lower:
            entities.countries.append(country)
    for alias, canonical in COUNTRY_ALIASES.items():
        if alias in lower and canonical not in entities.countries:
            entities.countries.append(canonical)

    # Phases
    for pattern, extractor in PHASE_PATTERNS:
        for match in pattern.finditer(query):
            phase = extractor(match)
            if phase not in entities.phases:
                entities.phases.append(phase)

    # Survey types
    for keyword, normalized in SURVEY_TYPE_KEYWORDS.items():
        if keyword in lower and normalized not in entities.survey_types:
            entities.survey_types.append(normalized)

    # Years
    entities.years = [int(m.group()) for m in YEAR_PATTERN.finditer(query)]

    return entities
