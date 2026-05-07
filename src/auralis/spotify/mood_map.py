"""Map an Auralis mood point onto a Spotify Search query.

Spotify deprecated the seed-based ``/recommendations`` and the genre-seed list
in November 2024, so anything we want to surface has to come back through the
public Search API. The Search endpoint understands free-text queries plus a
small set of field filters (``year:``, ``genre:``, ``tag:new``). We translate
the four valence/arousal quadrants into a *pool* of reasonable text queries
and pick one at random per request, so the same mood doesn't return the same
six tracks every time the user re-analyses something.

The query strings here are deliberately broad so that newer releases keep
showing up over time. They are also intentionally English-language friendly,
which matches the dataset Auralis was trained on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class MoodQuery:
    """The query payload Spotify's Search endpoint expects.

    ``queries`` is a small pool of phrasings for the same mood quadrant. The
    backend picks one per request to vary the results. The ``q`` property
    surfaces a single chosen query for compatibility with code that wants a
    deterministic value (e.g. for logging the query that was actually used).
    """

    queries: list[str]
    quadrant: str
    rationale: str

    def pick(self) -> str:
        return random.choice(self.queries) if self.queries else ""

    @property
    def q(self) -> str:
        # Convenience for callers that want one representative query string.
        return self.queries[0] if self.queries else ""


# Russell's circumplex split into four quadrants.
# valence > 0 = positive affect, arousal > 0 = activated.
#
# Query phrasing intentionally biases toward *vocal* tracks (singer-songwriter,
# pop, indie, hip-hop, R&B, etc.) rather than instrumental / lo-fi / ambient
# / "easy listening" pools. Spotify's search is keyword-matched, and queries
# like "lo-fi" or "ambient" pull a lot of background-music instrumentals
# that read as filler in a chat recommendation. Listeners who explicitly
# want instrumentals can ask for them — the LLM is instructed to default
# to lyric-bearing picks otherwise.
_QUADRANT_QUERIES = {
    "HVHA": MoodQuery(
        queries=[
            "upbeat pop vocal",
            "energetic hip hop anthem",
            "feel good rnb",
            "sing-along dance pop",
            "uplifting indie pop",
            "celebration anthem vocal",
        ],
        quadrant="HVHA",
        rationale="High valence, high arousal. Lifted and energised.",
    ),
    "LVHA": MoodQuery(
        queries=[
            "intense rap vocal",
            "edgy alternative rock vocal",
            "raw introspective hip hop",
            "fierce singer-songwriter",
            "powerful indie rock vocal",
            "brooding alt-pop",
        ],
        quadrant="LVHA",
        rationale="Low valence, high arousal. Tense and forceful.",
    ),
    "LVLA": MoodQuery(
        queries=[
            "sad indie singer-songwriter",
            "heartbreak ballad vocal",
            "introspective rnb",
            "wistful indie folk",
            "melancholic alt-pop vocal",
            "late night reflective vocals",
        ],
        quadrant="LVLA",
        rationale="Low valence, low arousal. Heavy and reflective.",
    ),
    "HVLA": MoodQuery(
        queries=[
            "warm acoustic singer-songwriter",
            "mellow indie pop vocal",
            "chill rnb vocal",
            "cozy folk-pop",
            "soft alt-pop vocal",
            "laid back vocal indie",
        ],
        quadrant="HVLA",
        rationale="High valence, low arousal. Soft and content.",
    ),
}


# Discrete-emotion fallbacks for legacy callers that don't pass valence/arousal.
_DISCRETE_QUERIES = {
    "happy":     _QUADRANT_QUERIES["HVHA"],
    "energetic": _QUADRANT_QUERIES["HVHA"],
    "sad":       _QUADRANT_QUERIES["LVLA"],
    "calm":      _QUADRANT_QUERIES["HVLA"],
}


def quadrant_for(valence: float, arousal: float) -> str:
    """Return the four-letter quadrant code for a (valence, arousal) point."""
    high_v = valence >= 0.0
    high_a = arousal >= 0.0
    if high_v and high_a:
        return "HVHA"
    if not high_v and high_a:
        return "LVHA"
    if not high_v and not high_a:
        return "LVLA"
    return "HVLA"


def mood_to_query(
    valence: float | None = None,
    arousal: float | None = None,
    discrete_emotion: str | None = None,
) -> MoodQuery:
    """Pick a query *pool* for a mood.

    Prefers (valence, arousal) when present. Falls back to the discrete
    emotion label, then finally to a neutral chill pool if nothing is given.
    """

    if valence is not None and arousal is not None:
        quad = quadrant_for(float(valence), float(arousal))
        return _QUADRANT_QUERIES[quad]

    if discrete_emotion:
        key = discrete_emotion.strip().lower()
        if key in _DISCRETE_QUERIES:
            return _DISCRETE_QUERIES[key]

    return _QUADRANT_QUERIES["HVLA"]
