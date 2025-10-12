"""
Built-in fallback themes used across the pipeline.

These are appended to model-extracted themes and use **reserved negative IDs**:
  - None         -> -1  (silence/noise/unintelligible)
  - Introduction -> -2  (interviewee introduces themselves)
  - Interviewer  -> -3  (questions/prompts from interviewer)

Call `get_default_themes()` to obtain a copy suitable for merging with generated themes.
"""
from typing import Any
DEFAULT_THEMES = [
    {
        "title": "None",
        "description": "No relevant speech or theme; includes silence, background noise, or unintelligible audio.",
        "keywords": ["silence", "noise", "background", "empty", "unclear"],
        "theme_id": -1
    },
    {
        "title": "Introduction",
        "description": "A speaker introducing themselves.",
        "keywords": ["name", "introduce", "spell", "background", "who am I", "my name is"],
        "theme_id": -2
    },
    {
        "title": "Interviewer",
        "description": "Spoken content from the interviewer such as questions or prompts.",
        "keywords": ["question", "can you", "tell me about", "talk about", "what's your"],
        "theme_id": -3
    }
]

# Single source of truth for reserved IDs (negative to avoid clashing with model-assigned IDs that start at 0+)
THEME_ID_MAP = {
    "None": -1,
    "Introduction": -2,
    "Interviewer": -3
}


def get_default_themes() -> list[dict[str, Any]]:
    """
    Return a fresh list of default theme dicts with canonical negative IDs.

    This function ensures `theme_id` values match THEME_ID_MAP even if someone
    edited DEFAULT_THEMES above. Use this when extending model-generated themes.
    """
    return [
        {
            **theme,
            "theme_id": THEME_ID_MAP[theme["title"]]
        }
        for theme in DEFAULT_THEMES
    ]

