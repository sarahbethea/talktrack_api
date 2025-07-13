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


THEME_ID_MAP = {
    "None": -1,
    "Introduction": -2,
    "Interviewer": -3
}


def get_default_themes() -> list[dict]:
    return [
        {
            **theme,
            "theme_id": THEME_ID_MAP[theme["title"]]
        }
        for theme in DEFAULT_THEMES
    ]

