# lang_utils.py
import langid

# Restrict to our target set to reduce false positives
TARGET = {"en","sw","ar","fr",}
langid.set_languages(list(TARGET))

def detect_lang(text: str) -> str:
    if not text or text.strip() == "":
        return "unknown"
    code, _ = langid.classify(text)
    return code if code in TARGET else code  # keep raw if outside TARGET
