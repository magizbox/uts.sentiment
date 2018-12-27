import string

import emoji


def remove_punctuation(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(i for i in text.split())
    return text


def remove_emoji(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = " ".join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text


def normalize_text(raw_text):
    text = " ".join(i for i in raw_text.split())
    filtered_text = remove_punctuation(text)
    clean_text = remove_emoji(filtered_text)
    return clean_text.lower()
