import pytest

from main import validate_model_lang_combo


def test_distil_with_english_is_allowed():
    validate_model_lang_combo("large-distil", "english")


def test_distil_with_auto_is_allowed():
    validate_model_lang_combo("large-distil", "auto")


def test_distil_with_japanese_is_rejected():
    with pytest.raises(ValueError, match="English-only"):
        validate_model_lang_combo("large-distil", "japanese")


def test_multilingual_model_with_japanese_is_allowed():
    validate_model_lang_combo("large", "japanese")
