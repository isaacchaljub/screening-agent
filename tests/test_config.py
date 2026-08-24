import pytest

from screening_agent import config


def test_app_env_is_valid():
    assert config.APP_ENV in config.VALID_ENVS


def test_zones_cover_both_countries():
    countries = {zone.country for zone in config.ZONES}
    assert countries == {"ES", "MX"}
    assert len(config.ZONES) == 10


def test_zone_ids_are_unique():
    ids = [zone.id for zone in config.ZONES]
    assert len(ids) == len(set(ids))


def test_tone_matches_process_design():
    assert config.TONE.max_words == 25
    assert config.TONE.spanish_register == "tú"
    assert config.TONE.one_question_per_message is True


def test_free_tier_guard_allows_dev(monkeypatch):
    monkeypatch.setattr(config, "APP_ENV", "dev")
    config.assert_model_allowed("google:gemini-flash-lite")
    config.assert_model_allowed("groq:llama-3.1-70b")


def test_free_tier_guard_blocks_non_dev(monkeypatch):
    monkeypatch.setattr(config, "APP_ENV", "prod")
    with pytest.raises(config.FreeTierModelError):
        config.assert_model_allowed("google:gemini-flash-lite")
    with pytest.raises(config.FreeTierModelError):
        config.assert_model_allowed("groq:llama-3.1-70b")


def test_paid_vendor_allowed_outside_dev(monkeypatch):
    monkeypatch.setattr(config, "APP_ENV", "prod")
    config.assert_model_allowed("openai:gpt-5.6-terra")
    config.assert_model_allowed("anthropic:claude-sonnet-5")
