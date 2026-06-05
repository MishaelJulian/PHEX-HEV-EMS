"""
test_ems_rules.py — Phase 3: EMS Rules Registry Unit Tests
"""

import pytest
from src.ems_rules import (
    EMS_RULES, get_rule_by_id, get_rule_by_name,
    validate_rule_registry, _REQUIRED_RULE_FIELDS,
)


class TestEMSRules:
    """Tests for the EMS rules registry."""

    def test_registry_has_8_rules(self):
        """Registry must contain exactly 8 rules."""
        assert len(EMS_RULES) == 8

    def test_all_rules_have_required_fields(self):
        """Every rule must have id, name, description, condition, action."""
        for rule in EMS_RULES:
            for field in _REQUIRED_RULE_FIELDS:
                assert field in rule, \
                    f"Rule {rule.get('id', 'UNKNOWN')} missing field: {field}"

    def test_rule_ids_unique(self):
        """All rule IDs must be unique."""
        ids = [r["id"] for r in EMS_RULES]
        assert len(ids) == len(set(ids))

    def test_rule_names_unique(self):
        """All rule names must be unique."""
        names = [r["name"] for r in EMS_RULES]
        assert len(names) == len(set(names))

    def test_lookup_by_id_found(self):
        """get_rule_by_id must return the correct rule."""
        rule = get_rule_by_id("RULE_01")
        assert rule is not None
        assert rule["name"] == "URBAN_EV_PRIORITY"

    def test_lookup_by_id_not_found(self):
        """get_rule_by_id must return None for unknown ID."""
        rule = get_rule_by_id("RULE_99")
        assert rule is None

    def test_lookup_by_name_found(self):
        """get_rule_by_name must return the correct rule."""
        rule = get_rule_by_name("REGEN_ON_BRAKING")
        assert rule is not None
        assert rule["id"] == "RULE_05"

    def test_validate_registry_passes(self):
        """validate_rule_registry must return True for the valid registry."""
        assert validate_rule_registry() is True

    def test_specific_rule_content(self):
        """Spot-check RULE_06 MAX_ACCEL_COMBINED."""
        rule = get_rule_by_id("RULE_06")
        assert rule is not None
        assert "MAX_ACCEL" in rule["name"]
        assert "ICE" in rule["action"]
        assert "Motor" in rule["action"]

    def test_all_rules_have_nonempty_descriptions(self):
        """All rule descriptions must be non-empty strings."""
        for rule in EMS_RULES:
            assert len(rule["description"]) > 0
            assert len(rule["condition"]) > 0
            assert len(rule["action"]) > 0
