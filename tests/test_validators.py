"""
Unit tests for input validation
"""
import pytest
from utils.validators import InputValidator

def test_valid_text():
    """Test validation with valid text"""
    validator = InputValidator()
    is_valid, error = validator.validate_text("This is valid text")
    assert is_valid is True
    assert error == ""

def test_none_text():
    """Test validation with None"""
    validator = InputValidator()
    is_valid, error = validator.validate_text(None)
    assert is_valid is False
    assert "None" in error

def test_empty_text():
    """Test validation with empty text"""
    validator = InputValidator()
    is_valid, error = validator.validate_text("")
    assert is_valid is False
    assert "empty" in error.lower()

def test_whitespace_only():
    """Test validation with whitespace only"""
    validator = InputValidator()
    is_valid, error = validator.validate_text("   ")
    assert is_valid is False
    assert "empty" in error.lower()

def test_too_long_text():
    """Test validation with text exceeding max length"""
    validator = InputValidator()
    long_text = "a" * 10000
    is_valid, error = validator.validate_text(long_text, max_length=100)
    assert is_valid is False
    assert "maximum length" in error.lower()

def test_non_string_input():
    """Test validation with non-string input"""
    validator = InputValidator()
    is_valid, error = validator.validate_text(12345)
    assert is_valid is False
    assert "string" in error.lower()
