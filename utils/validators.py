"""
Input validation utilities
"""
from typing import Optional, Tuple

class InputValidator:
    """Validates input data"""
    
    @staticmethod
    def validate_text(text: Optional[str], max_length: int = 5000) -> Tuple[bool, str]:
        """
        Validate input text
        
        Args:
            text: Input text to validate
            max_length: Maximum allowed text length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if text is None:
            return False, "Text cannot be None"
        
        if not isinstance(text, str):
            return False, "Text must be a string"
        
        if len(text.strip()) == 0:
            return False, "Text cannot be empty"
        
        if len(text) > max_length:
            return False, f"Text exceeds maximum length of {max_length} characters"
        
        return True, ""
