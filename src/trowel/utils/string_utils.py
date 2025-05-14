"""String utility functions for text processing and cleaning."""

import re
import string


def clean_unicode_chars(text):
    """Clean Unicode special characters from a string.

    This function removes various invisible Unicode characters that might appear
    at the beginning or throughout a string, including:
    - Byte Order Mark (BOM) characters
    - Zero-width spaces
    - Directional text markers
    - Other invisible formatting characters

    Args:
        text: String to clean

    Returns:
        Cleaned string with Unicode special characters removed
    """
    if not text:
        return text

    # Convert to string if not already
    text = str(text)

    # List of problematic Unicode characters to remove
    chars_to_remove = [
        '\ufeff',  # BOM (UTF-8)
        '\ufffe',  # BOM (UTF-16 LE)
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\u200e',  # Left-to-right mark
        '\u200f',  # Right-to-left mark
        '\u202a',  # Left-to-right embedding
        '\u202b',  # Right-to-left embedding
        '\u202c',  # Pop directional formatting
        '\u202d',  # Left-to-right override
        '\u202e',  # Right-to-left override
        '\u2060',  # Word joiner
        '\u2061',  # Function application
        '\u2062',  # Invisible times
        '\u2063',  # Invisible separator
        '\u2064',  # Invisible plus
        '\u2066',  # Left-to-right isolate
        '\u2067',  # Right-to-left isolate
        '\u2068',  # First strong isolate
        '\u2069',  # Pop directional isolate
        '\u206a',  # Inhibit symmetric swapping
        '\u206b',  # Activate symmetric swapping
        '\u206c',  # Inhibit Arabic form shaping
        '\u206d',  # Activate Arabic form shaping
        '\u206e',  # National digit shapes
        '\u206f',  # Nominal digit shapes
        '\ufe00',  # Variation selector-1
        '\ufe01',  # Variation selector-2
        '\ufe02',  # Variation selector-3
        '\ufe03',  # Variation selector-4
        '\ufe04',  # Variation selector-5
        '\ufe05',  # Variation selector-6
        '\ufe06',  # Variation selector-7
        '\ufe07',  # Variation selector-8
        '\ufe08',  # Variation selector-9
        '\ufe09',  # Variation selector-10
        '\ufe0a',  # Variation selector-11
        '\ufe0b',  # Variation selector-12
        '\ufe0c',  # Variation selector-13
        '\ufe0d',  # Variation selector-14
        '\ufe0e',  # Variation selector-15
        '\ufe0f',  # Variation selector-16
    ]

    # Remove all specified characters
    for char in chars_to_remove:
        text = text.replace(char, '')

    return text


def clean_punctuation(text, preserve_brackets=True):
    """Clean punctuation from the beginning and end of a string.

    Args:
        text: String to clean
        preserve_brackets: If True, keeps parentheses and square brackets at the end

    Returns:
        Cleaned string with punctuation removed from beginning and end
    """
    if not text:
        return text

    # Define punctuation to preserve
    preserve_chars = '()[]' if preserve_brackets else ''

    # Create a set of punctuation characters to remove
    punct_to_remove = ''.join(
        c for c in string.punctuation if c not in preserve_chars)

    # Remove leading punctuation from the text
    while text and (text[0] in punct_to_remove):
        text = text[1:]

    # Remove trailing punctuation from the text,
    # but preserve trailing parentheses and brackets if requested
    if preserve_brackets:
        # Only clean trailing punctuation that isn't a parenthesis or bracket
        # First, check if the string ends with a parenthesis/bracket section
        if not (text and (text[-1] in ')]')):
            while text and (text[-1] in punct_to_remove):
                text = text[:-1]
    else:
        # Clean all trailing punctuation
        while text and (text[-1] in string.punctuation):
            text = text[:-1]

    return text


def extract_units(name):
    """Extract the variable name and unit from a column name string.

    This function searches for units specified in parentheses at the end of a name.
    Examples:
        "density (g/cm3)" -> ("density", "g/cm3")
        "d18o (permil)" -> ("d18o", "permil")
        "temperature (¡c)" -> ("temperature", "¡c")
        "regular name" -> ("regular name", "")

    Args:
        name: String containing a possible variable name with units

    Returns:
        Tuple containing (variable_name, units)
    """
    if not name:
        return "", ""

    # Match pattern like "name (unit)" where unit can contain special characters
    pattern = r'^(.*?)\s*\(([^)]+)\)\s*$'
    match = re.match(pattern, name)

    if match:
        var_name = match.group(1).strip()
        unit = match.group(2).strip()
        return var_name, unit
    else:
        return name, ""
