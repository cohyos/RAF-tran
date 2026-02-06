"""
Safe I/O utilities for cross-platform compatibility.

This module provides utilities for handling Unicode output on systems
with restricted character encodings (e.g., Windows cp1255, cp1252).
"""

import sys


# ASCII replacements for common scientific symbols
UNICODE_TO_ASCII = {
    'μ': 'u',           # micro/mu
    'µ': 'u',           # micro sign (different codepoint)
    'τ': 'tau',         # tau (optical depth)
    'λ': 'lambda',      # lambda (wavelength)
    'θ': 'theta',       # theta (angle)
    'ρ': 'rho',         # rho (density)
    'σ': 'sigma',       # sigma (cross-section)
    'ω': 'omega',       # omega (single scattering albedo)
    'π': 'pi',          # pi
    'α': 'alpha',       # alpha (Angstrom exponent)
    'β': 'beta',        # beta
    'γ': 'gamma',       # gamma
    'δ': 'delta',       # delta
    'ε': 'epsilon',     # epsilon (emissivity)
    'ν': 'nu',          # nu (wavenumber)
    'Ω': 'Omega',       # Omega (solid angle)
    '°': ' deg',        # degree symbol
    '±': '+/-',         # plus-minus
    '×': 'x',           # multiplication
    '÷': '/',           # division
    '≈': '~',           # approximately
    '≤': '<=',          # less than or equal
    '≥': '>=',          # greater than or equal
    '≠': '!=',          # not equal
    '∞': 'inf',         # infinity
    '²': '^2',          # superscript 2
    '³': '^3',          # superscript 3
    '⁴': '^4',          # superscript 4
    '₀': '_0',          # subscript 0
    '₁': '_1',          # subscript 1
    '₂': '_2',          # subscript 2
    '₃': '_3',          # subscript 3
    '→': '->',          # arrow
    '←': '<-',          # left arrow
    '↑': '^',           # up arrow
    '↓': 'v',           # down arrow
    '✓': '[OK]',        # checkmark
    '✗': '[X]',         # X mark
}


def _can_encode(text: str, encoding: str = None) -> bool:
    """Check if text can be encoded with the given encoding."""
    if encoding is None:
        encoding = sys.stdout.encoding or 'utf-8'
    try:
        text.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def sanitize_for_console(text: str) -> str:
    """
    Replace Unicode characters with ASCII equivalents for console output.

    This function replaces Greek letters and other scientific symbols
    with ASCII representations to ensure compatibility with restricted
    character encodings like Windows cp1255 or cp1252.

    Parameters
    ----------
    text : str
        Input text potentially containing Unicode symbols

    Returns
    -------
    str
        Text with Unicode symbols replaced by ASCII equivalents
    """
    result = text
    for unicode_char, ascii_replacement in UNICODE_TO_ASCII.items():
        result = result.replace(unicode_char, ascii_replacement)
    return result


def safe_print(*args, **kwargs):
    """
    Print function that handles Unicode encoding errors gracefully.

    On systems with restricted encodings (e.g., Windows with cp1255),
    this function will automatically replace problematic Unicode characters
    with ASCII equivalents before printing.

    Parameters
    ----------
    *args : tuple
        Arguments to print (same as built-in print)
    **kwargs : dict
        Keyword arguments to print (same as built-in print)
    """
    # Convert all arguments to strings
    text_parts = [str(arg) for arg in args]
    text = kwargs.get('sep', ' ').join(text_parts)

    # Check if we can encode directly
    encoding = sys.stdout.encoding or 'utf-8'

    try:
        # Try to print directly first
        print(text, **{k: v for k, v in kwargs.items() if k != 'sep'})
    except UnicodeEncodeError:
        # Fall back to sanitized output
        sanitized = sanitize_for_console(text)
        try:
            print(sanitized, **{k: v for k, v in kwargs.items() if k != 'sep'})
        except UnicodeEncodeError:
            # Last resort: encode with replacement
            safe_text = sanitized.encode(encoding, errors='replace').decode(encoding)
            print(safe_text, **{k: v for k, v in kwargs.items() if k != 'sep'})


def configure_utf8_output():
    """
    Configure stdout to use UTF-8 encoding if possible.

    This is useful when running scripts that need to output Unicode
    characters on systems with restrictive default encodings.

    Call this at the start of your script to enable UTF-8 output.
    """
    import io

    if sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding='utf-8',
                errors='replace'
            )
        except (AttributeError, TypeError):
            # stdout doesn't have buffer attribute (e.g., in some IDEs)
            pass
