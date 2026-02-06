"""Strip personally identifiable information from dive log XML files."""

import re
import xml.etree.ElementTree as ET
from typing import Set

# XML tags that may contain PII
PII_TAGS: Set[str] = {'location', 'buddy', 'notes', 'gps', 'divemaster', 'suit'}

# GPS coordinate pattern
GPS_PATTERN = re.compile(r"gps='[^']*'|gps=\"[^\"]*\"")


def sanitize_xml(content: str) -> str:
    """
    Remove PII-containing elements from XML string.

    Args:
        content: Raw XML content

    Returns:
        Sanitized XML string with PII removed
    """
    try:
        root = ET.fromstring(content)
        _strip_elements(root)
        return ET.tostring(root, encoding='unicode')
    except ET.ParseError:
        # Fall back to regex stripping if XML is malformed
        return _regex_sanitize(content)


def _strip_elements(element: ET.Element) -> None:
    """
    Recursively remove PII tags from an XML tree.

    Args:
        element: Root element to process
    """
    children_to_remove = []

    for child in element:
        if child.tag in PII_TAGS:
            children_to_remove.append(child)
        else:
            _strip_elements(child)

    for child in children_to_remove:
        element.remove(child)

    # Strip gps attributes
    if 'gps' in element.attrib:
        del element.attrib['gps']


def _regex_sanitize(content: str) -> str:
    """
    Fallback regex-based PII removal for malformed XML.

    Args:
        content: Raw XML content

    Returns:
        Sanitized content with PII tags removed
    """
    for tag in PII_TAGS:
        content = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', content, flags=re.DOTALL)
        content = re.sub(f'<{tag}[^/]*/>', '', content)
    content = GPS_PATTERN.sub('', content)
    return content
