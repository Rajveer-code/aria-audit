"""Five axes of the ARIA runtime audit envelope.

Each module exposes a `score(...)` function returning a frozen result dataclass
from `aria_audit.core`. Modules are independently testable and importable.
"""
