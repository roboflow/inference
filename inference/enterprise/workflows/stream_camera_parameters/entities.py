from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ApplyCameraParametersResult:
    success: bool
    applied: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None
    message: Optional[str] = None

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], *, http_ok: bool = True
    ) -> "ApplyCameraParametersResult":
        return cls(
            success=bool(data.get("success", http_ok)),
            applied=list(data.get("applied") or []),
            failed=list(data.get("failed") or []),
            skipped=bool(data.get("skipped", False)),
            skip_reason=data.get("skip_reason"),
            message=data.get("message"),
        )
