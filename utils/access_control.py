from config import DEFAULT_ROLE, ROLE_LEVELS

def normalize_role(role: str | None) -> str:
    role = (role or "").strip().lower()
    return role if role in ROLE_LEVELS else DEFAULT_ROLE

def allowed_levels(role: str) -> list[str]:
    clearance = ROLE_LEVELS[normalize_role(role)]
    return [lvl for lvl, rank in ROLE_LEVELS.items() if rank <= clearance]

def valid_level(level: str) -> str:
    level = (level or "").strip().lower()
    return level if level in ROLE_LEVELS else DEFAULT_ROLE
