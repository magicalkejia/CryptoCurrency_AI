"""
crypto.skills.registry
=========================
The Skills layer (v6 §9 / §11): every capability an Agent can invoke is a named,
versioned, AUDITED tool function.  A Skill records, on every call, its name,
category, input keys, output keys, timestamp and duration into an audit log
(v6 principle "所有 Agent 行为可审计").

This is the registry + decorator + a SkillContext that carries the audit log.
Concrete skills (wrapping triple_barrier / fusion / risk / oms / ...) are
registered in crypto.skills.catalog.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class SkillRecord:
    skill: str
    category: str
    input_keys: List[str]
    output_keys: List[str]
    ok: bool
    duration_ms: float
    ts: float = field(default_factory=time.time)
    error: str = ""


@dataclass
class SkillContext:
    """Carries the audit log across a single decision run."""
    audit_log: List[SkillRecord] = field(default_factory=list)

    def record(self, rec: SkillRecord):
        self.audit_log.append(rec)

    def summary(self) -> Dict[str, Any]:
        n = len(self.audit_log)
        ok = sum(r.ok for r in self.audit_log)
        return {"skill_calls": n, "skill_success": ok,
                "skill_success_rate": (ok / n) if n else 1.0,
                "skills_used": [r.skill for r in self.audit_log]}


class SkillRegistry:
    def __init__(self):
        self._skills: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, category: str):
        def deco(fn: Callable):
            self._skills[name] = {"fn": fn, "category": category}
            return fn
        return deco

    def list(self, category: str | None = None) -> List[str]:
        return [n for n, m in self._skills.items()
                if category is None or m["category"] == category]

    def call(self, name: str, ctx: SkillContext, *args, **kwargs):
        if name not in self._skills:
            raise KeyError(f"unknown skill: {name}")
        meta = self._skills[name]
        t0 = time.time()
        ok, err, out = True, "", None
        try:
            out = meta["fn"](*args, **kwargs)
        except Exception as e:
            ok, err = False, repr(e)
            raise
        finally:
            ctx.record(SkillRecord(
                skill=name, category=meta["category"],
                input_keys=sorted(list(kwargs.keys())),
                output_keys=sorted(list(out.keys())) if isinstance(out, dict) else ["<value>"],
                ok=ok, duration_ms=(time.time() - t0) * 1000.0, error=err))
        return out


# global registry instance
REGISTRY = SkillRegistry()
