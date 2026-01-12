# Branch Comparison Analysis: Redis Features

This document compares the two Redis feature branches to ensure consistency when merging.

**Last Updated:** 2026-01-12

---

## Branch Overview

| Branch | Purpose | Status | Base |
|--------|---------|--------|------|
| `feat/redis-search-tools` | Redis search tools (VectorSearch, TextSearch, HybridSearch, RangeSearch) with config classes | Ready for review | `main` |
| `feat/redis-memory-sessions` | Redis memory service (RedisMemoryService) + improved session service | In development | `main` |

### Recent Commits

**`feat/redis-search-tools`** (6 commits ahead of main):
- `de730ed` feat(redis): add version-aware hybrid search with dual config support
- `8bd8dc3` docs(samples): update redis_vl_search sample to use config-based API
- `b4e81d0` test(redis): update tests to use config-based API
- `cb26dbb` feat(redis): export config classes from tools module
- `8e42778` refactor(redis): use config objects in Redis search tool constructors

**`feat/redis-memory-sessions`** (1 commit ahead of main):
- `3418722` feat: improve redis memory and session services

---

## 1. Directory Structure Comparison

### Source Code (`src/google/adk_community/`)

| Directory | `feat/redis-search-tools` | `feat/redis-memory-sessions` | Notes |
|-----------|---------------------------|------------------------------|-------|
| `tools/__init__.py` | ✅ New (93 lines) | ❌ Missing | Lazy loading with helpful error messages |
| `tools/redis/__init__.py` | ✅ New (61 lines) | ❌ Missing | Exports all search tools + configs |
| `tools/redis/base_search_tool.py` | ✅ New (223 lines) | ❌ Missing | Abstract base class |
| `tools/redis/config.py` | ✅ New (~200 lines) | ❌ Missing | Pydantic config classes |
| `tools/redis/vector_search_tool.py` | ✅ New (213 lines) | ❌ Missing | KNN vector search |
| `tools/redis/text_search_tool.py` | ✅ New (171 lines) | ❌ Missing | BM25 full-text search |
| `tools/redis/hybrid_search_tool.py` | ✅ New (246 lines) | ❌ Missing | Combined vector + text (version-aware) |
| `tools/redis/range_search_tool.py` | ✅ New (185 lines) | ❌ Missing | Distance threshold search |
| `memory/redis_memory_service.py` | ❌ Missing | ✅ New (297 lines) | RedisVL-based memory |
| `memory/__init__.py` | Unchanged | Unchanged | **Needs lazy loading** |
| `sessions/redis_session_service.py` | Modified | Modified | **Potential conflict** |

### Tests (`tests/unittests/`)

| Directory | `feat/redis-search-tools` | `feat/redis-memory-sessions` | Notes |
|-----------|---------------------------|------------------------------|-------|
| `tools/__init__.py` | ✅ New | ❌ Missing | Empty init |
| `tools/redis/__init__.py` | ✅ New | ❌ Missing | Empty init |
| `tools/redis/test_vector_search_tool.py` | ✅ New (274 lines) | ❌ Missing | |
| `tools/redis/test_text_search_tool.py` | ✅ New (174 lines) | ❌ Missing | |
| `tools/redis/test_hybrid_search_tool.py` | ✅ New (298 lines) | ❌ Missing | Updated for version-aware |
| `tools/redis/test_range_search_tool.py` | ✅ New (157 lines) | ❌ Missing | |
| `memory/test_redis_memory_service.py` | ❌ Missing | ✅ New (294 lines) | |
| `sessions/test_redis_session_service.py` | Modified | Modified | **Potential conflict** |

### Samples (`contributing/samples/`)

| Directory | `feat/redis-search-tools` | `feat/redis-memory-sessions` | Notes |
|-----------|---------------------------|------------------------------|-------|
| `redis_vl_search/README.md` | ✅ Comprehensive | ❌ Missing | ~180 lines |
| `redis_vl_search/schema.yaml` | ✅ Present | ❌ Missing | Index schema |
| `redis_vl_search/load_data.py` | ✅ Present | ❌ Missing | Data loader |
| `redis_vl_search/redis_vl_search_agent/` | ✅ Present | ⚠️ Partial | Only agent dir exists |
| `redis_memory/` | ❌ Missing | ❌ Missing | **Should be added** |

---

## 2. Optional Dependencies Comparison

### `pyproject.toml` - `[project.optional-dependencies]`

| Branch | Dependency Group | Packages |
|--------|------------------|----------|
| `feat/redis-search-tools` | `redis-vl` | `redisvl>=0.13.2`, `nltk>=3.8.0`, `sentence-transformers>=2.2.0` |
| `feat/redis-memory-sessions` | `redis-vl` | `redisvl>=0.13.2`, `nltk>=3.8.0`, `sentence-transformers>=2.2.0` |

**Status: ALIGNED** - Both branches now use the same `redis-vl` optional dependency group.

### Core Dependencies

| Dependency | Both Branches |
|------------|---------------|
| `redis` | `>=5.0.0, <6.0.0` |

**Status: ALIGNED**

---

## 3. Import Patterns Comparison

### `tools/__init__.py` (feat/redis-search-tools)

```python
# Uses lazy loading with __getattr__
_REDIS_TOOLS = {"RedisVectorSearchTool", "RedisHybridSearchTool", ...}
_REDIS_CONFIGS = {"RedisVectorQueryConfig", "RedisHybridQueryConfig", ...}

def __getattr__(name: str):
    if name in _REDIS_TOOLS:
        try:
            from .redis import ...
        except ImportError as e:
            raise ImportError(
                f"{name} requires redisvl. "
                "Install with: pip install google-adk-community[redis-vl]"
            ) from e
```

### `memory/__init__.py` (feat/redis-memory-sessions)

```python
# Direct imports only - no lazy loading for Redis services
from .open_memory_service import OpenMemoryService
from .open_memory_service import OpenMemoryServiceConfig

__all__ = ["OpenMemoryService", "OpenMemoryServiceConfig"]
# RedisMemoryService NOT exported!
```

### Inconsistency

| Aspect | Tools | Memory | Recommendation |
|--------|-------|--------|----------------|
| **Lazy loading** | ✅ Yes | ❌ No | Memory should use lazy loading too |
| **Error messages** | ✅ Helpful | ⚠️ Generic ImportError | Add helpful install instructions |
| **Exports** | ✅ All tools exported | ❌ Redis services not exported | Add to `__all__` with lazy loading |

---

## 4. Sample Code Organization

### Pattern in `feat/redis-search-tools`

```
contributing/samples/redis_vl_search/
├── README.md                          # Comprehensive documentation
├── schema.yaml                        # Redis index schema
├── load_data.py                       # Data loading script
└── redis_vl_search_agent/
    ├── __init__.py
    └── agent.py                       # Agent definition
```

### Pattern in `feat/redis-memory-sessions`

```
contributing/samples/redis_vl_search/
└── redis_vl_search_agent/             # Only this exists (incomplete)
```

**Missing for memory branch:**
- No `redis_memory/` sample directory
- No README for memory services
- No example agent using RedisMemoryService

---

## 5. Documentation Comparison

| Document | `feat/redis-search-tools` | `feat/redis-memory-sessions` |
|----------|---------------------------|------------------------------|
| Sample README | ✅ `redis_vl_search/README.md` (180+ lines) | ❌ Missing |
| PR Notes | ❌ Not in branch | ✅ `contributing/PR_NOTES.md` |
| Proposal Doc | ❌ N/A | ✅ `redis_memory_service_proposal.md` |

---

## 6. Recommendations for Consistency

### 6.1 Unified Optional Dependencies

```toml
[project.optional-dependencies]
# Existing
test = ["pytest>=8.4.2", "pytest-asyncio>=1.2.0"]

# Redis features (unified naming)
redis-tools = [
    "redisvl>=0.13.0",
    "nltk>=3.8.0",
    "sentence-transformers>=2.2.0",
]
redis-memory = [
    "redisvl>=0.13.0",
    "agent-memory-client>=0.1.0",
]

# Combined for convenience
redis-all = [
    "redisvl>=0.13.0",
    "nltk>=3.8.0",
    "sentence-transformers>=2.2.0",
    "agent-memory-client>=0.1.0",
]
```

### 6.2 Lazy Loading for Memory Module

Update `memory/__init__.py` to use lazy loading like tools:

```python
__all__ = [
    "OpenMemoryService",
    "OpenMemoryServiceConfig",
    "RedisMemoryService",
    "RedisMemoryServiceConfig",
    "RedisAgentMemoryService",
    "RedisAgentMemoryServiceConfig",
    "MemoryExtractionStrategy",
]

_REDIS_MEMORY_SERVICES = {
    "RedisMemoryService",
    "RedisMemoryServiceConfig",
}

_AGENT_MEMORY_SERVICES = {
    "RedisAgentMemoryService",
    "RedisAgentMemoryServiceConfig",
    "MemoryExtractionStrategy",
}

def __getattr__(name: str):
    if name in _REDIS_MEMORY_SERVICES:
        try:
            from .redis_memory_service import RedisMemoryService, RedisMemoryServiceConfig
            globals().update({...})
            return globals()[name]
        except ImportError as e:
            raise ImportError(
                f"{name} requires redisvl. "
                "Install with: pip install google-adk-community[redis-memory]"
            ) from e
    # Similar for agent memory services
```

### 6.3 Sample Directory Structure

Create consistent sample structure for memory:

```
contributing/samples/
├── open_memory/                    # Existing
│   ├── README.md
│   ├── main.py
│   └── open_memory_agent/
├── redis_vl_search/                # From feat/redis-search-tools
│   ├── README.md
│   ├── schema.yaml
│   ├── load_data.py
│   └── redis_vl_search_agent/
└── redis_memory/                   # NEW - should be added
    ├── README.md
    ├── redis_memory_agent/
    │   ├── __init__.py
    │   └── agent.py
    └── redis_agent_memory_agent/   # Optional: separate example
        ├── __init__.py
        └── agent.py
```

### 6.4 Documentation Location

| Document Type | Location | Notes |
|---------------|----------|-------|
| Sample READMEs | `contributing/samples/<sample>/README.md` | Per-sample docs |
| PR Notes | `contributing/PR_NOTES.md` | Track PR status |
| Design Proposals | Root directory or `docs/` | Keep out of `contributing/` |

---

## 7. Merge Checklist

### Before Merging `feat/redis-search-tools` (PR #1)

- [x] Verify all 4 search tools have tests (30 passed, 1 skipped)
- [x] Verify `redis_vl_search` sample is complete
- [x] Verify `pyproject.toml` has `redis-vl` dependency group
- [x] Verify `tools/__init__.py` has lazy loading
- [x] Add version-aware hybrid search with dual config support
- [x] Use `packaging.version.parse()` for version comparison

### Before Merging `feat/redis-memory-sessions` (PR #2)

- [ ] Add lazy loading to `memory/__init__.py` with helpful error messages
- [ ] Export `RedisMemoryService` and `RedisMemoryServiceConfig` in `__all__`
- [ ] Create `redis_memory/` sample directory with README
- [ ] Add `redis-memory` optional dependency group to `pyproject.toml`
- [ ] Align redisvl version with PR #1 (`>=0.13.0`)
- [ ] Remove upper bound from `redis` dependency (already done)
- [ ] Move design docs out of root (or delete after PR)

### After Both PRs Merged

- [ ] Verify no conflicts in `pyproject.toml`
- [ ] Verify both `tools/` and `memory/` modules work together
- [ ] Run all tests: `pytest tests/unittests/`
- [ ] Update main README with new features
- [ ] Consider adding `redis-all` convenience dependency group

---

## 8. Conflict Analysis

### Files Modified in Both Branches

| File | `feat/redis-search-tools` | `feat/redis-memory-sessions` | Conflict Risk |
|------|---------------------------|------------------------------|---------------|
| `pyproject.toml` | Modified (adds `redis-vl`) | Modified (adds `redis-memory`) | **Medium** - different sections |
| `memory/__init__.py` | Unchanged from main | Modified (adds Redis exports) | **None** |
| `sessions/redis_session_service.py` | Modified | Modified | **High** - same file |
| `tests/.../test_redis_session_service.py` | Modified | Modified | **High** - same file |

### Recommended Merge Order

1. **Merge `feat/redis-search-tools` first** (PR #1)
   - More complete (has samples, docs)
   - No dependency on memory services

2. **Rebase `feat/redis-memory-sessions` onto main** (after PR #1 merged)
   - Resolve any conflicts in `redis_session_service.py`
   - Align `pyproject.toml` dependencies

3. **Merge `feat/redis-memory-sessions`** (PR #2)

---

## 9. Summary

| Aspect | Current State | Recommendation |
|--------|---------------|----------------|
| **Dependency naming** | `redis-vl` vs `redis-memory` | Keep both, add `redis-all` |
| **Import pattern** | Inconsistent (lazy vs direct) | Use lazy loading for both |
| **Sample structure** | Tools has sample, Memory missing | Add `redis_memory/` sample |
| **Documentation** | Tools has README, Memory has proposal | Add sample README for memory |
| **redisvl version** | `>=0.13.2` vs `>=0.13.0` | Align to `>=0.13.0` |
| **Merge order** | N/A | Tools first, then Memory |

---

## 10. Action Items

### Immediate (Before PR #2)

1. [ ] Create `contributing/samples/redis_memory/` with README and example agent
2. [ ] Update `memory/__init__.py` to use lazy loading with helpful error messages
3. [ ] Add `redis-memory` optional dependency group to `pyproject.toml`:
   ```toml
   redis-memory = [
       "redisvl>=0.13.0",
   ]
   ```
4. [ ] Export `RedisMemoryService` and `RedisMemoryServiceConfig` in `memory/__init__.py`

### Post-Merge

1. [ ] Add `redis-all` convenience dependency group combining tools and memory
2. [ ] Update main README with Redis features overview
3. [ ] Clean up design docs from root directory

---

## 11. Key Implementation Differences

### Version Detection Pattern

**`feat/redis-search-tools`** uses `packaging.version.parse()`:
```python
from packaging.version import parse

def _supports_native_hybrid() -> bool:
    return parse(_get_redisvl_version()) >= parse("0.13.0")
```

**`feat/redis-memory-sessions`** uses import-time check:
```python
try:
    from redisvl.index import SearchIndex
    REDISVL_AVAILABLE = True
except ImportError:
    REDISVL_AVAILABLE = False
```

**Recommendation:** Both patterns are valid. The memory branch pattern is simpler for availability checks, while the search tools pattern is needed for version-specific feature detection.

### Config Pattern

**`feat/redis-search-tools`** uses Pydantic config classes:
```python
class RedisVectorQueryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vector_field_name: str = "embedding"
    num_results: int = Field(default=10, ge=1)
```

**`feat/redis-memory-sessions`** also uses Pydantic:
```python
class RedisMemoryServiceConfig(BaseModel):
    search_top_k: int = Field(default=10, ge=1, le=100)
    user_content_salience: float = Field(default=0.8, ge=0.0, le=1.0)
```

**Recommendation:** Both follow the same pattern. Consider adding `model_config = ConfigDict(extra="forbid")` to memory config for consistency.

