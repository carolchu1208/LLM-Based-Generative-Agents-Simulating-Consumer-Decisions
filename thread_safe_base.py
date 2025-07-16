import threading
from typing import Any, Dict, Optional, Protocol, TypeVar, Generic

# Error Classes
class SimulationError(Exception):
    """Base class for simulation-related errors."""
    pass

class AgentError(SimulationError):
    """Errors related to agent operations."""
    pass

class LocationError(SimulationError):
    """Errors related to location operations."""
    pass

class MemoryError(SimulationError):
    """Errors related to memory operations."""
    pass

class MetricsError(SimulationError):
    """Errors related to metrics operations."""
    pass

T = TypeVar('T')

class Result(Generic[T]):
    """A class to handle operation results with success/failure states."""
    def __init__(self, success: bool, value: Optional[T] = None, error: Optional[str] = None, fallback: Optional[T] = None):
        self.success = success
        self.value = value
        self.error = error
        self.fallback = fallback

    @classmethod
    def success(cls, value: T) -> 'Result[T]':
        return cls(True, value=value)

    @classmethod
    def failure(cls, error: str, fallback: Optional[T] = None) -> 'Result[T]':
        return cls(False, error=error, fallback=fallback)

    def get_value(self) -> T:
        if self.success:
            return self.value
        if self.fallback is not None:
            return self.fallback
        raise ValueError(f"Operation failed: {self.error}")

class ThreadSafeBase:
    """Base class for thread-safe operations."""
    def __init__(self):
        self._lock = threading.Lock()
        
    def _safe_get(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Thread-safe get operation."""
        with self._lock:
            return data.get(key, default)
            
    def _safe_set(self, data: Dict[str, Any], key: str, value: Any) -> None:
        """Thread-safe set operation."""
        with self._lock:
            data[key] = value
            
    def _safe_update(self, data: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Thread-safe update operation."""
        with self._lock:
            data.update(updates)
            
    def _safe_delete(self, data: Dict[str, Any], key: str) -> None:
        """Thread-safe delete operation."""
        with self._lock:
            data.pop(key, None)
            
    def _safe_clear(self, data: Dict[str, Any]) -> None:
        """Thread-safe clear operation."""
        with self._lock:
            data.clear()
            
    def _safe_get_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Thread-safe get all operation."""
        with self._lock:
            return data.copy()
            
    def _safe_has_key(self, data: Dict[str, Any], key: str) -> bool:
        """Thread-safe has key operation."""
        with self._lock:
            return key in data 