import threading
from typing import Any, Callable, Dict

class OperationManager:
    """Manages atomic operations to prevent duplicates and ensure thread safety."""
    
    def __init__(self):
        self.operation_locks = {}  # Locks for each operation type
        self.operation_states = {}  # Track operation states
        self._lock = threading.Lock()  # Lock for managing operation states
        
    def get_operation_lock(self, operation_type: str) -> threading.Lock:
        """Get or create a lock for an operation type."""
        with self._lock:
            if operation_type not in self.operation_locks:
                self.operation_locks[operation_type] = threading.Lock()
            return self.operation_locks[operation_type]
            
    def is_operation_in_progress(self, operation_type: str) -> bool:
        """Check if an operation is in progress."""
        with self._lock:
            return self.operation_states.get(operation_type, False)
            
    def set_operation_state(self, operation_type: str, in_progress: bool):
        """Set the state of an operation."""
        with self._lock:
            self.operation_states[operation_type] = in_progress
            
    def execute_atomic_operation(self, operation_type: str, operation_func: Callable[[], Any]) -> Any:
        """Execute an operation atomically."""
        if self.is_operation_in_progress(operation_type):
            print(f"Operation {operation_type} already in progress")
            return None
            
        lock = self.get_operation_lock(operation_type)
        try:
            with lock:
                self.set_operation_state(operation_type, True)
                result = operation_func()
                return result
        finally:
            self.set_operation_state(operation_type, False) 