# Standard library
import threading
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, Set, Tuple, List, Union, TYPE_CHECKING
from contextlib import contextmanager
import traceback

# Local
from simulation_types import MemoryType, MemoryEvent

# Type hints for circular imports
if TYPE_CHECKING:
    from stability_classes import Agent, Location

class SharedLocationTracker:
    """Thread-safe location tracking for agents."""
    
    def __init__(self, simulation=None):
        self._lock = threading.Lock()
        self.valid_locations = set()
        self.agent_positions = {}  # agent_name -> (location_name, grid_coordinate, timestamp)
        self.location_agents = {}  # location_name -> set of agent_names
        self.coordinate_agents = {}  # grid_coordinate -> set of agent_names
        self.simulation = simulation
        
    def set_simulation(self, simulation):
        """Set the simulation instance for location lookups."""
        with self._lock:
            self.simulation = simulation
            
    def set_valid_locations(self, locations: Set[str]):
        """Set the valid locations for tracking."""
        with self._lock:
            self.valid_locations = set(locations)
            
    def update_agent_position(self, agent_name: str, location_name: str, grid_coordinate: Optional[Tuple[int, int]], timestamp: float):
        """Update agent's position in the shared tracker."""
        with self._lock:
            # This method will now be protected externally by the LocationLockManager.
            # Validate location
            if location_name and location_name not in self.valid_locations:
                print(f"Warning: Attempting to update agent {agent_name} to invalid location {location_name}")
                return
                
            # Remove from old position
            old_position = self.agent_positions.get(agent_name)
            if old_position:
                old_location, old_coord, _ = old_position
                if old_location in self.location_agents:
                    self.location_agents[old_location].discard(agent_name)
                if old_coord in self.coordinate_agents:
                    self.coordinate_agents[old_coord].discard(agent_name)
            
            # Update to new position
            self.agent_positions[agent_name] = (location_name, grid_coordinate, timestamp)
            
            # Update location tracking
            if location_name:
                if location_name not in self.location_agents:
                    self.location_agents[location_name] = set()
                self.location_agents[location_name].add(agent_name)
            
            # Update coordinate tracking
            if grid_coordinate:
                if grid_coordinate not in self.coordinate_agents:
                    self.coordinate_agents[grid_coordinate] = set()
                self.coordinate_agents[grid_coordinate].add(agent_name)
                
    def get_agents_at_location(self, location_name: str) -> Set[str]:
        """Get set of agents at a specific location."""
        with self._lock:
            if location_name not in self.valid_locations:
                print(f"Warning: Attempting to get agents at invalid location {location_name}")
                print(f"Valid locations: {self.valid_locations}")
                return set()
            return self.location_agents.get(location_name, set()).copy()
        
    def get_agents_at_coordinate(self, coordinate: Tuple[int, int]) -> Set[str]:
        """Get set of agents at a specific coordinate."""
        with self._lock:
            return self.coordinate_agents.get(coordinate, set()).copy()
        
    def get_agent_position(self, agent_name: str) -> Optional[Tuple[str, Optional[Tuple[int, int]], float]]:
        """Get agent's current position."""
        with self._lock:
            return self.agent_positions.get(agent_name)
        
    def clear(self):
        """Clear all tracking data."""
        with self._lock:
            self.agent_positions.clear()
            self.location_agents.clear()
            self.coordinate_agents.clear()

    def get_location(self, location_name: str) -> Optional['Location']:
        """Get a Location object by name."""
        with self._lock:
            if not self.simulation or not hasattr(self.simulation, 'locations'):
                print(f"Warning: Simulation instance not available for location lookup: {location_name}")
                return None
            return self.simulation.locations.get(location_name)

class SharedResourceManager:
    """Manages shared resources in the simulation."""
    def __init__(self):
        self._lock = threading.Lock()
        self.resources = {}  # resource_name -> resource_data
        
    def acquire_resource(self, resource_name: str, agent_name: str, timeout: float = 1.0) -> bool:
        """Try to acquire a resource for an agent."""
        with self._lock:
            if resource_name not in self.resources:
                self.resources[resource_name] = {
                    'owner': None,
                    'queue': [],
                    'last_used': None
                }
            
            resource = self.resources[resource_name]
            
            # If resource is free, acquire it
            if resource['owner'] is None:
                resource['owner'] = agent_name
                resource['last_used'] = datetime.now()
                return True
                
            # If agent already owns the resource
            if resource['owner'] == agent_name:
                return True
                
            # Add to queue if not already there
            if agent_name not in resource['queue']:
                resource['queue'].append(agent_name)
                
            return False
            
    def release_resource(self, resource_name: str, agent_name: str) -> Optional[str]:
        """Release a resource and return next agent in queue."""
        with self._lock:
            if resource_name not in self.resources:
                return None
                
            resource = self.resources[resource_name]
            
            # Check if agent owns the resource
            if resource['owner'] != agent_name:
                return None
                
            # Release resource
            resource['owner'] = None
            
            # Get next agent from queue
            if resource['queue']:
                next_agent = resource['queue'].pop(0)
                resource['owner'] = next_agent
                resource['last_used'] = datetime.now()
                return next_agent
                
            return None
            
    def get_resource_status(self, resource_name: str) -> Dict[str, Any]:
        """Get current status of a resource."""
        with self._lock:
            if resource_name not in self.resources:
                return {
                    'owner': None,
                    'queue': [],
                    'last_used': None
                }
                
            resource = self.resources[resource_name]
            return {
                'owner': resource['owner'],
                'queue': resource['queue'].copy(),
                'last_used': resource['last_used']
            }
            
    def clear_resource(self, resource_name: str):
        """Clear a resource and its queue."""
        with self._lock:
            if resource_name in self.resources:
                del self.resources[resource_name]
                
    def clear_all_resources(self):
        """Clear all resources."""
        with self._lock:
            self.resources.clear()

class LocationLockManager:
    """Manages locks for locations to prevent race conditions during travel and activities."""
    def __init__(self, debug: bool = False):
        self.location_locks: Dict[str, threading.Lock] = {}
        self.agent_locks: Dict[str, threading.Lock] = {}
        self.pair_locks: Dict[Tuple[str, str], threading.Lock] = {}
        self._lock = threading.Lock()
        self.debug = debug
        self.logger = logging.getLogger(__name__)

    def get_location_lock(self, location_name: str) -> threading.Lock:
        with self._lock:
            if location_name not in self.location_locks:
                self.location_locks[location_name] = threading.Lock()
            return self.location_locks[location_name]

    def get_agent_lock(self, agent_name: str) -> threading.Lock:
        with self._lock:
            if agent_name not in self.agent_locks:
                self.agent_locks[agent_name] = threading.Lock()
            return self.agent_locks[agent_name]

    def get_pair_lock(self, from_location: str, to_location: str) -> threading.Lock:
        with self._lock:
            pair = (from_location, to_location)
            if pair not in self.pair_locks:
                self.pair_locks[pair] = threading.Lock()
            return self.pair_locks[pair]

    def release_all_locks(self) -> None:
        """Release all locks held by the manager."""
        try:
            with self._lock:
                # Release all location locks
                for lock in self.location_locks.values():
                    if lock.locked():
                        lock.release()
                
                # Release all agent locks
                for lock in self.agent_locks.values():
                    if lock.locked():
                        lock.release()
                
                # Release all pair locks
                for lock in self.pair_locks.values():
                    if lock.locked():
                        lock.release()
                
                # Clear the lock dictionaries
                self.location_locks.clear()
                self.agent_locks.clear()
                self.pair_locks.clear()
                
                self.logger.info("All locks released successfully")
                
        except Exception as e:
            self.logger.error(f"Error releasing locks: {str(e)}")
            traceback.print_exc()

    @contextmanager
    def location_lock(self, location_name: str, timeout: Optional[float] = None):
        """Context manager for acquiring and releasing a location lock."""
        lock = self.get_location_lock(location_name)
        acquired = False
        try:
            if self.debug:
                self.logger.info(f"Attempting to acquire lock for location: {location_name}")

            if timeout is not None:
                if not lock.acquire(timeout=timeout):
                    raise TimeoutError(f"Could not acquire lock for location {location_name} within {timeout}s")
            else:
                lock.acquire()

            acquired = True
            if self.debug:
                self.logger.info(f"Lock acquired for location: {location_name}")
            
            yield
        finally:
            if acquired:
                lock.release()
                if self.debug:
                    self.logger.info(f"Lock released for location: {location_name}")

    @contextmanager
    def agent_lock(self, agent_name: str, timeout: Optional[float] = None):
        """Context manager for acquiring and releasing an agent lock."""
        lock = self.get_agent_lock(agent_name)
        acquired = False
        try:
            if self.debug:
                self.logger.info(f"Attempting to acquire lock for agent: {agent_name}")
            
            if timeout is not None:
                if not lock.acquire(timeout=timeout):
                    raise TimeoutError(f"Could not acquire lock for agent {agent_name} within {timeout}s")
            else:
                lock.acquire()

            acquired = True
            if self.debug:
                self.logger.info(f"Lock acquired for agent: {agent_name}")
            
            yield
        finally:
            if acquired:
                lock.release()
                if self.debug:
                    self.logger.info(f"Lock released for agent: {agent_name}")

    @contextmanager
    def pair_lock(self, from_location: str, to_location: str, timeout: Optional[float] = None):
        """Context manager for acquiring and releasing a pair lock."""
        lock = self.get_pair_lock(from_location, to_location)
        acquired = False
        try:
            if self.debug:
                self.logger.info(f"Attempting to acquire pair lock for: {from_location} -> {to_location}")
            
            if timeout is not None:
                if not lock.acquire(timeout=timeout):
                    raise TimeoutError(f"Could not acquire pair lock for {from_location} -> {to_location} within {timeout}s")
            else:
                lock.acquire()
            
            acquired = True
            if self.debug:
                self.logger.info(f"Pair lock acquired for: {from_location} -> {to_location}")
            
            yield
        finally:
            if acquired:
                lock.release()
                if self.debug:
                    self.logger.info(f"Pair lock released for: {from_location} -> {to_location}") 