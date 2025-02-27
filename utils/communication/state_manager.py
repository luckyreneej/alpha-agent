import threading
import copy
import json
import os
import time
from typing import Dict, Any, Optional, List, Set, Callable
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages shared state and memory for agents in the multi-agent system.
    Provides persistence, versioning, and access control for agent states.
    """
    
    def __init__(self, state_dir: str = 'data/agent_states'):
        """
        Initialize the state manager.
        
        Args:
            state_dir: Directory for persistent storage of agent states
        """
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        
        # Global shared state
        self.global_state: Dict[str, Any] = {}
        
        # Per-agent states
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        
        # Memory stores for agents
        self.memory_stores: Dict[str, List[Dict[str, Any]]] = {}
        
        # Working memory for ongoing reasoning
        self.working_memory: Dict[str, Dict[str, Any]] = {}
        
        # Shared knowledge base
        self.knowledge_base: Dict[str, Any] = {}
        
        # Access control - which agents can access which state keys
        self.access_control: Dict[str, Set[str]] = {}  # key -> set of agent_ids
        
        # State versioning
        self.state_versions: Dict[str, int] = {}  # key -> version
        
        # Locks for thread safety
        self.global_lock = threading.RLock()
        self.agent_locks: Dict[str, threading.RLock] = {}
        
        # Event handlers for state changes
        self.state_change_handlers: Dict[str, List[Callable]] = {}  # key -> list of handlers
        
        # Load any persisted state
        self._load_persisted_state()
    
    def initialize_agent_state(self, agent_id: str, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize state for a new agent.
        
        Args:
            agent_id: Unique agent identifier
            initial_state: Initial state dictionary
        """
        with self.global_lock:
            # Create agent lock if doesn't exist
            if agent_id not in self.agent_locks:
                self.agent_locks[agent_id] = threading.RLock()
            
            with self.agent_locks[agent_id]:
                # Initialize agent state
                self.agent_states[agent_id] = initial_state or {}
                
                # Initialize memory store
                self.memory_stores[agent_id] = []
                
                # Initialize working memory
                self.working_memory[agent_id] = {}
                
                logger.info(f"Initialized state for agent {agent_id}")
    
    def cleanup_agent_state(self, agent_id: str) -> None:
        """
        Clean up state when an agent is removed.
        
        Args:
            agent_id: Unique agent identifier
        """
        with self.global_lock:
            if agent_id in self.agent_states:
                # First persist any important state
                self._persist_agent_state(agent_id)
                
                # Remove from dictionaries
                del self.agent_states[agent_id]
                
                if agent_id in self.memory_stores:
                    del self.memory_stores[agent_id]
                
                if agent_id in self.working_memory:
                    del self.working_memory[agent_id]
                
                if agent_id in self.agent_locks:
                    del self.agent_locks[agent_id]
                
                # Remove from access control
                for key, agents in self.access_control.items():
                    if agent_id in agents:
                        agents.remove(agent_id)
                
                logger.info(f"Cleaned up state for agent {agent_id}")
    
    def update_global_state(self, key: str, value: Any, agent_id: Optional[str] = None) -> None:
        """
        Update a value in the global state.
        
        Args:
            key: State key
            value: New state value
            agent_id: If provided, checks access control
        """
        with self.global_lock:
            # Check access control if agent_id is provided
            if agent_id is not None:
                if key in self.access_control and agent_id not in self.access_control[key]:
                    logger.warning(f"Agent {agent_id} attempted to update global state key {key} without permission")
                    return
            
            # Update the state
            old_value = self.global_state.get(key)
            self.global_state[key] = copy.deepcopy(value)  # Deep copy to prevent modification by reference
            
            # Update version
            if key not in self.state_versions:
                self.state_versions[key] = 1
            else:
                self.state_versions[key] += 1
            
            # Notify change handlers
            self._notify_state_change(key, old_value, value)
            
            # Persist important global states
            # Currently we only persist after all updates, not on each change
            # self._persist_global_state()
    
    def get_global_state(self, key: str, agent_id: Optional[str] = None, default: Any = None) -> Any:
        """
        Get a value from the global state.
        
        Args:
            key: State key
            agent_id: If provided, checks access control
            default: Default value if key doesn't exist
            
        Returns:
            State value or default
        """
        with self.global_lock:
            # Check access control if agent_id is provided
            if agent_id is not None:
                if key in self.access_control and agent_id not in self.access_control[key]:
                    logger.warning(f"Agent {agent_id} attempted to read global state key {key} without permission")
                    return default
            
            # Return a deep copy to prevent modification by reference
            value = self.global_state.get(key, default)
            return copy.deepcopy(value)
    
    def update_agent_state(self, agent_id: str, key: str, value: Any) -> None:
        """
        Update a value in an agent's state.
        
        Args:
            agent_id: Unique agent identifier
            key: State key
            value: New state value
        """
        with self.global_lock:
            if agent_id not in self.agent_locks:
                logger.warning(f"Agent {agent_id} not initialized")
                return
            
            with self.agent_locks[agent_id]:
                if agent_id not in self.agent_states:
                    self.agent_states[agent_id] = {}
                
                # Update the state
                old_value = self.agent_states[agent_id].get(key)
                self.agent_states[agent_id][key] = copy.deepcopy(value)  # Deep copy to prevent modification by reference
                
                # Update version
                state_key = f"{agent_id}.{key}"
                if state_key not in self.state_versions:
                    self.state_versions[state_key] = 1
                else:
                    self.state_versions[state_key] += 1
    
    def get_agent_state(self, agent_id: str, key: str, default: Any = None) -> Any:
        """
        Get a value from an agent's state.
        
        Args:
            agent_id: Unique agent identifier
            key: State key
            default: Default value if key doesn't exist
            
        Returns:
            State value or default
        """
        with self.global_lock:
            if agent_id not in self.agent_states:
                return default
            
            # Return a deep copy to prevent modification by reference
            value = self.agent_states[agent_id].get(key, default)
            return copy.deepcopy(value)
    
    def add_to_memory(self, agent_id: str, memory_item: Dict[str, Any]) -> None:
        """
        Add an item to an agent's memory store.
        
        Args:
            agent_id: Unique agent identifier
            memory_item: Memory item to add (should include timestamp)
        """
        with self.global_lock:
            if agent_id not in self.memory_stores:
                self.memory_stores[agent_id] = []
            
            # Add timestamp if not present
            if 'timestamp' not in memory_item:
                memory_item['timestamp'] = time.time()
                
            self.memory_stores[agent_id].append(copy.deepcopy(memory_item))
            
            # Limit memory size (keep only the most recent 1000 items by default)
            max_memory_size = 1000
            if len(self.memory_stores[agent_id]) > max_memory_size:
                self.memory_stores[agent_id] = self.memory_stores[agent_id][-max_memory_size:]
    
    def get_memory(self, agent_id: str, query: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get items from an agent's memory store, optionally filtered by a query.
        
        Args:
            agent_id: Unique agent identifier
            query: Filter query (dict with keys and values to match)
            limit: Maximum number of items to return
            
        Returns:
            List of memory items
        """
        with self.global_lock:
            if agent_id not in self.memory_stores:
                return []
            
            # Start with all memories
            memories = self.memory_stores[agent_id]
            
            # Filter by query if provided
            if query:
                filtered_memories = []
                for memory in memories:
                    # Check if all query keys match
                    if all(memory.get(key) == value for key, value in query.items()):
                        filtered_memories.append(memory)
                memories = filtered_memories
            
            # Return most recent up to limit
            memories = sorted(memories, key=lambda x: x.get('timestamp', 0), reverse=True)
            return copy.deepcopy(memories[:limit])
    
    def update_working_memory(self, agent_id: str, key: str, value: Any) -> None:
        """
        Update working memory for an agent's ongoing reasoning.
        
        Args:
            agent_id: Unique agent identifier
            key: Working memory key
            value: New value
        """
        with self.global_lock:
            if agent_id not in self.working_memory:
                self.working_memory[agent_id] = {}
            
            self.working_memory[agent_id][key] = copy.deepcopy(value)
    
    def get_working_memory(self, agent_id: str, key: Optional[str] = None) -> Any:
        """
        Get working memory for an agent.
        
        Args:
            agent_id: Unique agent identifier
            key: Specific working memory key, or None for all
            
        Returns:
            Working memory value or dict of all values
        """
        with self.global_lock:
            if agent_id not in self.working_memory:
                return {} if key is None else None
            
            if key is None:
                return copy.deepcopy(self.working_memory[agent_id])
            else:
                return copy.deepcopy(self.working_memory[agent_id].get(key))
    
    def clear_working_memory(self, agent_id: str) -> None:
        """
        Clear all working memory for an agent.
        
        Args:
            agent_id: Unique agent identifier
        """
        with self.global_lock:
            if agent_id in self.working_memory:
                self.working_memory[agent_id] = {}
    
    def update_knowledge_base(self, key: str, value: Any, agent_id: Optional[str] = None) -> None:
        """
        Update a value in the shared knowledge base.
        
        Args:
            key: Knowledge key
            value: New value
            agent_id: If provided, checks access control
        """
        with self.global_lock:
            # Check access control if agent_id is provided
            if agent_id is not None:
                kb_key = f"knowledge_base.{key}"
                if kb_key in self.access_control and agent_id not in self.access_control[kb_key]:
                    logger.warning(f"Agent {agent_id} attempted to update knowledge base key {key} without permission")
                    return
            
            self.knowledge_base[key] = copy.deepcopy(value)
    
    def get_knowledge(self, key: str, agent_id: Optional[str] = None, default: Any = None) -> Any:
        """
        Get a value from the shared knowledge base.
        
        Args:
            key: Knowledge key
            agent_id: If provided, checks access control
            default: Default value if key doesn't exist
            
        Returns:
            Knowledge value or default
        """
        with self.global_lock:
            # Check access control if agent_id is provided
            if agent_id is not None:
                kb_key = f"knowledge_base.{key}"
                if kb_key in self.access_control and agent_id not in self.access_control[kb_key]:
                    logger.warning(f"Agent {agent_id} attempted to read knowledge base key {key} without permission")
                    return default
            
            return copy.deepcopy(self.knowledge_base.get(key, default))
    
    def set_access_control(self, key: str, agent_ids: List[str]) -> None:
        """
        Set access control for a state key.
        
        Args:
            key: State key to control access to
            agent_ids: List of agent IDs allowed to access this key
        """
        with self.global_lock:
            self.access_control[key] = set(agent_ids)
    
    def register_state_change_handler(self, key: str, handler: Callable) -> None:
        """
        Register a handler to be called when a state key changes.
        
        Args:
            key: State key to watch
            handler: Function to call when state changes (handler(key, old_value, new_value))
        """
        with self.global_lock:
            if key not in self.state_change_handlers:
                self.state_change_handlers[key] = []
            
            self.state_change_handlers[key].append(handler)
    
    def unregister_state_change_handler(self, key: str, handler: Callable) -> None:
        """
        Unregister a state change handler.
        
        Args:
            key: State key
            handler: Handler to remove
        """
        with self.global_lock:
            if key in self.state_change_handlers and handler in self.state_change_handlers[key]:
                self.state_change_handlers[key].remove(handler)
    
    def _notify_state_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """
        Notify handlers about a state change.
        
        Args:
            key: State key that changed
            old_value: Previous value
            new_value: New value
        """
        if key in self.state_change_handlers:
            for handler in self.state_change_handlers[key]:
                try:
                    handler(key, old_value, new_value)
                except Exception as e:
                    logger.error(f"Error in state change handler for {key}: {e}")
    
    def persist_all_states(self) -> None:
        """
        Persist all states to disk.
        """
        with self.global_lock:
            self._persist_global_state()
            
            for agent_id in self.agent_states:
                self._persist_agent_state(agent_id)
            
            # Persist knowledge base
            self._persist_knowledge_base()
            
            logger.info("All states persisted to disk")
    
    def _persist_global_state(self) -> None:
        """
        Persist global state to disk.
        """
        state_file = os.path.join(self.state_dir, 'global_state.json')
        
        try:
            with open(state_file, 'w') as f:
                json.dump(self.global_state, f, indent=2)
        except Exception as e:
            logger.error(f"Error persisting global state: {e}")
    
    def _persist_agent_state(self, agent_id: str) -> None:
        """
        Persist an agent's state to disk.
        
        Args:
            agent_id: Unique agent identifier
        """
        if agent_id not in self.agent_states:
            return
        
        # Create agent directory if it doesn't exist
        agent_dir = os.path.join(self.state_dir, agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        
        # Persist agent state
        state_file = os.path.join(agent_dir, 'state.json')
        
        try:
            with open(state_file, 'w') as f:
                json.dump(self.agent_states[agent_id], f, indent=2)
        except Exception as e:
            logger.error(f"Error persisting state for agent {agent_id}: {e}")
        
        # Persist agent memory
        if agent_id in self.memory_stores:
            memory_file = os.path.join(agent_dir, 'memory.json')
            
            try:
                with open(memory_file, 'w') as f:
                    json.dump(self.memory_stores[agent_id], f, indent=2)
            except Exception as e:
                logger.error(f"Error persisting memory for agent {agent_id}: {e}")
    
    def _persist_knowledge_base(self) -> None:
        """
        Persist knowledge base to disk.
        """
        kb_file = os.path.join(self.state_dir, 'knowledge_base.json')
        
        try:
            with open(kb_file, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
        except Exception as e:
            logger.error(f"Error persisting knowledge base: {e}")
    
    def _load_persisted_state(self) -> None:
        """
        Load persisted states from disk.
        """
        # Load global state
        global_state_file = os.path.join(self.state_dir, 'global_state.json')
        if os.path.exists(global_state_file):
            try:
                with open(global_state_file, 'r') as f:
                    self.global_state = json.load(f)
                logger.info("Loaded global state from disk")
            except Exception as e:
                logger.error(f"Error loading global state: {e}")
        
        # Load knowledge base
        kb_file = os.path.join(self.state_dir, 'knowledge_base.json')
        if os.path.exists(kb_file):
            try:
                with open(kb_file, 'r') as f:
                    self.knowledge_base = json.load(f)
                logger.info("Loaded knowledge base from disk")
            except Exception as e:
                logger.error(f"Error loading knowledge base: {e}")
        
        # Load agent states
        for item in os.listdir(self.state_dir):
            agent_dir = os.path.join(self.state_dir, item)
            
            if os.path.isdir(agent_dir):
                agent_id = item
                
                # Load agent state
                state_file = os.path.join(agent_dir, 'state.json')
                if os.path.exists(state_file):
                    try:
                        with open(state_file, 'r') as f:
                            self.agent_states[agent_id] = json.load(f)
                            
                        # Create agent lock
                        self.agent_locks[agent_id] = threading.RLock()
                        
                        logger.info(f"Loaded state for agent {agent_id} from disk")
                    except Exception as e:
                        logger.error(f"Error loading state for agent {agent_id}: {e}")
                
                # Load agent memory
                memory_file = os.path.join(agent_dir, 'memory.json')
                if os.path.exists(memory_file):
                    try:
                        with open(memory_file, 'r') as f:
                            self.memory_stores[agent_id] = json.load(f)
                        logger.info(f"Loaded memory for agent {agent_id} from disk")
                    except Exception as e:
                        logger.error(f"Error loading memory for agent {agent_id}: {e}")