import logging
import threading
import queue
import time
from typing import Dict, Any, List, Callable, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentCommunicator:
    """
    Manages communication between agents in the multi-agent system.
    Implements a central data store and message passing mechanism.
    """
    
    def __init__(self):
        # Central data store for sharing information between agents
        self.data_store = {}
        self.message_queues = {}
        self.agents = {}
        self.lock = threading.Lock()
        self.event_handlers = {}
    
    def register_agent(self, agent_id: str, agent) -> None:
        """
        Register an agent with the communicator
        
        Args:
            agent_id: Unique identifier for the agent
            agent: The agent object
        """
        with self.lock:
            self.agents[agent_id] = agent
            self.message_queues[agent_id] = queue.Queue()
            logger.info(f"Registered agent: {agent_id}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the communicator
        
        Args:
            agent_id: Unique identifier for the agent
        """
        with self.lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
            if agent_id in self.message_queues:
                del self.message_queues[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    def update_data(self, key: str, value: Any) -> None:
        """
        Update or add data to the central data store
        
        Args:
            key: Data identifier
            value: Data value
        """
        with self.lock:
            self.data_store[key] = value
            # Trigger event handlers for this key
            if key in self.event_handlers:
                for handler in self.event_handlers[key]:
                    handler(key, value)
    
    def get_data(self, key: str) -> Optional[Any]:
        """
        Get data from the central data store
        
        Args:
            key: Data identifier
            
        Returns:
            Data value or None if not found
        """
        return self.data_store.get(key)
    
    def send_message(self, from_agent: str, to_agent: str, message: Dict[str, Any]) -> bool:
        """
        Send a message from one agent to another
        
        Args:
            from_agent: Sender agent ID
            to_agent: Recipient agent ID
            message: Message payload
            
        Returns:
            True if message was sent, False otherwise
        """
        if to_agent not in self.message_queues:
            logger.error(f"Agent {to_agent} not found")
            return False
        
        # Add sender information to message
        message['from_agent'] = from_agent
        message['timestamp'] = time.time()
        
        # Add message to recipient's queue
        self.message_queues[to_agent].put(message)
        logger.debug(f"Message sent from {from_agent} to {to_agent}: {message}")
        return True
    
    def broadcast_message(self, from_agent: str, message: Dict[str, Any]) -> None:
        """
        Send a message from one agent to all other agents
        
        Args:
            from_agent: Sender agent ID
            message: Message payload
        """
        for agent_id in self.message_queues:
            if agent_id != from_agent:  # Don't send to self
                self.send_message(from_agent, agent_id, message.copy())
    
    def get_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get all pending messages for an agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of messages
        """
        if agent_id not in self.message_queues:
            logger.error(f"Agent {agent_id} not found")
            return []
        
        messages = []
        queue = self.message_queues[agent_id]
        
        # Get all messages without blocking
        while not queue.empty():
            try:
                messages.append(queue.get_nowait())
                queue.task_done()
            except queue.Empty:
                break
                
        return messages
    
    def register_event_handler(self, key: str, handler: Callable) -> None:
        """
        Register an event handler function to be called when data is updated
        
        Args:
            key: Data key to watch
            handler: Function to call when the key is updated (handler(key, value))
        """
        with self.lock:
            if key not in self.event_handlers:
                self.event_handlers[key] = []
            self.event_handlers[key].append(handler)
    
    def unregister_event_handler(self, key: str, handler: Callable) -> None:
        """
        Unregister an event handler
        
        Args:
            key: Data key
            handler: Handler function to unregister
        """
        with self.lock:
            if key in self.event_handlers and handler in self.event_handlers[key]:
                self.event_handlers[key].remove(handler)
    
    def get_all_data(self) -> Dict[str, Any]:
        """
        Get all data in the data store
        
        Returns:
            Copy of the data store
        """
        return self.data_store.copy()