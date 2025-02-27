import logging
from typing import Dict, List, Any, Optional, Union
import time
import uuid

from utils.communication.message import Message, MessageType
from utils.communication.communication_manager import CommunicationManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base class for all agents in the alpha-agent system.
    Provides communication capabilities and state management.
    """
    
    def __init__(self, agent_id: str, communicator: CommunicationManager):
        """
        Initialize the agent.
        
        Args:
            agent_id: Unique identifier for this agent
            communicator: Communication manager for inter-agent messaging
        """
        self.agent_id = agent_id
        self.communicator = communicator
        self.state: Dict[str, Any] = {}  # Agent's internal state
        self.context: Dict[str, Any] = {}  # Current working context
        self.memory: Dict[str, Any] = {}  # Agent's memory of past events
        self.running = False
        
        # Register with communication manager
        if self.communicator:
            self.communicator.register_agent(self.agent_id)
    
    def start(self):
        """
        Start agent operation.
        """
        self.running = True
        logger.info(f"Agent {self.agent_id} started")
    
    def stop(self):
        """
        Stop agent operation.
        """
        self.running = False
        logger.info(f"Agent {self.agent_id} stopped")
    
    def send_message(self, 
                   receiver_id: str, 
                   message_type: Union[MessageType, str], 
                   content: Any,
                   correlation_id: Optional[str] = None,
                   reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a message to another agent.
        
        Args:
            receiver_id: ID of the receiving agent
            message_type: Type of message (e.g., DATA, REQUEST, COMMAND)
            content: Message content
            correlation_id: Optional correlation ID for related messages
            reply_to: Optional ID of message this is replying to
            metadata: Optional additional metadata
            
        Returns:
            Message ID
        """
        if not self.communicator:
            logger.error(f"Agent {self.agent_id} has no communicator")
            return ""
        
        # Convert string message type to enum if needed
        if isinstance(message_type, str):
            message_type = MessageType(message_type)
        
        # Create message
        message = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id,
            reply_to=reply_to,
            metadata=metadata
        )
        
        # Send message
        success = self.communicator.send_message(message)
        
        if not success:
            logger.warning(f"Failed to send message from {self.agent_id} to {receiver_id}")
        
        return message.id
    
    def broadcast_message(self, 
                        message_type: Union[MessageType, str], 
                        content: Any,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Broadcast a message to all agents.
        
        Args:
            message_type: Type of message
            content: Message content
            metadata: Optional additional metadata
            
        Returns:
            Message ID
        """
        if not self.communicator:
            logger.error(f"Agent {self.agent_id} has no communicator")
            return ""
        
        # Convert string message type to enum if needed
        if isinstance(message_type, str):
            message_type = MessageType(message_type)
        
        # Create message
        message = Message(
            sender_id=self.agent_id,
            receiver_id=None,  # None indicates broadcast
            message_type=MessageType.BROADCAST,
            content=content,
            metadata=metadata
        )
        
        # Send broadcast message
        success = self.communicator.send_message(message)
        
        if not success:
            logger.warning(f"Failed to broadcast message from {self.agent_id}")
        
        return message.id
    
    def send_request(self, 
                   receiver_id: str, 
                   request_type: str, 
                   content: Any,
                   metadata: Optional[Dict[str, Any]] = None,
                   timeout: float = 30.0) -> Optional[Any]:
        """
        Send a request and wait for a response.
        
        Args:
            receiver_id: ID of the receiving agent
            request_type: Type of request
            content: Request content
            metadata: Optional additional metadata
            timeout: Timeout in seconds
            
        Returns:
            Response content or None if timed out
        """
        if not self.communicator:
            logger.error(f"Agent {self.agent_id} has no communicator")
            return None
        
        # Combine metadata
        if metadata is None:
            metadata = {}
        metadata['request_type'] = request_type
        
        # Use the send_request_and_wait method of communication manager
        response = self.communicator.send_request_and_wait(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            request_type=request_type,
            content=content,
            timeout=timeout,
            metadata=metadata
        )
        
        if response is None:
            logger.warning(f"Request from {self.agent_id} to {receiver_id} timed out")
            return None
        
        return response.content
    
    def register_request_handler(self, request_type: str, handler: callable) -> bool:
        """
        Register a handler for a specific request type.
        
        Args:
            request_type: Type of request
            handler: Function to handle the request
            
        Returns:
            True if registration was successful
        """
        if not self.communicator:
            logger.error(f"Agent {self.agent_id} has no communicator")
            return False
        
        return self.communicator.register_request_handler(
            agent_id=self.agent_id,
            request_type=request_type,
            handler=handler
        )
    
    def subscribe_to_topic(self, topic: str) -> bool:
        """
        Subscribe to a topic to receive published messages.
        
        Args:
            topic: Topic to subscribe to
            
        Returns:
            True if subscription was successful
        """
        if not self.communicator:
            logger.error(f"Agent {self.agent_id} has no communicator")
            return False
        
        return self.communicator.subscribe_to_topic(self.agent_id, topic)
    
    def publish_to_topic(self, topic: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Publish a message to a topic.
        
        Args:
            topic: Topic to publish to
            content: Message content
            metadata: Optional additional metadata
            
        Returns:
            Correlation ID for the published messages
        """
        if not self.communicator:
            logger.error(f"Agent {self.agent_id} has no communicator")
            return ""
        
        return self.communicator.publish_to_topic(
            sender_id=self.agent_id,
            topic=topic,
            content=content,
            metadata=metadata
        )
    
    def get_messages(self, wait: bool = False, timeout: Optional[float] = None) -> List[Message]:
        """
        Get messages sent to this agent.
        
        Args:
            wait: Whether to wait for a message if none available
            timeout: Timeout when waiting
            
        Returns:
            List of messages
        """
        if not self.communicator:
            logger.error(f"Agent {self.agent_id} has no communicator")
            return []
        
        return self.communicator.get_messages(self.agent_id, wait, timeout)
    
    def process_messages(self) -> None:
        """
        Process all pending messages for this agent.
        Override this method in derived classes to implement specific processing logic.
        """
        messages = self.get_messages()
        
        for message in messages:
            self.process_message(message)
    
    def process_message(self, message: Message) -> None:
        """
        Process a single message.
        Override this method in derived classes to implement specific processing logic.
        
        Args:
            message: Message to process
        """
        logger.debug(f"Agent {self.agent_id} received message: {message}")
    
    def update_state(self, key: str, value: Any) -> None:
        """
        Update agent's internal state.
        
        Args:
            key: State key
            value: State value
        """
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get a value from agent's internal state.
        
        Args:
            key: State key
            default: Default value if key doesn't exist
            
        Returns:
            State value or default
        """
        return self.state.get(key, default)
    
    def update_memory(self, key: str, value: Any) -> None:
        """
        Update agent's memory.
        
        Args:
            key: Memory key
            value: Memory value
        """
        self.memory[key] = {
            'value': value,
            'timestamp': time.time()
        }
    
    def recall(self, key: str, default: Any = None) -> Any:
        """
        Recall a value from agent's memory.
        
        Args:
            key: Memory key
            default: Default value if key doesn't exist
            
        Returns:
            Remembered value or default
        """
        memory_entry = self.memory.get(key)
        if memory_entry is None:
            return default
        return memory_entry['value']
    
    def update_context(self, key: str, value: Any) -> None:
        """
        Update agent's current working context.
        
        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get a value from agent's current working context.
        
        Args:
            key: Context key
            default: Default value if key doesn't exist
            
        Returns:
            Context value or default
        """
        return self.context.get(key, default)
    
    def clear_context(self) -> None:
        """
        Clear agent's current working context.
        """
        self.context = {}
    
    def join_conversation(self, conversation_id: str, initiator_id: str) -> bool:
        """
        Join an existing conversation.
        
        Args:
            conversation_id: ID of the conversation
            initiator_id: ID of the agent that initiated the conversation
            
        Returns:
            True if joined successfully
        """
        # This is a placeholder for conversation functionality
        # In a real implementation, this would coordinate with the communication manager
        logger.info(f"Agent {self.agent_id} joined conversation {conversation_id}")
        self.update_context('conversation_id', conversation_id)
        self.update_context('conversation_initiator', initiator_id)
        return True