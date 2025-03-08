import logging
from typing import List, Any, Optional, Union, Dict
import time

from utils.communication.message import Message, MessageType, MessagePriority, create_message
from utils.communication.unified_communication import UnifiedCommunicationManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all agents in the trading system.
    Provides core communication capabilities and state management.
    """

    def __init__(self, agent_id_or_config, communicator):
        """
        Initialize the agent.

        Args:
            agent_id_or_config: Either a string agent_id or a configuration dictionary
            communicator: Communication manager for inter-agent messaging
        """
        # Handle both initialization methods
        if isinstance(agent_id_or_config, str):
            self.agent_id = agent_id_or_config
            self.config = {}
        else:
            self.config = agent_id_or_config
            # Get agent type from class name if not provided
            self.agent_type = self.__class__.__name__.lower()
            # Remove 'agent' from the end if present
            if self.agent_type.endswith('agent'):
                self.agent_type = self.agent_type[:-5]
            self.agent_id = f"{self.agent_type}_agent"
        
        self.communicator = communicator
        self.state = {}  # Agent's internal state
        self.running = False
        self._init_communicator_compat()

        # Note: We don't register here anymore, registration is handled in main.py

    def _init_communicator_compat(self):
        """
        Initialize compatibility methods for the communicator.
        This makes sure older agent code works with the new UnifiedCommunicationManager.
        """
        if not self.communicator:
            return

        # Add missing methods if they don't exist
        if not hasattr(self.communicator, 'subscribe_to_topic'):
            setattr(self.communicator, 'subscribe_to_topic', lambda topic: None)

        if not hasattr(self.communicator, 'publish_to_topic'):
            logger.info(f"Adding compatibility method publish_to_topic to communicator")
            def publish_to_topic(topic: str, message: Union[Message, Any]) -> bool:
                if not isinstance(message, Message):
                    message = create_message(
                        message_type=MessageType.DATA,
                        sender_id=self.agent_id,
                        content=message,
                        metadata={"topic": topic}
                    )
                return self.communicator.publish(message, topic)
            self.communicator.publish_to_topic = publish_to_topic

        if not hasattr(self.communicator, 'broadcast_message'):
            logger.info(f"Adding compatibility method broadcast_message to communicator")
            def broadcast_message(message: Union[Message, Any]) -> bool:
                if not isinstance(message, Message):
                    message = create_message(
                        message_type=MessageType.BROADCAST,
                        sender_id=self.agent_id,
                        content=message,
                        receiver_id=None
                    )
                return self.communicator.publish(message, "broadcast")
            self.communicator.broadcast_message = broadcast_message

    def start(self) -> None:
        """Start agent operation."""
        self.running = True
        logger.info(f"Agent {self.agent_id} started")

    def stop(self) -> None:
        """Stop agent operation."""
        self.running = False
        logger.info(f"Agent {self.agent_id} stopped")

    def send_message(self, receiver_id: str, message_type: Union[MessageType, str],
                    content: Any, metadata: Optional[dict] = None) -> str:
        """
        Send a message to another agent.

        Args:
            receiver_id: ID of the receiving agent
            message_type: Type of message
            content: Message content
            metadata: Optional metadata for the message

        Returns:
            Message ID
        """
        if not self.communicator:
            logger.error(f"Agent {self.agent_id} has no communicator")
            return ""

        # Convert string message type to enum if needed
        if isinstance(message_type, str):
            message_type = MessageType(message_type)

        # Create and send message
        message = create_message(
            message_type=message_type,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=content,
            metadata=metadata
        )

        success = self.communicator.send_message(message)
        if not success:
            logger.warning(f"Failed to send message from {self.agent_id} to {receiver_id}")

        return message.id

    def broadcast_message(self, content: Any, metadata: Optional[dict] = None) -> str:
        """
        Broadcast a message to all agents.

        Args:
            content: Message content
            metadata: Optional metadata for the message

        Returns:
            Message ID
        """
        if not self.communicator:
            logger.error(f"Agent {self.agent_id} has no communicator")
            return ""

        message = create_message(
            message_type=MessageType.BROADCAST,
            sender_id=self.agent_id,
            receiver_id=None,
            content=content,
            metadata=metadata
        )

        success = self.communicator.send_message(message)
        if not success:
            logger.warning(f"Failed to broadcast message from {self.agent_id}")

        return message.id

    def send_request(self, receiver_id: str, request_type: str, content: Any,
                    timeout: float = 30.0, metadata: Optional[dict] = None) -> Optional[Any]:
        """
        Send a request and wait for a response.

        Args:
            receiver_id: ID of the receiving agent
            request_type: Type of request
            content: Request content
            timeout: Timeout in seconds
            metadata: Optional metadata for the request

        Returns:
            Response content or None if timed out
        """
        if not self.communicator:
            logger.error(f"Agent {self.agent_id} has no communicator")
            return None

        request = create_message(
            message_type=MessageType.REQUEST,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content={"request_type": request_type, "content": content},
            metadata=metadata
        )

        response = self.communicator.send_request_and_wait(request, timeout)
        if response is None:
            logger.warning(f"Request from {self.agent_id} to {receiver_id} timed out")
            return None

        return response.content

    def subscribe_to_topic(self, topic: str) -> bool:
        """
        Subscribe to a topic to receive messages.

        Args:
            topic: Topic to subscribe to

        Returns:
            True if successful
        """
        if not self.communicator:
            return False
        return self.communicator.subscribe_to_topic(self.agent_id, topic)

    def publish_to_topic(self, topic: str, content: Any, metadata: Optional[dict] = None) -> str:
        """
        Publish a message to a topic.

        Args:
            topic: Topic to publish to
            content: Message content
            metadata: Optional metadata for the message

        Returns:
            Message ID
        """
        if not self.communicator:
            return ""

        # Ensure metadata exists
        if metadata is None:
            metadata = {}
        metadata["topic"] = topic

        # Create message with topic as receiver_id
        message = create_message(
            message_type=MessageType.DATA,
            sender_id=self.agent_id,
            receiver_id=topic,  # Use topic as receiver_id
            content=content,
            metadata=metadata
        )

        success = self.communicator.publish_to_topic(topic, message)
        if not success:
            logger.warning(f"Failed to publish message to topic {topic}")

        return message.id

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
            return []

        messages = []
        try:
            if wait:
                message = self.communicator.receive_message(self.agent_id, timeout)
                if message:
                    messages.append(message)
            else:
                while True:
                    message = self.communicator.receive_message(self.agent_id, 0)
                    if not message:
                        break
                    messages.append(message)
        except Exception as e:
            logger.error(f"Error getting messages: {e}")

        return messages

    def process_messages(self) -> None:
        """Process all pending messages."""
        messages = self.get_messages()
        for message in messages:
            try:
                self.process_message(message)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                error_msg = create_message(
                    message_type=MessageType.ERROR,
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    content=str(e),
                    metadata={"original_message_id": message.id}
                )
                self.communicator.send_message(error_msg)

    def process_message(self, message: Message) -> None:
        """
        Process a single message. Override this in derived classes.

        Args:
            message: Message to process
        """
        logger.debug(f"Base agent {self.agent_id} received message: {message}")

    def update_state(self, key: str, value: Any) -> None:
        """
        Update agent's internal state.

        Args:
            key: State key to update
            value: New value
        """
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get value from agent's internal state.

        Args:
            key: State key to get
            default: Default value if key not found

        Returns:
            State value or default
        """
        return self.state.get(key, default)

    def run(self) -> None:
        """Main agent loop. Override this in derived classes."""
        while self.running:
            try:
                self.process_messages()
                time.sleep(0.1)  # Prevent busy waiting
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                time.sleep(1)  # Wait before retrying

    def handle_message(self, message: Message) -> Optional[Any]:
        """
        Default message handler for all agents.
        
        Args:
            message: Message to handle
            
        Returns:
            Optional response
        """
        try:
            # Check if we have a specific handler for this message type
            if hasattr(self, '_message_handlers') and message.message_type in self._message_handlers:
                return self._message_handlers[message.message_type](message)
            
            # Check if we have a type-specific method
            handler_name = f"handle_{message.message_type.value.lower()}"
            if hasattr(self, handler_name):
                return getattr(self, handler_name)(message)
            
            # Default handling based on message type
            if message.message_type == MessageType.DATA:
                return self.process_data(message.content, message.metadata)
            elif message.message_type == MessageType.STATUS:
                return self.process_status(message.content)
            elif message.message_type == MessageType.SYSTEM:
                return self.process_system_message(message.content)
            else:
                logger.debug(f"No specific handler for message type {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error handling message in {self.agent_id}: {e}")
            return None
            
    def process_data(self, data: Any, metadata: Optional[Dict] = None) -> None:
        """
        Default data processing method.
        
        Args:
            data: Data to process
            metadata: Optional metadata about the data
        """
        logger.debug(f"Agent {self.agent_id} received data: {data}")
        
    def process_status(self, status: Dict) -> None:
        """
        Default status processing method.
        
        Args:
            status: Status information
        """
        logger.debug(f"Agent {self.agent_id} received status update: {status}")
        
    def process_system_message(self, content: Any) -> None:
        """
        Default system message processing method.
        
        Args:
            content: System message content
        """
        logger.debug(f"Agent {self.agent_id} received system message: {content}")
