import logging
from typing import List, Any, Optional, Union
import time

from utils.communication.message import Message, MessageType
from utils.communication.unified_communication import UnifiedCommunicationManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all agents in the trading system.
    Provides core communication capabilities and state management.
    """

    def __init__(self, agent_id: str, communicator: UnifiedCommunicationManager):
        """
        Initialize the agent.

        Args:
            agent_id: Unique identifier for this agent
            communicator: Communication manager for inter-agent messaging
        """
        self.agent_id = agent_id
        self.communicator = communicator
        self.state = {}  # Agent's internal state
        self.running = False

        # Register with communication manager
        if self.communicator:
            self.communicator.register_agent(self.agent_id)

    def start(self):
        """Start agent operation."""
        self.running = True
        logger.info(f"Agent {self.agent_id} started")

    def stop(self):
        """Stop agent operation."""
        self.running = False
        logger.info(f"Agent {self.agent_id} stopped")

    # Core messaging methods

    def send_message(self, receiver_id: str, message_type: Union[MessageType, str],
                     content: Any, reply_to: Optional[str] = None) -> str:
        """
        Send a message to another agent.

        Args:
            receiver_id: ID of the receiving agent
            message_type: Type of message
            content: Message content
            reply_to: Optional ID of message this is replying to

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
            reply_to=reply_to
        )

        # Send message
        success = self.communicator.send_message(message)

        if not success:
            logger.warning(f"Failed to send message from {self.agent_id} to {receiver_id}")

        return message.id

    def broadcast_message(self, message_type: Union[MessageType, str], content: Any) -> str:
        """
        Broadcast a message to all agents.

        Args:
            message_type: Type of message
            content: Message content

        Returns:
            Message ID
        """
        if not self.communicator:
            logger.error(f"Agent {self.agent_id} has no communicator")
            return ""

        # Convert string message type to enum if needed
        if isinstance(message_type, str):
            message_type = MessageType(message_type)

        # Create message for broadcast
        message = Message(
            sender_id=self.agent_id,
            receiver_id=None,  # None indicates broadcast
            message_type=MessageType.BROADCAST,
            content=content
        )

        # Send broadcast message
        success = self.communicator.send_message(message)

        if not success:
            logger.warning(f"Failed to broadcast message from {self.agent_id}")

        return message.id

    def send_request(self, receiver_id: str, request_type: str, content: Any,
                     timeout: float = 30.0) -> Optional[Any]:
        """
        Send a request and wait for a response.

        Args:
            receiver_id: ID of the receiving agent
            request_type: Type of request
            content: Request content
            timeout: Timeout in seconds

        Returns:
            Response content or None if timed out
        """
        if not self.communicator:
            logger.error(f"Agent {self.agent_id} has no communicator")
            return None

        # Use the send_request_and_wait method
        response = self.communicator.send_request_and_wait(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            request_type=request_type,
            content=content,
            timeout=timeout
        )

        if response is None:
            logger.warning(f"Request from {self.agent_id} to {receiver_id} timed out")
            return None

        return response.content

    # Topic-based messaging

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

    def publish_to_topic(self, topic: str, content: Any) -> str:
        """
        Publish a message to a topic.

        Args:
            topic: Topic to publish to
            content: Message content

        Returns:
            Correlation ID
        """
        if not self.communicator:
            return ""

        return self.communicator.publish_to_topic(
            sender_id=self.agent_id,
            topic=topic,
            content=content
        )

    # Message processing

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

        return self.communicator.get_messages(self.agent_id, wait, timeout)

    def process_messages(self) -> None:
        """Process all pending messages."""
        messages = self.get_messages()

        for message in messages:
            self.process_message(message)

    def process_message(self, message: Message) -> None:
        """
        Process a single message. Override in derived classes.

        Args:
            message: Message to process
        """
        logger.debug(f"Agent {self.agent_id} received message: {message}")

    # State management

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

    # Main execution loop

    def run(self) -> None:
        """
        Main execution loop. Override in derived classes.
        """
        self.start()

        try:
            while self.running:
                # Process messages
                self.process_messages()

                # Sleep to avoid CPU spinning
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info(f"Agent {self.agent_id} received keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in agent {self.agent_id} execution: {e}", exc_info=True)
        finally:
            self.stop()
