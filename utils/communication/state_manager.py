from typing import Dict, List, Any, Optional, Union
import uuid
import datetime
import json
from enum import Enum


class MessageType(Enum):
    """Types of messages for agent communication."""
    SYSTEM = "system"  # System messages
    DATA = "data"  # Data transfer
    REQUEST = "request"  # Request for information/action
    RESPONSE = "response"  # Response to a request
    COMMAND = "command"  # Command to execute
    ERROR = "error"  # Error messages
    BROADCAST = "broadcast"  # Broadcast to all agents
    STATUS = "status"  # Status updates
    RESULT = "result"  # Result of a command execution
    REFLECTION = "reflection"  # Agent's reflection/reasoning
    EVENT = "event"  # Event notification
    QUERY = "query"  # Query for information
    PROPOSE = "propose"  # Proposal of ideas or actions
    FEEDBACK = "feedback"  # Feedback on actions/information
    REASONING = "reasoning"  # Step-by-step reasoning process
    SUMMARY = "summary"  # Summary of information


class MessagePriority(Enum):
    """Priority levels for messages."""
    LOW = 0  # Low priority, can be delayed
    NORMAL = 1  # Normal priority
    HIGH = 2  # High priority
    CRITICAL = 3  # Critical priority, process immediately


class Message:
    """A structured message for agent communication."""

    def __init__(self,
                 sender_id: str,
                 receiver_id: Optional[str],
                 message_type: MessageType,
                 content: Any,
                 correlation_id: Optional[str] = None,
                 reply_to: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 priority: MessagePriority = MessagePriority.NORMAL):
        """
        Initialize a message.

        Args:
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent, or None for broadcast
            message_type: Type of message (system, data, request, etc.)
            content: Message payload
            correlation_id: ID to correlate related messages
            reply_to: ID of the message this is replying to
            metadata: Additional message metadata
            priority: Message priority level
        """
        self.id = str(uuid.uuid4())
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.content = content
        self.correlation_id = correlation_id or self.id
        self.reply_to = reply_to
        self.metadata = metadata or {}
        self.timestamp = datetime.datetime.now().isoformat()
        self.priority = priority
        self.processed = False
        self.retries = 0
        self.max_retries = 3
        self.expiration = None  # Optional expiration time

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "priority": self.priority.value,
            "processed": self.processed,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "expiration": self.expiration
        }

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary representation."""
        # Convert string message type to enum
        if isinstance(data["message_type"], str):
            message_type = MessageType(data["message_type"])
        else:
            message_type = data["message_type"]

        # Convert priority string to enum if needed
        if "priority" in data:
            if isinstance(data["priority"], str):
                priority = MessagePriority(data["priority"])
            elif isinstance(data["priority"], int):
                priority = MessagePriority(data["priority"])
            else:
                priority = data["priority"]
        else:
            priority = MessagePriority.NORMAL

        msg = cls(
            sender_id=data["sender_id"],
            receiver_id=data.get("receiver_id"),
            message_type=message_type,
            content=data["content"],
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            metadata=data.get("metadata", {}),
            priority=priority
        )

        # Set additional fields
        if "id" in data:
            msg.id = data["id"]
        if "timestamp" in data:
            msg.timestamp = data["timestamp"]
        if "processed" in data:
            msg.processed = data["processed"]
        if "retries" in data:
            msg.retries = data["retries"]
        if "max_retries" in data:
            msg.max_retries = data["max_retries"]
        if "expiration" in data:
            msg.expiration = data["expiration"]

        return msg

    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create message from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def create_reply(self,
                     content: Any,
                     message_type: Optional[MessageType] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> 'Message':
        """Create a reply to this message."""
        reply_type = message_type or MessageType.RESPONSE
        if self.message_type == MessageType.REQUEST:
            reply_type = MessageType.RESPONSE
        elif self.message_type == MessageType.COMMAND:
            reply_type = MessageType.RESULT

        # Merge metadata with original metadata if provided
        merged_metadata = self.metadata.copy()
        if metadata:
            merged_metadata.update(metadata)

        return Message(
            sender_id=self.receiver_id,  # Swap sender and receiver
            receiver_id=self.sender_id,
            message_type=reply_type,
            content=content,
            correlation_id=self.correlation_id,  # Keep same correlation ID
            reply_to=self.id,  # Reference this message
            metadata=merged_metadata
        )

    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expiration is None:
            return False

        # Convert timestamp strings to datetime objects for comparison
        if isinstance(self.expiration, str):
            expiration_dt = datetime.datetime.fromisoformat(self.expiration)
            now = datetime.datetime.now()
            return now > expiration_dt
        else:
            # Assume it's a timestamp
            return datetime.datetime.now().timestamp() > self.expiration

    def increment_retry(self) -> bool:
        """Increment retry count and check if max retries reached."""
        self.retries += 1
        return self.retries <= self.max_retries

    def set_expiration(self, seconds: float) -> None:
        """Set expiration time in seconds from now."""
        expiration_time = datetime.datetime.now() + datetime.timedelta(seconds=seconds)
        self.expiration = expiration_time.isoformat()

    def mark_processed(self) -> None:
        """Mark message as processed."""
        self.processed = True

    def __repr__(self) -> str:
        return (f"Message(id={self.id}, type={self.message_type.value}, "
                f"sender={self.sender_id}, receiver={self.receiver_id})")


# Helper functions for creating specific types of messages

def create_data_message(sender_id: str, receiver_id: str, data: Any,
                        metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a data message."""
    return Message(
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.DATA,
        content=data,
        metadata=metadata
    )


def create_request_message(sender_id: str, receiver_id: str, request: Any, request_type: str,
                           metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a request message."""
    if metadata is None:
        metadata = {}
    metadata['request_type'] = request_type

    return Message(
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.REQUEST,
        content=request,
        metadata=metadata
    )


def create_response_message(sender_id: str, receiver_id: str,
                            response: Any, reply_to: str,
                            correlation_id: str,
                            metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a response message."""
    return Message(
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.RESPONSE,
        content=response,
        correlation_id=correlation_id,
        reply_to=reply_to,
        metadata=metadata
    )


def create_broadcast_message(sender_id: str, content: Any, message_type: MessageType = MessageType.BROADCAST,
                             metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a broadcast message."""
    return Message(
        sender_id=sender_id,
        receiver_id=None,  # None indicates broadcast
        message_type=message_type,
        content=content,
        metadata=metadata
    )


def create_error_message(sender_id: str, receiver_id: str,
                         error: str, reply_to: Optional[str] = None,
                         correlation_id: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create an error message."""
    return Message(
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.ERROR,
        content=error,
        correlation_id=correlation_id,
        reply_to=reply_to,
        metadata=metadata,
        priority=MessagePriority.HIGH  # Errors get high priority
    )


def create_command_message(sender_id: str, receiver_id: str,
                           command: Dict[str, Any],
                           metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a command message."""
    return Message(
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.COMMAND,
        content=command,
        metadata=metadata
    )


def create_status_message(sender_id: str, receiver_id: str,
                          status: Dict[str, Any],
                          metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a status update message."""
    return Message(
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.STATUS,
        content=status,
        metadata=metadata
    )


def create_reflection_message(sender_id: str, receiver_id: str,
                              reflection: Any,
                              metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a reflection message with agent's reasoning."""
    return Message(
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=MessageType.REFLECTION,
        content=reflection,
        metadata=metadata
    )


def create_system_message(content: Any, receiver_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          priority: MessagePriority = MessagePriority.HIGH) -> Message:
    """Create a system message."""
    return Message(
        sender_id="system",
        receiver_id=receiver_id,  # None for broadcast to all
        message_type=MessageType.SYSTEM,
        content=content,
        metadata=metadata,
        priority=priority  # System messages typically get high priority
    )
