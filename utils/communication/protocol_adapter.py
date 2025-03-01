import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum

from utils.communication.message import Message, MessageType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Types of communication protocols."""
    INTERNAL = "internal"  # Internal message bus protocol
    REST_API = "rest_api"  # REST API protocol
    WEBSOCKET = "websocket"  # WebSocket protocol
    GRPC = "grpc"  # gRPC protocol
    KAFKA = "kafka"  # Kafka messaging protocol


class ProtocolAdapter:
    """
    Base adapter for translating between different communication protocols.
    Adapters convert between internal message format and external protocols.
    """

    def __init__(self, protocol_type: ProtocolType):
        """
        Initialize the protocol adapter.

        Args:
            protocol_type: Type of protocol this adapter handles
        """
        self.protocol_type = protocol_type

    def to_internal(self, external_message: Any) -> Optional[Message]:
        """
        Convert an external message to the internal Message format.

        Args:
            external_message: Message in external protocol format

        Returns:
            Converted internal Message or None if conversion failed
        """
        raise NotImplementedError("Subclasses must implement to_internal")

    def from_internal(self, message: Message) -> Any:
        """
        Convert an internal Message to the external protocol format.

        Args:
            message: Internal Message object

        Returns:
            Converted message in external protocol format
        """
        raise NotImplementedError("Subclasses must implement from_internal")


class JSONProtocolAdapter(ProtocolAdapter):
    """
    Adapter for JSON-based protocols (REST API, WebSocket, etc.).
    Converts between internal Message objects and JSON serializable dictionaries.
    """

    def __init__(self, protocol_type: ProtocolType = ProtocolType.REST_API):
        """
        Initialize the JSON protocol adapter.

        Args:
            protocol_type: Type of protocol this adapter handles
        """
        super().__init__(protocol_type)

    def to_internal(self, external_message: Dict[str, Any]) -> Optional[Message]:
        """
        Convert a JSON dictionary to an internal Message.

        Args:
            external_message: JSON dictionary representing a message

        Returns:
            Converted internal Message or None if conversion failed
        """
        try:
            if not isinstance(external_message, dict):
                if isinstance(external_message, str):
                    try:
                        external_message = json.loads(external_message)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON: {e}")
                        return None
                else:
                    logger.error(f"Expected dict or JSON string, got {type(external_message)}")
                    return None

            # Check for required fields
            required_fields = ["sender_id", "content", "message_type"]
            for field in required_fields:
                if field not in external_message:
                    logger.error(f"Missing required field: {field}")
                    return None

            # Convert message_type string to enum
            try:
                message_type = MessageType(external_message["message_type"])
            except ValueError:
                logger.error(f"Invalid message_type: {external_message['message_type']}")
                return None

            # Create Message object
            message = Message(
                sender_id=external_message["sender_id"],
                receiver_id=external_message.get("receiver_id"),
                message_type=message_type,
                content=external_message["content"],
                correlation_id=external_message.get("correlation_id"),
                reply_to=external_message.get("reply_to"),
                metadata=external_message.get("metadata")
            )

            # Override id and timestamp if provided
            if "id" in external_message:
                message.id = external_message["id"]
            if "timestamp" in external_message:
                message.timestamp = external_message["timestamp"]

            return message

        except Exception as e:
            logger.error(f"Error converting to internal format: {e}")
            return None

    def from_internal(self, message: Message) -> Dict[str, Any]:
        """
        Convert an internal Message to a JSON dictionary.

        Args:
            message: Internal Message object

        Returns:
            Dictionary representation of the message
        """
        return message.to_dict()


class KafkaProtocolAdapter(ProtocolAdapter):
    """
    Adapter for Kafka messaging protocol.
    Handles conversion between internal messages and Kafka-specific formats.
    """

    def __init__(self):
        """
        Initialize the Kafka protocol adapter.
        """
        super().__init__(ProtocolType.KAFKA)
        self.json_adapter = JSONProtocolAdapter()

    def to_internal(self, external_message: Dict[str, Any]) -> Optional[Message]:
        """
        Convert a Kafka message to an internal Message.

        Args:
            external_message: Kafka message

        Returns:
            Converted internal Message or None if conversion failed
        """
        try:
            # Extract message value from Kafka message
            if "value" in external_message:
                message_value = external_message["value"]

                # Use JSON adapter to convert the message value
                if isinstance(message_value, str):
                    try:
                        message_value = json.loads(message_value)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Kafka message value: {e}")
                        return None

                message = self.json_adapter.to_internal(message_value)

                # Add Kafka-specific metadata if available
                if message and "metadata" in external_message:
                    kafka_metadata = external_message["metadata"]
                    if message.metadata is None:
                        message.metadata = {}
                    message.metadata.update({
                        "kafka": kafka_metadata
                    })

                return message
            else:
                logger.error("Kafka message missing 'value' field")
                return None

        except Exception as e:
            logger.error(f"Error converting Kafka message: {e}")
            return None

    def from_internal(self, message: Message) -> Dict[str, Any]:
        """
        Convert an internal Message to a Kafka message.

        Args:
            message: Internal Message object

        Returns:
            Kafka message format
        """
        # Convert message to JSON
        message_json = self.json_adapter.from_internal(message)

        # Extract Kafka-specific metadata if available
        kafka_metadata = {}
        if message.metadata and "kafka" in message.metadata:
            kafka_metadata = message.metadata["kafka"]

        # Determine topic based on message type or metadata
        topic = f"alpha_agent_{message.message_type.value}"
        if kafka_metadata.get("topic"):
            topic = kafka_metadata.get("topic")

        # Create Kafka message
        kafka_message = {
            "topic": topic,
            "value": json.dumps(message_json),
            "key": message.id,  # Use message ID as key
            "headers": [
                {"sender": message.sender_id},
                {"message_type": message.message_type.value}
            ]
        }

        # Add partition information if available
        if "partition" in kafka_metadata:
            kafka_message["partition"] = kafka_metadata["partition"]

        return kafka_message


class GRPCProtocolAdapter(ProtocolAdapter):
    """
    Adapter for gRPC protocol.
    Note: This is a simplified implementation that doesn't handle actual gRPC.
    """

    def __init__(self):
        """
        Initialize the gRPC protocol adapter.
        """
        super().__init__(ProtocolType.GRPC)
        self.json_adapter = JSONProtocolAdapter()

    def to_internal(self, external_message: Any) -> Optional[Message]:
        """
        Convert a gRPC message to an internal Message.

        Args:
            external_message: gRPC message

        Returns:
            Converted internal Message or None if conversion failed
        """
        # In a real implementation, this would handle proper gRPC message types
        # Here we're using a simplified approach
        try:
            if hasattr(external_message, "to_dict"):
                message_dict = external_message.to_dict()
                return self.json_adapter.to_internal(message_dict)
            elif isinstance(external_message, dict):
                return self.json_adapter.to_internal(external_message)
            else:
                logger.error(f"Unsupported gRPC message type: {type(external_message)}")
                return None
        except Exception as e:
            logger.error(f"Error converting gRPC message: {e}")
            return None

    def from_internal(self, message: Message) -> Dict[str, Any]:
        """
        Convert an internal Message to a gRPC message.

        Args:
            message: Internal Message object

        Returns:
            gRPC message format (simplified as dict)
        """
        # In a real implementation, this would create proper gRPC message types
        # Here we're returning a dict that could be used to construct a gRPC message
        return self.json_adapter.from_internal(message)


class ProtocolAdapterFactory:
    """
    Factory for creating protocol adapters based on protocol type.
    """

    @staticmethod
    def create_adapter(protocol_type: Union[ProtocolType, str]) -> ProtocolAdapter:
        """
        Create an appropriate protocol adapter for the given protocol type.

        Args:
            protocol_type: Type of protocol

        Returns:
            Protocol adapter instance
        """
        if isinstance(protocol_type, str):
            try:
                protocol_type = ProtocolType(protocol_type)
            except ValueError:
                logger.error(f"Invalid protocol type: {protocol_type}")
                protocol_type = ProtocolType.INTERNAL

        if protocol_type == ProtocolType.KAFKA:
            return KafkaProtocolAdapter()
        elif protocol_type == ProtocolType.GRPC:
            return GRPCProtocolAdapter()
        elif protocol_type in [ProtocolType.REST_API, ProtocolType.WEBSOCKET]:
            return JSONProtocolAdapter(protocol_type)
        else:
            # Default to JSON adapter for INTERNAL
            return JSONProtocolAdapter(ProtocolType.INTERNAL)