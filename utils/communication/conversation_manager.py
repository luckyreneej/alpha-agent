import logging
import threading
import time
from typing import Dict, List, Optional, Any, Set, Tuple, Union
import uuid

from utils.communication.message import Message, MessageType, MessagePriority, create_data_message
from utils.communication.unified_communication import UnifiedCommunicationManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages multi-agent conversations and dialogue contexts.
    Works with the CommunicationManager to provide structured conversation capabilities.
    """

    def __init__(self, communication_manager: UnifiedCommunicationManager):
        """
        Initialize the conversation manager.

        Args:
            communication_manager: Communication manager instance
        """
        self.comm_manager = communication_manager
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.agent_conversations: Dict[str, Set[str]] = {}  # agent_id -> set of conversation_ids
        self.lock = threading.RLock()

    def create_conversation(self,
                            initiator_id: str,
                            participants: List[str],
                            topic: str,
                            initial_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Create a new conversation between agents.

        Args:
            initiator_id: ID of the agent initiating the conversation
            participants: List of participant agent IDs (including initiator)
            topic: Conversation topic
            initial_context: Optional initial context data

        Returns:
            Conversation ID or None if creation failed
        """
        with self.lock:
            # Verify all participants are registered
            for agent_id in participants:
                if agent_id not in self.comm_manager.registered_agents:
                    logger.warning(f"Participant {agent_id} not registered, cannot create conversation")
                    return None

            # Create a unique ID for this conversation
            conversation_id = str(uuid.uuid4())

            # Initialize the conversation
            self.conversations[conversation_id] = {
                'topic': topic,
                'initiator_id': initiator_id,
                'participants': set(participants),
                'created_at': time.time(),
                'updated_at': time.time(),
                'messages': [],
                'context': initial_context or {}
            }

            # Update agent participation tracking
            for agent_id in participants:
                if agent_id not in self.agent_conversations:
                    self.agent_conversations[agent_id] = set()
                self.agent_conversations[agent_id].add(conversation_id)

            # Notify all participants about the new conversation
            notification = {
                'event': 'conversation_created',
                'conversation_id': conversation_id,
                'topic': topic,
                'initiator_id': initiator_id,
                'participants': list(participants)
            }

            for agent_id in participants:
                # Skip sending to initiator since they know they created it
                if agent_id != initiator_id:
                    self._send_system_message(agent_id, notification)

            logger.info(
                f"Created conversation {conversation_id} with topic '{topic}', {len(participants)} participants")
            return conversation_id

    def join_conversation(self, agent_id: str, conversation_id: str) -> bool:
        """
        Add an agent to an existing conversation.

        Args:
            agent_id: Agent ID joining the conversation
            conversation_id: Conversation ID

        Returns:
            True if joined successfully
        """
        with self.lock:
            if conversation_id not in self.conversations:
                logger.warning(f"Conversation {conversation_id} not found")
                return False

            if agent_id not in self.comm_manager.registered_agents:
                logger.warning(f"Agent {agent_id} not registered")
                return False

            # Add agent to conversation
            conversation = self.conversations[conversation_id]
            conversation['participants'].add(agent_id)
            conversation['updated_at'] = time.time()

            # Update agent tracking
            if agent_id not in self.agent_conversations:
                self.agent_conversations[agent_id] = set()
            self.agent_conversations[agent_id].add(conversation_id)

            # Notify other participants
            notification = {
                'event': 'agent_joined',
                'conversation_id': conversation_id,
                'agent_id': agent_id
            }

            for participant_id in conversation['participants']:
                if participant_id != agent_id:
                    self._send_system_message(participant_id, notification)

            logger.info(f"Agent {agent_id} joined conversation {conversation_id}")
            return True

    def leave_conversation(self, agent_id: str, conversation_id: str) -> bool:
        """
        Remove an agent from a conversation.

        Args:
            agent_id: Agent ID leaving the conversation
            conversation_id: Conversation ID

        Returns:
            True if left successfully
        """
        with self.lock:
            if conversation_id not in self.conversations:
                logger.warning(f"Conversation {conversation_id} not found")
                return False

            conversation = self.conversations[conversation_id]

            if agent_id not in conversation['participants']:
                logger.warning(f"Agent {agent_id} is not in conversation {conversation_id}")
                return False

            # Remove agent from conversation
            conversation['participants'].remove(agent_id)
            conversation['updated_at'] = time.time()

            # Update agent tracking
            if agent_id in self.agent_conversations and conversation_id in self.agent_conversations[agent_id]:
                self.agent_conversations[agent_id].remove(conversation_id)

            # Notify other participants
            notification = {
                'event': 'agent_left',
                'conversation_id': conversation_id,
                'agent_id': agent_id
            }

            for participant_id in conversation['participants']:
                self._send_system_message(participant_id, notification)

            # If no more participants, archive or remove the conversation
            if not conversation['participants']:
                logger.info(f"Conversation {conversation_id} has no more participants, removing")
                del self.conversations[conversation_id]

            logger.info(f"Agent {agent_id} left conversation {conversation_id}")
            return True

    def send_conversation_message(self,
                                  conversation_id: str,
                                  sender_id: str,
                                  content: Any,
                                  message_type: MessageType = MessageType.DATA,
                                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a message to all participants in a conversation.

        Args:
            conversation_id: Conversation ID
            sender_id: ID of the sending agent
            content: Message content
            message_type: Message type
            metadata: Additional metadata

        Returns:
            True if message sent successfully
        """
        with self.lock:
            if conversation_id not in self.conversations:
                logger.warning(f"Conversation {conversation_id} not found")
                return False

            conversation = self.conversations[conversation_id]

            if sender_id not in conversation['participants']:
                logger.warning(f"Agent {sender_id} is not in conversation {conversation_id}")
                return False

            # Add conversation metadata
            if metadata is None:
                metadata = {}
            metadata['conversation_id'] = conversation_id
            metadata['conversation_topic'] = conversation['topic']

            # Generate a correlation ID for this conversation message
            correlation_id = str(uuid.uuid4())

            # Send message to all participants except sender
            for agent_id in conversation['participants']:
                if agent_id != sender_id:
                    message = Message(
                        sender_id=sender_id,
                        receiver_id=agent_id,
                        message_type=message_type,
                        content=content,
                        correlation_id=correlation_id,
                        metadata=metadata.copy()
                    )
                    self.comm_manager.send_message(message)

            # Record message in conversation history
            msg_record = {
                'sender_id': sender_id,
                'timestamp': time.time(),
                'content': content,
                'message_type': message_type.value,
                'correlation_id': correlation_id
            }
            conversation['messages'].append(msg_record)
            conversation['updated_at'] = time.time()

            # Limit conversation history size if needed
            if len(conversation['messages']) > 1000:  # Example limit
                conversation['messages'] = conversation['messages'][-1000:]

            logger.info(f"Message sent to conversation {conversation_id} by {sender_id}")
            return True

    def get_conversation_history(self,
                                 conversation_id: str,
                                 agent_id: str,
                                 limit: int = 50) -> Optional[List[Dict[str, Any]]]:
        """
        Get the message history for a conversation.

        Args:
            conversation_id: Conversation ID
            agent_id: Agent requesting the history
            limit: Maximum number of messages to return

        Returns:
            List of message records or None if conversation not found
        """
        with self.lock:
            if conversation_id not in self.conversations:
                logger.warning(f"Conversation {conversation_id} not found")
                return None

            conversation = self.conversations[conversation_id]

            if agent_id not in conversation['participants']:
                logger.warning(f"Agent {agent_id} is not in conversation {conversation_id}")
                return None

            # Return the most recent messages up to the limit
            return conversation['messages'][-limit:]

    def update_conversation_context(self,
                                    conversation_id: str,
                                    agent_id: str,
                                    key: str,
                                    value: Any) -> bool:
        """
        Update a value in the conversation context.

        Args:
            conversation_id: Conversation ID
            agent_id: Agent updating the context
            key: Context key
            value: Context value

        Returns:
            True if updated successfully
        """
        with self.lock:
            if conversation_id not in self.conversations:
                logger.warning(f"Conversation {conversation_id} not found")
                return False

            conversation = self.conversations[conversation_id]

            if agent_id not in conversation['participants']:
                logger.warning(f"Agent {agent_id} is not in conversation {conversation_id}")
                return False

            # Update context
            conversation['context'][key] = value
            conversation['updated_at'] = time.time()

            # Optionally notify other participants of context change
            # This can be useful for collaborative contexts
            notification = {
                'event': 'context_updated',
                'conversation_id': conversation_id,
                'agent_id': agent_id,
                'key': key
                # We don't include the value here - agents should explicitly request it if needed
            }

            for participant_id in conversation['participants']:
                if participant_id != agent_id:
                    self._send_system_message(participant_id, notification, MessagePriority.LOW)

            return True

    def get_conversation_context(self,
                                 conversation_id: str,
                                 agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the conversation context.

        Args:
            conversation_id: Conversation ID
            agent_id: Agent requesting the context

        Returns:
            Conversation context or None if conversation not found
        """
        with self.lock:
            if conversation_id not in self.conversations:
                logger.warning(f"Conversation {conversation_id} not found")
                return None

            conversation = self.conversations[conversation_id]

            if agent_id not in conversation['participants']:
                logger.warning(f"Agent {agent_id} is not in conversation {conversation_id}")
                return None

            # Return a copy of the context to prevent modification
            return conversation['context'].copy()

    def get_conversation_participants(self, conversation_id: str, agent_id: str) -> Optional[List[str]]:
        """
        Get the list of participants in a conversation.

        Args:
            conversation_id: Conversation ID
            agent_id: Agent requesting the participants

        Returns:
            List of participant agent IDs or None if conversation not found
        """
        with self.lock:
            if conversation_id not in self.conversations:
                logger.warning(f"Conversation {conversation_id} not found")
                return None

            conversation = self.conversations[conversation_id]

            if agent_id not in conversation['participants']:
                logger.warning(f"Agent {agent_id} is not in conversation {conversation_id}")
                return None

            return list(conversation['participants'])

    def get_agent_conversations(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get all conversations that an agent is participating in.

        Args:
            agent_id: Agent ID

        Returns:
            List of conversation summary dictionaries
        """
        with self.lock:
            if agent_id not in self.agent_conversations:
                return []

            result = []
            for conv_id in self.agent_conversations[agent_id]:
                if conv_id in self.conversations:
                    conv = self.conversations[conv_id]
                    result.append({
                        'conversation_id': conv_id,
                        'topic': conv['topic'],
                        'initiator_id': conv['initiator_id'],
                        'participant_count': len(conv['participants']),
                        'created_at': conv['created_at'],
                        'updated_at': conv['updated_at'],
                        'message_count': len(conv['messages'])
                    })

            return result

    def _send_system_message(self,
                             agent_id: str,
                             content: Any,
                             priority: MessagePriority = MessagePriority.NORMAL):
        """
        Send a system message to an agent.

        Args:
            agent_id: Agent ID
            content: Message content
            priority: Message priority
        """
        message = Message(
            sender_id="system",
            receiver_id=agent_id,
            message_type=MessageType.SYSTEM,
            content=content,
            priority=priority
        )
        self.comm_manager.send_message(message)