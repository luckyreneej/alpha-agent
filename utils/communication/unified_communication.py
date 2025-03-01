import threading
import queue
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Callable, Union

from utils.communication.message import Message, MessageType, MessagePriority, create_error_message, \
    create_system_message

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedCommunicationManager:
    """
    A unified communication system for multi-agent environments.
    Combines direct messaging, pub-sub, request-response, and conversation capabilities.
    """

    def __init__(self):
        """
        Initialize the unified communication manager.
        """
        # Core message routing components
        self.agent_queues: Dict[str, queue.PriorityQueue] = {}
        self.topic_subscriptions: Dict[str, Set[str]] = {}
        self.registered_agents: Set[str] = set()
        self.request_handlers: Dict[str, Dict[str, Callable]] = {}

        # Message tracking
        self.message_history: Dict[str, List[Message]] = {}
        self.pending_responses: Dict[str, threading.Event] = {}  # message_id -> event
        self.response_messages: Dict[str, Message] = {}  # message_id -> response message
        self.timeout_messages: Dict[str, float] = {}  # message_id -> expiration time

        # Shared data (common memory between agents)
        self.shared_data: Dict[str, Any] = {}

        # Conversation management
        self.conversations: Dict[str, Dict[str, Any]] = {}  # conversation_id -> conversation data
        self.agent_conversations: Dict[str, Set[str]] = {}  # agent_id -> set of conversation_ids

        # Threading components
        self.running = False
        self.router_thread = None
        self.timeout_thread = None
        self.lock = threading.RLock()  # Reentrant lock for thread safety

        # Configuration
        self.max_history_size = 1000  # Maximum messages per agent history

    # -------------------------------------------------------------------------
    # Core Agent Registration Methods
    # -------------------------------------------------------------------------

    def start(self):
        """
        Start the message router and monitoring threads.
        """
        with self.lock:
            if self.running:
                logger.warning("Communication manager already running")
                return

            self.running = True

            # Start router thread
            self.router_thread = threading.Thread(target=self._message_router, daemon=True)
            self.router_thread.start()

            # Start timeout monitor thread
            self.timeout_thread = threading.Thread(target=self._timeout_monitor, daemon=True)
            self.timeout_thread.start()

            logger.info("Communication manager started successfully")

    def stop(self):
        """
        Stop the message router and monitoring threads.
        """
        with self.lock:
            if not self.running:
                logger.warning("Communication manager already stopped")
                return

            self.running = False

            # Wait for threads to terminate
            if self.router_thread:
                self.router_thread.join(timeout=5)
            if self.timeout_thread:
                self.timeout_thread.join(timeout=5)

            logger.info("Communication manager stopped successfully")

    def register_agent(self, agent_id: str) -> bool:
        """
        Register an agent with the communication manager.

        Args:
            agent_id: Unique identifier for this agent

        Returns:
            True if registration was successful, False otherwise
        """
        with self.lock:
            if agent_id in self.registered_agents:
                logger.warning(f"Agent {agent_id} already registered")
                return False

            self.registered_agents.add(agent_id)
            self.agent_queues[agent_id] = queue.PriorityQueue()
            self.message_history[agent_id] = []
            self.request_handlers[agent_id] = {}

            logger.info(f"Agent {agent_id} registered successfully")
            return True

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the communication manager.

        Args:
            agent_id: Agent ID to unregister

        Returns:
            True if successfully unregistered, False if not registered
        """
        with self.lock:
            if agent_id not in self.registered_agents:
                logger.warning(f"Agent {agent_id} not registered")
                return False

            # Remove from registered agents
            self.registered_agents.remove(agent_id)

            # Clean up agent's queue and history
            if agent_id in self.agent_queues:
                del self.agent_queues[agent_id]
            if agent_id in self.message_history:
                del self.message_history[agent_id]
            if agent_id in self.request_handlers:
                del self.request_handlers[agent_id]

            # Remove from topic subscriptions
            for topic, subscribers in list(self.topic_subscriptions.items()):
                if agent_id in subscribers:
                    subscribers.remove(agent_id)
                    # Remove topic if no more subscribers
                    if not subscribers:
                        del self.topic_subscriptions[topic]

            # Remove agent from conversations
            self._cleanup_agent_conversations(agent_id)

            logger.info(f"Agent {agent_id} unregistered successfully")
            return True

    def _cleanup_agent_conversations(self, agent_id: str) -> None:
        """
        Remove an agent from all conversations they're participating in.

        Args:
            agent_id: Agent ID to clean up
        """
        if agent_id not in self.agent_conversations:
            return

        # Get all conversations this agent is part of
        conversation_ids = list(self.agent_conversations[agent_id])

        # Remove agent from each conversation
        for conv_id in conversation_ids:
            if conv_id in self.conversations:
                if agent_id in self.conversations[conv_id]['participants']:
                    self.conversations[conv_id]['participants'].remove(agent_id)

                    # Notify other participants
                    notification = {
                        'event': 'agent_left',
                        'conversation_id': conv_id,
                        'agent_id': agent_id,
                        'reason': 'agent_unregistered'
                    }

                    for participant_id in self.conversations[conv_id]['participants']:
                        self._send_system_message(participant_id, notification)

                    # Remove conversation if empty
                    if not self.conversations[conv_id]['participants']:
                        del self.conversations[conv_id]

        # Clear agent's conversation associations
        del self.agent_conversations[agent_id]

    # -------------------------------------------------------------------------
    # Direct Messaging Methods
    # -------------------------------------------------------------------------

    def send_message(self, message: Message) -> bool:
        """
        Send a message to its recipient.

        Args:
            message: Message to send

        Returns:
            True if message was sent successfully, False otherwise
        """
        with self.lock:
            # Validate sender
            if message.sender_id not in self.registered_agents and message.sender_id != "system":
                logger.warning(f"Sender {message.sender_id} not registered")
                return False

            # Handle broadcast messages
            if message.message_type == MessageType.BROADCAST:
                for agent_id in self.registered_agents:
                    if agent_id != message.sender_id:  # Don't send to self
                        # Create a copy of the message for each recipient
                        broadcast_copy = Message(
                            sender_id=message.sender_id,
                            receiver_id=agent_id,
                            message_type=message.message_type,
                            content=message.content,
                            correlation_id=message.correlation_id,
                            metadata=message.metadata.copy() if message.metadata else None
                        )
                        self._route_message(broadcast_copy)
                return True

            # For directed messages, verify receiver is registered
            if message.receiver_id not in self.registered_agents:
                logger.warning(f"Receiver {message.receiver_id} not registered")
                return False

            # Route the message
            self._route_message(message)
            return True

    def _route_message(self, message: Message) -> None:
        """
        Route a message to its destination and handle special cases.

        Args:
            message: Message to route
        """
        # Check for response messages
        if message.message_type == MessageType.RESPONSE and message.correlation_id:
            # Store response and signal waiting thread
            if message.correlation_id in self.pending_responses:
                self.response_messages[message.correlation_id] = message
                self.pending_responses[message.correlation_id].set()

        # Check for request messages and automatically handle if there's a registered handler
        if message.message_type == MessageType.REQUEST:
            if message.receiver_id in self.request_handlers:
                request_type = message.metadata.get('request_type', '') if message.metadata else ''
                if request_type and request_type in self.request_handlers[message.receiver_id]:
                    # Schedule automatic handling of the request
                    handler = self.request_handlers[message.receiver_id][request_type]
                    threading.Thread(
                        target=self._handle_request,
                        args=(message, handler),
                        daemon=True
                    ).start()

        # Put message in recipient's queue
        if message.receiver_id in self.agent_queues:
            # Determine priority (lower number = higher priority)
            priority = 1 if message.priority == MessagePriority.HIGH else 2 if message.priority == MessagePriority.NORMAL else 3

            # Add to queue with priority
            self.agent_queues[message.receiver_id].put((priority, message))

            # Add to history
            if message.receiver_id in self.message_history:
                history = self.message_history[message.receiver_id]
                history.append(message)

                # Limit history size
                if len(history) > self.max_history_size:
                    self.message_history[message.receiver_id] = history[-self.max_history_size:]

    def _handle_request(self, request_message: Message, handler: Callable) -> None:
        """
        Handle a request with a registered handler and send response.

        Args:
            request_message: The request message
            handler: Handler function for this request type
        """
        try:
            # Call handler with the message
            result = handler(request_message)

            # Create response message
            response = Message(
                sender_id=request_message.receiver_id,
                receiver_id=request_message.sender_id,
                message_type=MessageType.RESPONSE,
                content=result,
                correlation_id=request_message.id
            )

            # Send response
            self.send_message(response)
        except Exception as e:
            # Create error response
            error_msg = create_error_message(
                sender_id=request_message.receiver_id,
                receiver_id=request_message.sender_id,
                error_message=str(e),
                correlation_id=request_message.id
            )
            self.send_message(error_msg)
            logger.error(f"Error handling request {request_message.id}: {e}")

    def _send_system_message(self, receiver_id: str, content: Any,
                             priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """
        Send a system message to an agent.

        Args:
            receiver_id: Receiver agent ID
            content: Message content
            priority: Message priority

        Returns:
            True if message sent successfully
        """
        if receiver_id not in self.registered_agents:
            return False

        message = create_system_message(receiver_id, content, priority=priority)
        return self.send_message(message)

    def _message_router(self) -> None:
        """
        Main message routing thread. Handles timed deliveries and other scheduled tasks.
        """
        logger.info("Message router thread started")

        while self.running:
            try:
                # Process any pending timed deliveries or other tasks here

                # Sleep to prevent CPU spinning
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in message router thread: {e}")

        logger.info("Message router thread stopped")

    def _timeout_monitor(self) -> None:
        """
        Monitor for request timeouts and clean up.
        """
        logger.info("Timeout monitor thread started")

        while self.running:
            try:
                current_time = time.time()

                # Check for timeout messages
                with self.lock:
                    expired_ids = [mid for mid, expiry in self.timeout_messages.items() if expiry <= current_time]

                    for message_id in expired_ids:
                        # Clean up expired messages
                        if message_id in self.timeout_messages:
                            del self.timeout_messages[message_id]

                        # If there's a pending response, notify with timeout
                        if message_id in self.pending_responses:
                            # Create a timeout response
                            receiver_id = None
                            for agent_id in self.registered_agents:
                                for msg in self.message_history.get(agent_id, []):
                                    if msg.id == message_id:
                                        receiver_id = msg.sender_id
                                        break
                                if receiver_id:
                                    break

                            if receiver_id:
                                timeout_response = create_error_message(
                                    sender_id="system",
                                    receiver_id=receiver_id,
                                    error_message="Request timed out",
                                    correlation_id=message_id
                                )

                                self.response_messages[message_id] = timeout_response
                                self.pending_responses[message_id].set()

                # Sleep to prevent CPU spinning
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in timeout monitor thread: {e}")

        logger.info("Timeout monitor thread stopped")

    def send_request_and_wait(self,
                              sender_id: str,
                              receiver_id: str,
                              request_type: str,
                              content: Any,
                              timeout: float = 30.0,
                              metadata: Optional[Dict[str, Any]] = None) -> Optional[Message]:
        """
        Send a request and wait for a response.

        Args:
            sender_id: Sender agent ID
            receiver_id: Receiver agent ID
            request_type: Type of request
            content: Request content
            timeout: Timeout in seconds
            metadata: Additional metadata

        Returns:
            Response message or None if timed out
        """
        # Validate agents
        with self.lock:
            if sender_id not in self.registered_agents:
                logger.warning(f"Sender {sender_id} not registered")
                return None
            if receiver_id not in self.registered_agents:
                logger.warning(f"Receiver {receiver_id} not registered")
                return None

        # Create metadata with request_type if not provided
        if metadata is None:
            metadata = {}
        metadata['request_type'] = request_type

        # Create request message
        request_message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.REQUEST,
            content=content,
            metadata=metadata
        )

        # Create response event and register it
        with self.lock:
            response_event = threading.Event()
            self.pending_responses[request_message.id] = response_event

            # Set timeout if specified
            if timeout > 0:
                self.timeout_messages[request_message.id] = time.time() + timeout

        # Send the request
        if not self.send_message(request_message):
            # Failed to send
            with self.lock:
                if request_message.id in self.pending_responses:
                    del self.pending_responses[request_message.id]
                if request_message.id in self.timeout_messages:
                    del self.timeout_messages[request_message.id]
            return None

        # Wait for response with timeout
        response_received = response_event.wait(timeout=timeout)

        # Clean up and return result
        with self.lock:
            if request_message.id in self.pending_responses:
                del self.pending_responses[request_message.id]
            if request_message.id in self.timeout_messages:
                del self.timeout_messages[request_message.id]

            if not response_received:
                logger.warning(f"Request {request_message.id} timed out after {timeout} seconds")
                return None

            if request_message.id in self.response_messages:
                response = self.response_messages[request_message.id]
                del self.response_messages[request_message.id]
                return response

            logger.warning(f"Response event triggered but no response found for {request_message.id}")
            return None

    def register_request_handler(self, agent_id: str, request_type: str, handler: Callable[[Message], Any]) -> bool:
        """
        Register a function to handle specific request types.

        Args:
            agent_id: Agent ID
            request_type: Type of request to handle
            handler: Function that takes a message and returns a response

        Returns:
            True if handler was registered successfully, False otherwise
        """
        with self.lock:
            if agent_id not in self.registered_agents:
                logger.warning(f"Agent {agent_id} not registered")
                return False

            if agent_id not in self.request_handlers:
                self.request_handlers[agent_id] = {}

            self.request_handlers[agent_id][request_type] = handler
            logger.info(f"Handler for {request_type} registered by agent {agent_id}")
            return True

    def get_messages(self,
                     agent_id: str,
                     wait: bool = False,
                     timeout: Optional[float] = None) -> List[Message]:
        """
        Get messages for an agent.

        Args:
            agent_id: Agent ID
            wait: Whether to wait for a message if none available
            timeout: Timeout when waiting

        Returns:
            List of messages
        """
        if agent_id not in self.registered_agents:
            logger.warning(f"Agent {agent_id} not registered")
            return []

        try:
            queue_obj = self.agent_queues[agent_id]
            messages = []

            # If wait is true and queue is empty, block until a message arrives or timeout
            if wait and queue_obj.empty():
                try:
                    # Get the priority and message
                    priority, message = queue_obj.get(block=True, timeout=timeout)
                    messages.append(message)
                    queue_obj.task_done()
                except queue.Empty:
                    # Timeout or no messages
                    return []

            # Get any remaining messages without blocking
            while not queue_obj.empty():
                try:
                    priority, message = queue_obj.get_nowait()
                    messages.append(message)
                    queue_obj.task_done()
                except queue.Empty:
                    break

            return messages

        except Exception as e:
            logger.error(f"Error retrieving messages for agent {agent_id}: {e}")
            return []

    def get_message_history(self, agent_id: Optional[str] = None, limit: int = 100) -> List[Message]:
        """
        Get message history for an agent or all messages if agent_id is None.

        Args:
            agent_id: Optional agent ID
            limit: Maximum number of messages to return

        Returns:
            List of historical messages
        """
        with self.lock:
            if agent_id is not None:
                if agent_id not in self.message_history:
                    return []
                history = self.message_history[agent_id]
                return history[-limit:] if limit > 0 else history.copy()
            else:
                # Combine all histories, sort by timestamp, and return most recent ones
                all_messages = []
                for history in self.message_history.values():
                    all_messages.extend(history)
                all_messages.sort(key=lambda m: m.timestamp, reverse=True)
                return all_messages[:limit] if limit > 0 else all_messages

    # -------------------------------------------------------------------------
    # Topic Subscription Methods
    # -------------------------------------------------------------------------

    def subscribe_to_topic(self, agent_id: str, topic: str) -> bool:
        """
        Subscribe an agent to a topic.

        Args:
            agent_id: Agent ID
            topic: Topic to subscribe to

        Returns:
            True if subscription was successful, False otherwise
        """
        with self.lock:
            if agent_id not in self.registered_agents:
                logger.warning(f"Agent {agent_id} not registered")
                return False

            if topic not in self.topic_subscriptions:
                self.topic_subscriptions[topic] = set()

            self.topic_subscriptions[topic].add(agent_id)
            logger.info(f"Agent {agent_id} subscribed to topic {topic}")
            return True

    def unsubscribe_from_topic(self, agent_id: str, topic: str) -> bool:
        """
        Unsubscribe an agent from a topic.

        Args:
            agent_id: Agent ID
            topic: Topic to unsubscribe from

        Returns:
            True if successfully unsubscribed, False otherwise
        """
        with self.lock:
            if topic not in self.topic_subscriptions or agent_id not in self.topic_subscriptions[topic]:
                logger.warning(f"Agent {agent_id} not subscribed to topic {topic}")
                return False

            self.topic_subscriptions[topic].remove(agent_id)

            # Remove topic if no more subscribers
            if not self.topic_subscriptions[topic]:
                del self.topic_subscriptions[topic]

            logger.info(f"Agent {agent_id} unsubscribed from topic {topic}")
            return True

    def publish_to_topic(self,
                         sender_id: str,
                         topic: str,
                         content: Any,
                         metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Publish a message to all subscribers of a topic.

        Args:
            sender_id: ID of the sender
            topic: Topic to publish to
            content: Message content
            metadata: Additional metadata

        Returns:
            Correlation ID of the published messages or None on failure
        """
        with self.lock:
            if sender_id not in self.registered_agents and sender_id != "system":
                logger.warning(f"Sender {sender_id} not registered")
                return None

            if topic not in self.topic_subscriptions or not self.topic_subscriptions[topic]:
                logger.warning(f"No subscribers for topic {topic}")
                return None

            # Create a correlation ID for all published messages
            correlation_id = str(uuid.uuid4())

            # Add topic to metadata
            if metadata is None:
                metadata = {}
            metadata['topic'] = topic

            # Send message to all subscribers
            for subscriber_id in self.topic_subscriptions[topic]:
                # Skip sending to self unless explicitly requested
                if subscriber_id != sender_id or metadata.get('send_to_self', False):
                    message = Message(
                        sender_id=sender_id,
                        receiver_id=subscriber_id,
                        message_type=MessageType.DATA,
                        content=content,
                        correlation_id=correlation_id,
                        metadata=metadata.copy()  # Copy to avoid shared references
                    )
                    self._route_message(message)

            logger.info(f"Message published to topic {topic} by {sender_id} with correlation_id {correlation_id}")
            return correlation_id

    # -------------------------------------------------------------------------
    # Shared Data Methods
    # -------------------------------------------------------------------------

    def update_data(self, key: str, value: Any) -> None:
        """
        Update a value in the shared data.

        Args:
            key: Data key
            value: Data value
        """
        with self.lock:
            self.shared_data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the shared data.

        Args:
            key: Data key
            default: Default value if key doesn't exist

        Returns:
            Data value or default
        """
        with self.lock:
            return self.shared_data.get(key, default)

    # -------------------------------------------------------------------------
    # Conversation Management Methods
    # -------------------------------------------------------------------------

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
                if agent_id not in self.registered_agents:
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

            if agent_id not in self.registered_agents:
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
                    self._route_message(message)

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
            if len(conversation['messages']) > self.max_history_size:
                conversation['messages'] = conversation['messages'][-self.max_history_size:]

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

            # Notify other participants of context change
            notification = {
                'event': 'context_updated',
                'conversation_id': conversation_id,
                'agent_id': agent_id,
                'key': key
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
                    summary = {
                        'conversation_id': conv_id,
                        'topic': conv['topic'],
                        'initiator_id': conv['initiator_id'],
                        'participants': list(conv['participants']),
                        'created_at': conv['created_at'],
                        'updated_at': conv['updated_at'],
                        'message_count': len(conv['messages'])
                    }
                    result.append(summary)

            return result

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

    def _set_timeout(self, message_id: str, timeout_seconds: float):
        """
        Set a timeout for a message.

        Args:
            message_id: Message ID
            timeout_seconds: Timeout in seconds
        """
        with self.lock:
            expiry_time = time.time() + timeout_seconds
            self.timeout_messages[message_id] = expiry_time