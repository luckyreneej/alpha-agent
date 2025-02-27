import threading
import queue
import logging
import time
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
import uuid

from utils.communication.message import Message, MessageType, MessagePriority, create_error_message, create_response_message

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CommunicationManager:
    """
    Central communication manager for multi-agent systems.
    Handles message routing, topic subscriptions, and request-response patterns.
    Implements a pub-sub model similar to metaGPT's messaging system.
    """
    
    def __init__(self):
        """
        Initialize the communication manager.
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
        
        # Threading components
        self.running = False
        self.router_thread = None
        self.timeout_thread = None
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Configuration
        self.max_history_size = 1000  # Maximum messages per agent history
    
    def start(self):
        """
        Start the message router thread.
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
        Stop the message router thread.
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
            
            # Remove from all topic subscriptions
            for topic, subscribers in list(self.topic_subscriptions.items()):
                if agent_id in subscribers:
                    subscribers.remove(agent_id)
                    # Remove topic if no more subscribers
                    if not subscribers:
                        del self.topic_subscriptions[topic]
            
            logger.info(f"Agent {agent_id} unregistered successfully")
            return True
    
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
        
        # Send the request
        if not self.send_message(request_message):
            # Failed to send
            with self.lock:
                if request_message.id in self.pending_responses:
                    del self.pending_responses[request_message.id]
            return None
        
        # Wait for response with timeout
        response_received = response_event.wait(timeout=timeout)
        
        # Clean up and return result
        with self.lock:
            if request_message.id in self.pending_responses:
                del self.pending_responses[request_message.id]
            
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
    
    def _route_message(self, message: Message, priority: int = MessagePriority.NORMAL.value):
        """
        Route a message to its recipient's queue.
        
        Args:
            message: Message to route
            priority: Message priority (lower number = higher priority)
        """
        with self.lock:
            receiver_id = message.receiver_id
            
            # Handle request messages to check for automatic handling
            if message.message_type == MessageType.REQUEST:
                request_type = message.metadata.get('request_type') if message.metadata else None
                
                # Check if a handler is registered for this request type
                if request_type and receiver_id in self.request_handlers and request_type in self.request_handlers[receiver_id]:
                    handler = self.request_handlers[receiver_id][request_type]
                    
                    try:
                        # Process the request in a separate thread to avoid blocking
                        threading.Thread(target=self._handle_request, 
                                       args=(message, handler, request_type)).start()
                        return
                    except Exception as e:
                        logger.error(f"Error launching request handler thread: {e}")
                        # Continue with normal message routing if thread creation fails
            
            # Handle response messages to check if a pending request is waiting for it
            if message.message_type == MessageType.RESPONSE and message.reply_to:
                request_id = message.reply_to
                
                if request_id in self.pending_responses:
                    # Store the response and signal the waiting thread
                    self.response_messages[request_id] = message
                    self.pending_responses[request_id].set()
                    
                    # Still route the message to the receiver's queue
            
            # Add message to receiver's queue
            if receiver_id in self.agent_queues:
                # Determine the priority value (use message's priority if available)
                if isinstance(message.priority, MessagePriority):
                    msg_priority = message.priority.value
                else:
                    msg_priority = priority
                
                self.agent_queues[receiver_id].put((msg_priority, message))
                
                # Add to message history
                if receiver_id in self.message_history:
                    self.message_history[receiver_id].append(message)
                    
                    # Trim history if it exceeds max size
                    if len(self.message_history[receiver_id]) > self.max_history_size:
                        self.message_history[receiver_id] = self.message_history[receiver_id][-self.max_history_size:]
            else:
                logger.warning(f"Cannot route message: receiver {receiver_id} has no queue")
    
    def _handle_request(self, request: Message, handler: Callable, request_type: str):
        """
        Handle a request message using the registered handler.
        
        Args:
            request: Request message
            handler: Handler function
            request_type: Type of request
        """
        try:
            # Call the handler with the message
            result = handler(request)
            
            # Create a response message
            response = Message(
                sender_id=request.receiver_id,
                receiver_id=request.sender_id,
                message_type=MessageType.RESPONSE,
                content=result,
                correlation_id=request.correlation_id,
                reply_to=request.id,
                metadata={
                    'request_type': request_type,
                    'auto_response': True
                }
            )
            
            # Send the response
            self._route_message(response)
            
        except Exception as e:
            logger.error(f"Error in request handler for {request_type}: {e}")
            
            # Send error response
            error_message = create_error_message(
                sender_id=request.receiver_id,
                receiver_id=request.sender_id,
                error=f"Error handling request: {str(e)}",
                reply_to=request.id,
                correlation_id=request.correlation_id,
                metadata={
                    'request_type': request_type,
                    'auto_response': True
                }
            )
            
            self._route_message(error_message, MessagePriority.HIGH.value)
    
    def _message_router(self):
        """
        Background thread for routing messages.
        Currently a placeholder since routing is done directly in _route_message().
        """
        logger.info("Message router thread started")
        
        while self.running:
            # The main routing is done in _route_message, but this thread
            # could be used for periodic tasks related to message routing
            time.sleep(1)  # Sleep to avoid busy waiting
            
            # Check for expired timeout messages periodically
            # (Actual timeout handling is in _timeout_monitor thread)
        
        logger.info("Message router thread stopped")
    
    def _timeout_monitor(self):
        """
        Background thread for monitoring message timeouts.
        """
        logger.info("Timeout monitor thread started")
        
        while self.running:
            try:
                with self.lock:
                    now = time.time()
                    expired_messages = [msg_id for msg_id, expiry in self.timeout_messages.items() if now > expiry]
                    
                    for msg_id in expired_messages:
                        # Handle the timeout - could notify sender or trigger other actions
                        del self.timeout_messages[msg_id]
                        
                        # If this is a pending request that timed out, signal the event
                        if msg_id in self.pending_responses:
                            self.pending_responses[msg_id].set()
                            # The request handler will notice the response message is missing
            except Exception as e:
                logger.error(f"Error in timeout monitor: {e}")
            
            time.sleep(0.5)  # Check every half second
        
        logger.info("Timeout monitor thread stopped")
    
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