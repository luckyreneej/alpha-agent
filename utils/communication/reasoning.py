from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import time
import logging
import uuid
import threading
import json
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReasoningStep(Enum):
    """Types of reasoning steps in the reasoning chain."""
    OBSERVATION = "observation"  # Observing data or environment
    THOUGHT = "thought"          # Thinking about the problem
    ANALYSIS = "analysis"        # Analyzing data or situation
    REASONING = "reasoning"      # Making logical deductions
    REFLECTION = "reflection"    # Reflecting on previous steps
    ACTION = "action"            # Taking an action
    CONCLUSION = "conclusion"    # Final conclusion

class ReasoningChain:
    """
    Implements a chain of reasoning steps that can be shared between agents.
    Enables structured thinking similar to metaGPT.
    """
    
    def __init__(self, topic: str, creator_id: str):
        """
        Initialize a new reasoning chain.
        
        Args:
            topic: Topic or subject of the reasoning
            creator_id: ID of the agent creating the chain
        """
        self.chain_id = str(uuid.uuid4())
        self.topic = topic
        self.creator_id = creator_id
        self.steps: List[Dict[str, Any]] = []
        self.conclusions: List[Dict[str, Any]] = []
        self.status = "active"  # active, paused, completed
        self.created_at = time.time()
        self.updated_at = time.time()
        self.contributors: Dict[str, int] = {creator_id: 0}  # agent_id -> contribution count
        self.metadata: Dict[str, Any] = {}
        
        # Add initial entry for the reasoning chain
        self._add_entry({
            "type": "chain_start",
            "topic": topic,
            "creator_id": creator_id,
            "timestamp": self.created_at
        })
    
    def add_step(self, 
               step_type: ReasoningStep, 
               content: Any, 
               agent_id: str,
               references: Optional[List[str]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a reasoning step to the chain.
        
        Args:
            step_type: Type of reasoning step
            content: Content of the reasoning
            agent_id: ID of the contributing agent
            references: Optional list of step IDs this step refers to
            metadata: Optional metadata for the step
            
        Returns:
            ID of the added step
        """
        step_id = str(uuid.uuid4())
        step = {
            "id": step_id,
            "type": step_type.value,
            "content": content,
            "agent_id": agent_id,
            "timestamp": time.time(),
            "references": references or []
        }
        
        if metadata:
            step["metadata"] = metadata
        
        self._add_entry(step)
        
        # Update contributor stats
        if agent_id in self.contributors:
            self.contributors[agent_id] += 1
        else:
            self.contributors[agent_id] = 1
        
        self.updated_at = time.time()
        return step_id
    
    def add_conclusion(self, conclusion: Any, agent_id: str, confidence: float = 1.0) -> str:
        """
        Add a conclusion to the reasoning chain.
        
        Args:
            conclusion: Conclusion content
            agent_id: ID of the agent providing the conclusion
            confidence: Confidence level (0.0-1.0)
            
        Returns:
            ID of the added conclusion
        """
        conclusion_id = str(uuid.uuid4())
        conclusion_entry = {
            "id": conclusion_id,
            "content": conclusion,
            "agent_id": agent_id,
            "confidence": confidence,
            "timestamp": time.time()
        }
        
        self.conclusions.append(conclusion_entry)
        
        # Add a conclusion step to the main chain
        self.add_step(
            step_type=ReasoningStep.CONCLUSION,
            content=conclusion,
            agent_id=agent_id,
            metadata={"conclusion_id": conclusion_id, "confidence": confidence}
        )
        
        self.updated_at = time.time()
        return conclusion_id
    
    def get_steps(self, 
                agent_id: Optional[str] = None, 
                step_types: Optional[List[ReasoningStep]] = None) -> List[Dict[str, Any]]:
        """
        Get steps from the reasoning chain, with optional filtering.
        
        Args:
            agent_id: Filter by contributing agent
            step_types: Filter by step types
            
        Returns:
            List of matching steps
        """
        filtered_steps = self.steps.copy()
        
        # Filter by agent ID if specified
        if agent_id is not None:
            filtered_steps = [step for step in filtered_steps if step.get("agent_id") == agent_id]
        
        # Filter by step types if specified
        if step_types is not None:
            step_type_values = [step_type.value for step_type in step_types]
            filtered_steps = [step for step in filtered_steps 
                             if step.get("type") in step_type_values]
        
        return filtered_steps
    
    def get_conclusions(self) -> List[Dict[str, Any]]:
        """Get all conclusions."""
        return self.conclusions.copy()
    
    def get_latest_step(self) -> Optional[Dict[str, Any]]:
        """Get the most recent step in the chain."""
        if self.steps:
            return self.steps[-1]
        return None
    
    def get_step_by_id(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific step by its ID."""
        for step in self.steps:
            if step.get("id") == step_id:
                return step
        return None
    
    def _add_entry(self, entry: Dict[str, Any]) -> None:
        """Add an entry to the steps list."""
        self.steps.append(entry)
    
    def complete(self) -> None:
        """Mark the reasoning chain as completed."""
        self.status = "completed"
        self.updated_at = time.time()
        
        # Add completion entry
        self._add_entry({
            "type": "chain_complete",
            "timestamp": time.time(),
            "contributor_summary": self.contributors
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reasoning chain to dictionary."""
        return {
            "chain_id": self.chain_id,
            "topic": self.topic,
            "creator_id": self.creator_id,
            "steps": self.steps,
            "conclusions": self.conclusions,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "contributors": self.contributors,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert reasoning chain to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningChain':
        """Create reasoning chain from dictionary."""
        chain = cls(topic=data["topic"], creator_id=data["creator_id"])
        chain.chain_id = data["chain_id"]
        chain.steps = data["steps"]
        chain.conclusions = data["conclusions"]
        chain.status = data["status"]
        chain.created_at = data["created_at"]
        chain.updated_at = data["updated_at"]
        chain.contributors = data["contributors"]
        chain.metadata = data.get("metadata", {})
        return chain

class ReasoningManager:
    """
    Manages reasoning chains between agents.
    Provides utilities for cooperative reasoning and knowledge exchange.
    """
    
    def __init__(self):
        """
        Initialize the reasoning manager.
        """
        self.chains: Dict[str, ReasoningChain] = {}
        self.active_chains: Dict[str, List[str]] = {}  # agent_id -> list of active chain IDs
        self.topic_index: Dict[str, List[str]] = {}  # topic keywords -> list of chain IDs
        self.pending_contributions: Dict[str, List[Dict]] = {}  # chain_id -> list of pending contributions
        self.approval_callbacks: Dict[str, Dict[str, Callable]] = {}  # chain_id -> {contribution_id -> callback}
        
        self.lock = threading.RLock()
    
    def create_chain(self, topic: str, creator_id: str) -> str:
        """
        Create a new reasoning chain.
        
        Args:
            topic: Topic or subject of the reasoning
            creator_id: ID of the creating agent
            
        Returns:
            Chain ID
        """
        with self.lock:
            # Create new chain
            chain = ReasoningChain(topic=topic, creator_id=creator_id)
            chain_id = chain.chain_id
            
            # Store chain
            self.chains[chain_id] = chain
            
            # Add to agent's active chains
            if creator_id not in self.active_chains:
                self.active_chains[creator_id] = []
            self.active_chains[creator_id].append(chain_id)
            
            # Index by topic keywords
            keywords = self._extract_keywords(topic)
            for keyword in keywords:
                if keyword not in self.topic_index:
                    self.topic_index[keyword] = []
                self.topic_index[keyword].append(chain_id)
            
            logger.info(f"Created reasoning chain {chain_id} for topic '{topic}'")
            return chain_id
    
    def add_to_chain(self, 
                    chain_id: str, 
                    step_type: ReasoningStep, 
                    content: Any, 
                    agent_id: str,
                    references: Optional[List[str]] = None,
                    require_approval: bool = False) -> str:
        """
        Add a step to a reasoning chain.
        
        Args:
            chain_id: ID of the reasoning chain
            step_type: Type of reasoning step
            content: Content of the reasoning
            agent_id: ID of the contributing agent
            references: Optional list of step IDs this step refers to
            require_approval: Whether this contribution requires approval
            
        Returns:
            ID of the contribution (step ID or pending contribution ID)
        """
        with self.lock:
            if chain_id not in self.chains:
                logger.warning(f"Chain {chain_id} not found")
                return ""
            
            chain = self.chains[chain_id]
            
            # If completed, don't allow more steps
            if chain.status == "completed":
                logger.warning(f"Cannot add to completed chain {chain_id}")
                return ""
            
            # Add agent to active chains if not already present
            if agent_id not in self.active_chains:
                self.active_chains[agent_id] = []
            if chain_id not in self.active_chains[agent_id]:
                self.active_chains[agent_id].append(chain_id)
            
            # Handle contributions requiring approval
            if require_approval and agent_id != chain.creator_id:
                contribution_id = str(uuid.uuid4())
                
                # Store pending contribution
                if chain_id not in self.pending_contributions:
                    self.pending_contributions[chain_id] = []
                
                contribution = {
                    "id": contribution_id,
                    "step_type": step_type,
                    "content": content,
                    "agent_id": agent_id,
                    "references": references or [],
                    "timestamp": time.time()
                }
                
                self.pending_contributions[chain_id].append(contribution)
                logger.info(f"Added pending contribution {contribution_id} to chain {chain_id}")
                
                return contribution_id
            else:
                # Add step directly
                step_id = chain.add_step(
                    step_type=step_type,
                    content=content,
                    agent_id=agent_id,
                    references=references
                )
                
                return step_id
    
    def approve_contribution(self, chain_id: str, contribution_id: str, approver_id: str) -> Optional[str]:
        """
        Approve a pending contribution to a reasoning chain.
        
        Args:
            chain_id: ID of the reasoning chain
            contribution_id: ID of the pending contribution
            approver_id: ID of the approving agent
            
        Returns:
            Step ID if approved, None otherwise
        """
        with self.lock:
            if (chain_id not in self.chains or 
                chain_id not in self.pending_contributions):
                logger.warning(f"Chain {chain_id} not found or has no pending contributions")
                return None
            
            chain = self.chains[chain_id]
            
            # Check if approver is creator or has permission
            if approver_id != chain.creator_id:
                logger.warning(f"Agent {approver_id} not authorized to approve contributions to chain {chain_id}")
                return None
            
            # Find the contribution
            for i, contribution in enumerate(self.pending_contributions[chain_id]):
                if contribution["id"] == contribution_id:
                    # Remove from pending
                    self.pending_contributions[chain_id].pop(i)
                    
                    # Add to chain
                    step_id = chain.add_step(
                        step_type=contribution["step_type"],
                        content=contribution["content"],
                        agent_id=contribution["agent_id"],
                        references=contribution["references"]
                    )
                    
                    # Run approval callback if registered
                    if (chain_id in self.approval_callbacks and 
                        contribution_id in self.approval_callbacks[chain_id]):
                        callback = self.approval_callbacks[chain_id][contribution_id]
                        callback(True, step_id)
                        del self.approval_callbacks[chain_id][contribution_id]
                    
                    logger.info(f"Approved contribution {contribution_id} for chain {chain_id}")
                    return step_id
            
            logger.warning(f"Contribution {contribution_id} not found in chain {chain_id}")
            return None
    
    def reject_contribution(self, chain_id: str, contribution_id: str, approver_id: str, reason: str = "") -> bool:
        """
        Reject a pending contribution to a reasoning chain.
        
        Args:
            chain_id: ID of the reasoning chain
            contribution_id: ID of the pending contribution
            approver_id: ID of the rejecting agent
            reason: Reason for rejection
            
        Returns:
            True if rejected, False otherwise
        """
        with self.lock:
            if (chain_id not in self.chains or 
                chain_id not in self.pending_contributions):
                logger.warning(f"Chain {chain_id} not found or has no pending contributions")
                return False
            
            chain = self.chains[chain_id]
            
            # Check if approver is creator or has permission
            if approver_id != chain.creator_id:
                logger.warning(f"Agent {approver_id} not authorized to reject contributions to chain {chain_id}")
                return False
            
            # Find the contribution
            for i, contribution in enumerate(self.pending_contributions[chain_id]):
                if contribution["id"] == contribution_id:
                    # Remove from pending
                    self.pending_contributions[chain_id].pop(i)
                    
                    # Run approval callback if registered
                    if (chain_id in self.approval_callbacks and 
                        contribution_id in self.approval_callbacks[chain_id]):
                        callback = self.approval_callbacks[chain_id][contribution_id]
                        callback(False, reason)
                        del self.approval_callbacks[chain_id][contribution_id]
                    
                    logger.info(f"Rejected contribution {contribution_id} for chain {chain_id}: {reason}")
                    return True
            
            logger.warning(f"Contribution {contribution_id} not found in chain {chain_id}")
            return False
    
    def get_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """
        Get a reasoning chain by ID.
        
        Args:
            chain_id: ID of the reasoning chain
            
        Returns:
            ReasoningChain or None if not found
        """
        with self.lock:
            return self.chains.get(chain_id)
    
    def get_pending_contributions(self, chain_id: str) -> List[Dict[str, Any]]:
        """
        Get pending contributions for a chain.
        
        Args:
            chain_id: ID of the reasoning chain
            
        Returns:
            List of pending contributions
        """
        with self.lock:
            if chain_id not in self.pending_contributions:
                return []
            
            return self.pending_contributions[chain_id].copy()
    
    def search_chains_by_topic(self, query: str) -> List[str]:
        """
        Search for reasoning chains by topic.
        
        Args:
            query: Search query
            
        Returns:
            List of matching chain IDs
        """
        with self.lock:
            keywords = self._extract_keywords(query)
            matching_chains = set()
            
            for keyword in keywords:
                if keyword in self.topic_index:
                    matching_chains.update(self.topic_index[keyword])
            
            return list(matching_chains)
    
    def get_agent_chains(self, agent_id: str, active_only: bool = True) -> List[ReasoningChain]:
        """
        Get all reasoning chains an agent is involved with.
        
        Args:
            agent_id: Agent ID
            active_only: Whether to return only active chains
            
        Returns:
            List of reasoning chains
        """
        with self.lock:
            if agent_id not in self.active_chains:
                return []
            
            chain_ids = self.active_chains[agent_id]
            chains = []
            
            for chain_id in chain_ids:
                if chain_id in self.chains:
                    chain = self.chains[chain_id]
                    
                    # Filter by status if active_only
                    if not active_only or chain.status == "active":
                        chains.append(chain)
            
            return chains
    
    def complete_chain(self, chain_id: str, agent_id: str) -> bool:
        """
        Mark a reasoning chain as completed.
        
        Args:
            chain_id: ID of the reasoning chain
            agent_id: ID of the agent completing the chain
            
        Returns:
            True if completed, False otherwise
        """
        with self.lock:
            if chain_id not in self.chains:
                logger.warning(f"Chain {chain_id} not found")
                return False
            
            chain = self.chains[chain_id]
            
            # Check if agent is creator or has permission
            if agent_id != chain.creator_id:
                logger.warning(f"Agent {agent_id} not authorized to complete chain {chain_id}")
                return False
            
            chain.complete()
            logger.info(f"Completed reasoning chain {chain_id}")
            
            # Remove from active chains
            for agent_id, chain_ids in self.active_chains.items():
                if chain_id in chain_ids:
                    chain_ids.remove(chain_id)
            
            return True
    
    def register_approval_callback(self, 
                                 chain_id: str, 
                                 contribution_id: str, 
                                 callback: Callable[[bool, Any], None]) -> None:
        """
        Register a callback for when a contribution is approved or rejected.
        
        Args:
            chain_id: ID of the reasoning chain
            contribution_id: ID of the pending contribution
            callback: Function to call when contribution is processed (approved, result_or_reason)
        """
        with self.lock:
            if chain_id not in self.approval_callbacks:
                self.approval_callbacks[chain_id] = {}
            
            self.approval_callbacks[chain_id][contribution_id] = callback
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for indexing.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - split by spaces and take words > 3 chars
        words = text.lower().split()
        return [word for word in words if len(word) > 3]
    
    def get_chain_summary(self, chain_id: str, max_length: int = 500) -> str:
        """
        Generate a summary of the reasoning chain.
        
        Args:
            chain_id: Chain ID
            max_length: Maximum summary length
            
        Returns:
            Chain summary as string
        """
        with self.lock:
            if chain_id not in self.chains:
                return "Chain not found"
            
            chain = self.chains[chain_id]
            
            summary = f"Reasoning Chain: {chain.topic}\n"
            summary += f"Status: {chain.status}\n"
            summary += f"Contributors: {len(chain.contributors)}\n"
            
            # Get conclusions
            conclusions = chain.get_conclusions()
            if conclusions:
                summary += "\nConclusions:\n"
                for conclusion in conclusions[:3]:  # Show at most 3 conclusions
                    summary += f"- {conclusion['content'][:100]}...\n"
            
            # Get key steps (just thoughts and conclusions for brevity)
            thought_steps = chain.get_steps(step_types=[ReasoningStep.THOUGHT, ReasoningStep.CONCLUSION])
            if thought_steps:
                summary += "\nKey Points:\n"
                for step in thought_steps[:5]:  # Show at most 5 key points
                    summary += f"- {step['content'][:100]}...\n"
            
            # Trim if needed
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return summary