import threading


# Communication framework for multi-agent consensus
class AgentCoordinator:
    def __init__(self):
        self.shared_data = {}
        self.lock = threading.Lock()
        self.subscribers = []  # Agents subscribe to receive updates

    def update_data(self, key, value):
        with self.lock:
            self.shared_data[key] = value
        self.notify_agents()

    def get_data(self, key):
        with self.lock:
            return self.shared_data.get(key, None)

    def notify_agents(self):
        for agent in self.subscribers:
            agent.react_to_update()

    def subscribe(self, agent):
        self.subscribers.append(agent)
