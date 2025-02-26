# Base Agent class for shared behavior
class BaseAgent:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.coordinator.subscribe(self)

    def react_to_update(self):
        pass