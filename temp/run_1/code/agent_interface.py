from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract Base Class for all Dots and Boxes players (Human, Random, MCTS, Deep Learning).
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_move(self, game_state) -> int:
        """
        Given the current DotsAndBoxesGame object, return an integer representing
        the line index to draw. 
        
        For a Human agent, this can return None or raise an exception since 
        the GUI handles click events directly.
        """
        pass

class HumanAgent(BaseAgent):
    def __init__(self, name: str = "Human"):
        super().__init__(name)

    def get_move(self, game_state) -> int:
        # Humans interact via clicks, GUI will bypass this
        return None