import MemoryLevel
import LogicLevel
import time

class LevelClassificationModel():
    def __init__(self):
        self.memory_level = MemoryLevel.MemoryLevelModel()
        self.logic_level = LogicLevel.LogicLevelModel()

    def predict(self, input_a, input_b):
        memory_output = self.memory_level.predict(input_a, input_b)
        logic_output = self.logic_level.predict(input_a, input_b)

        return memory_output, logic_output
