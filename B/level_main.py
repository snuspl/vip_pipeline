import MemoryLevel
import LogicLevel
import time

class LevelClassificationModel():
    def __init__(self):
        self.memory_level = MemoryLevel.MemoryLevelModel()
        self.logic_level = LogicLevel.LogicLevelModel()

    def predict(self, input_data):
        text_a = input_data["question"]
        text_b = input_data["a"]

        memory_output = self.memory_level.predict(text_a, text_b)
        logic_output = self.logic_level.predict(text_a, text_b)

        return [memory_output, logic_output]
