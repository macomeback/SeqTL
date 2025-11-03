from abc import ABC, abstractmethod
import random

class Oracle(ABC):
	@abstractmethod
	def compute(self, query: int, trace: list[object], fromm: int, to: int) -> float:
		pass

class RandomOracle(Oracle):
	def compute(self, query: int, trace: list[object], fromm: int, to: int) -> float:
		response = random.random()
		return response
