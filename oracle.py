from abc import ABC, abstractmethod
import random

class Oracle(ABC):
	@abstractmethod
	def compute(query: int, trace: list[object], fromm: int, to: int) -> float:
		pass

class RandomOracle(Oracle):
	def compute(query: int, trace: list[object], fromm: int, to: int) -> float:
		return random.random()
