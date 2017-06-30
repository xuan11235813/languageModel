import random

class BaumWelch:

	stateNum = 0
	tranProb = []
	startProb = []
	emiProb = []

	def __init__(self, _stateNum):
		self.stateNum = _stateNum
		for i in range(_stateNum):
			self.startProb.append(random.uniform(0,1))
			self.emiProb.append(random.uniform(0,1))
			item = []
			normalizeTerm = 0;
			for j in range(_stateNum):
				magic = random.uniform(0,1)
				item.append(magic)
				normalizeTerm += magic
			for j in range(_stateNum):
				item[j]/= normalizeTerm
			self.tranProb.append(item)
		normalizeTermStart = sum(self.startProb)
		normalizeTermEmission = sum(self.emiProb)
		for i in range(_stateNum):
			self.startProb[i]/= normalizeTermStart
			self.emiProb[i]/= normalizeTermEmission

	def transferData(self, _stateNum, _tranProb, _emiProb, _startProb):
		self.stateNum = _stateNum
		self.tranProb = _tranProb
		self.startProb = _startProb
		self.emiProb = _emiProb

	def excute(self):
		print(self.tranProb)
		print(self.startProb)
		print(self.emiProb)

		for i in range(self.stateNum):
			print(sum(self.tranProb[i]))


a = BaumWelch(10)

a.excute();





