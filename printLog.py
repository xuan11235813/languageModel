from time import gmtime, strftime, localtime

class Log:
	def __init__(self):
		self.fileName = 'log.log'
		self.lastRecordFileName = 'record.config.log'

	def writeSequence(self, str):
		try:
			f = open(self.fileName, 'a')
			f.write(strftime('%y-%m-%d %H:%M:%S', localtime()) + ': ')
			f.write(str + '\n')
			f.close()

		except IOError:
			f = open(self.fileName, 'w')
			f.write(strftime('%y-%m-%d %H:%M:%S', localtime()) + ': ')
			f.write(str + '\n')
			f.close()
	
	# These two function will be depracated later
	def writeRecordInformation(self, str):
		self.saveParameter('oldTrainingFilePosition', str)

	def readRecordInformation(self):
		itemDic = {}
		try:
			f = open(self.lastRecordFileName, 'r')
			for line in f:
				item = []
				for word in line.split(" "):
					item.append(word)
				itemDic[item[0].rstrip()] = item[1].rstrip()
			f.close()
			return int(itemDic['oldTrainingFilePosition'])
		except:
			return 0

	def saveParameter(self, parameterName, parameterValue):
		
		try:
			itemDic = {}
			f = open(self.lastRecordFileName, 'r')
			for line in f:
				item = []
				for word in line.split(" "):
					item.append(word)
				itemDic[item[0].rstrip()] = item[1].rstrip()
			f.close()

			itemDic[parameterName] = parameterValue

			f = open(self.lastRecordFileName, 'w')
			for key in itemDic:
				f.write(key + ' ')
				f.write(itemDic[key] + '\n')
			f.close

		except IOError:
			f = open(self.lastRecordFileName, 'w')
			f.write(parameterName + ' ')
			f.write(parameterValue + '\n')
			f.close()
			
		return 0

		

	def readParameter(self, parameterName):
		itemDic = {}
		try:
			f = open(self.lastRecordFileName, 'r')
			for line in f:
				item = []
				for word in line.split(" "):
					item.append(word)
				itemDic[item[0].rstrip()] = item[1].rstrip()
			f.close()
			return itemDic[parameterName]
		except:
			
			return 0


