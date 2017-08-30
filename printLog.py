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
	def writeRecordInformation(self, str):
		f = open(self.lastRecordFileName, 'w')			
		f.write(str + '\n')
		f.close()

	def readRecordInformation(self):
		try:
			f = open(self.lastRecordFileName, 'r')
			line = int(f.readline())
			return line

		except:
			return 0