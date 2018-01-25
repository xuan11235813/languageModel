import dataInput as data
import process
import printLog
import para

parameter = para.Para()

_measureData = data.ReadData(parameter.ReadTestFile())

_process = process.ProcessLSTM()

_process.processPerplexity(_measureData.getCurrentBatch())