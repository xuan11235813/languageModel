import data
import process


_measureData = data.ReadData(1)

#initialize the process

# 0 or null 
_process = process.ProcessTraditional()


_process.processPerplexity(_measureData.getCurrentBatch())