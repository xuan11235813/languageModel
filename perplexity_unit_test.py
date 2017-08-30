import numpy as np
import forward_backward as FB
import perplexity as pp
targetNum = 4
sourceNum = 3

alignment = np.array([
[0.0,0.0,0.1,0.2,0.3],
[0.0,0.1,0.2,0.3,0.0],
[0.1,0.2,0.3,0.0,0.0],
[0.0,0.0,0.1,0.2,0.3],
[0.0,0.1,0.2,0.3,0.0],
[0.1,0.2,0.3,0.0,0.0],
[0.0,0.0,0.1,0.2,0.3],
[0.0,0.1,0.2,0.3,0.0],
[0.1,0.2,0.3,0.0,0.0]
])

lexicon = np.array([0.1,0.2,0.1,
			0.2,0.2,0.2,
			0.1,0.2,0.2,
			0.2,0.2,0.1,
			])

alignmentInitial= [np.array([0.0,0.0,0.2,0.2,0.6])]

print(alignment)
print(lexicon)

perplexity = pp.Perplexity()

perplexity.addSequence(lexicon, alignment, alignmentInitial, targetNum, sourceNum)

print(perplexity.getPerplexity())