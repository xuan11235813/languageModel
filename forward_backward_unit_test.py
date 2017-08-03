import numpy as np
import forward_backward as FB
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

print(alignment)
print(lexicon)

fb = FB.ForwardBackward()

gamma, alignmentGamma = fb.calculateForwardBackward(lexicon, alignment,targetNum, sourceNum)

print(gamma)
print(alignmentGamma)
