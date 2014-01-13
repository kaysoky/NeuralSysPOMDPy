from DialogModel import DialogPOMDP
import numpy

numObs = 3
obsMat = numpy.random.rand(numObs, numObs)
for i in range(numObs):
    obsMat[:, i] /= sum(obsMat[:, i])
    
dialogGenerator = DialogPOMDP("Examples/TestQuestions.json", obsMat)
dialogGenerator.GenerateFile("Examples/Test.pomdp")

# Call the POMDP solver outside of Python
# ie. ./pomdpsol --precision 0.25 --timeout 50000 --output Examples/Test.policy Examples/Test.pomdp
#   This will run for about 15 hours or until it reaches an expected reward range of 0.25 