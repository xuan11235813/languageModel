import printLog
import tensorflow as tf

ll = printLog.Log()

ll.saveParameter('ab', '3.4')
ll.saveParameter('bb', '2.8')
ll.saveParameter('cb', '333333')
ll.saveParameter('ab', '22')
ll.saveParameter('ab', '3.43')
ll.saveParameter('bb', '22.8')
ll.saveParameter('cb', '3323333')
ll.saveParameter('ab', '25')
ll.writeRecordInformation('222222222222')
print(ll.readParameter('ab'))
print(ll.readParameter('bb'))
print(ll.readParameter('cb'))