import sys

f = open(sys.argv[1],'r')
data = f.read()
f.close()
words = data.split()
dealed = []
count = []
for word in words:
	if word in dealed:
		for index in range(len(dealed)):
			if dealed[index] == word:
				count[index] += 1
				break
	else:
		dealed.append(word)
		count.append(1)
f = open('Q1.txt','w')
for i in range(len(dealed)):
	if i == len(dealed)-1:
		f.write("%s %d %d" %(dealed[i], i, count[i]))
	else:
		f.write("%s %d %d\n" %(dealed[i], i, count[i]))
f.close()