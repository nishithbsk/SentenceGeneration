import sys
from collections import defaultdict

file_name = sys.argv[1]
ppdb = defaultdict(list)

count = 0

with open(file_name) as f:
	for line in f:
		data = line.split("|||")
		pos_tag = data[0]
		source = data[1]
		target = data[2]
		ppdb[source].append(target)
		print "source: " + source 
		print "target: " +  target
		print ""
		count += 1

print "Ready..."

def query_phrase(p):
		print ppdb[source]

while True:
	source = input("Type input to find paraphrases for (or q to quit): ")
	if source == 'q':
		break
