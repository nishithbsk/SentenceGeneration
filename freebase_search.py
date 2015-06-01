import json
import urllib
import sys
from collections import defaultdict

# remove '_' if it is concatenating two words
# remove '_' and append s if it is plural
def sanitize_relation(relation):
	tokens = relation.split('_')
	if tokens[-1] == 's':
		tokens[-2] = tokens[-2] + 's'
		tokens.pop()

	return " ".join(tokens)

def sanitize_compound_arg(arg):
	tokens = arg.split(' - ')
	del tokens[-2:]
	return " ".join(tokens)

def search_api_request(api_key, query):
	service_url = 'https://www.googleapis.com/freebase/v1/search'
	params = {
	  'key': api_key,
	  'query': query
	}

	url = service_url + '?' + urllib.urlencode(params)
	response = json.loads(urllib.urlopen(url).read())
	first_result = response['result'][0]
	topic_name = str(first_result['name'])
	topic_id = str(first_result['mid']) # To be used by the topic API in scraping relations

	return (topic_name, topic_id)


def topic_api_request(api_key, topic):
	service_url = 'https://www.googleapis.com/freebase/v1/topic'
	params = {
	  'key': api_key,
	  'filter': '/people' # Can this be dynamically gotten from search?
	}

	topic_name = topic[0]
	topic_id = topic[1]
	url = service_url + topic_id + '?' + urllib.urlencode(params)
	topic = json.loads(urllib.urlopen(url).read())

	tuples = []
	for property in topic['property']:
		simple_prop = str(property.split('/')[-1])
		simple_prop = sanitize_relation(simple_prop)

		for value in topic['property'][property]['values']:
			arg = value['text'].encode("utf8") 
			if topic['property'][property]['valuetype'] == 'compound':
				arg = sanitize_compound_arg(arg) 
			tuples.append((topic_name, simple_prop, str(arg)))

	return tuples

def print_tuples(tuples):
	for t in tuples:
		print t

def construct_ppdb_from_file(file_name):
	ppdb = defaultdict(list)
	count = 0
	with open(file_name) as f:
		for line in f:
			data = line.split("|||")
			pos_tag = data[0]
			source = data[1]
			target = data[2]
			ppdb[source].append(target)
			count += 1

	print str(count) + " paraphrases added to PPDB"


def expand_tuple_for_phrase(tuple, phrase, ppdb, tuples):
	print "** expand_tuple_for_phrase ** tuple: " + tuple + " phrase: " + phrase
	for paraphrase in ppdb[phrase]:
		new_tuple = (tuple[0], paraphrase, tuple[2])
		tuples.append(new_tuple)
		print "new_tuple: " + new_tuple

def expand_tuples_with_ppdb(ppdb, tuples):
	for t in tuples:
		phrase = t[1]
		expand_tuple_for_phrase(t, phrase, ppdb, tuples)
		for sub_phrase in phrase.split(' '):
			expand_tuple_for_phrase(t, sub_phrase, ppdb, tuples)


api_key = open(".api_key").read()

query = " ".join(sys.argv[1:]) # Join all arguments to form one search query
topic = search_api_request(api_key, query)

print "***"
print "Query: \"" + query + "\" => found topic_name: " + topic[0] + " topic_id:  " + topic[1]
print "***"

tuples = topic_api_request(api_key, topic)

print_tuples(tuples)

ppdb = construct_ppdb_from_file("ppdb-1.0-s-all")

expand_tuples_with_ppdb(ppdb, tuples)



