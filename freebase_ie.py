
# coding: utf-8

import json
import urllib
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
        'query': query,
    }
    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())
    return response

def extract_first_result(response):
    first_result = response['result'][0]
    topic_name = str(first_result['name'])
    topic_id = str(first_result['mid']) # To be used by the topic API in scraping relations
    return (topic_name, topic_id)

def topic_api_request(api_key, topic):
    service_url = 'https://www.googleapis.com/freebase/v1/topic'
    params = {
      'key': api_key,
    }
    topic_id = topic[1]
    url = service_url + topic_id + '?' + urllib.urlencode(params)
    topic = json.loads(urllib.urlopen(url).read())
    return topic

def print_tuples(tuples):
    for t in tuples:
        print t[0]
        print "==="
        print t[1]
        print "==="
        print t[2]
        print "\n"

def construct_ppdb_from_file(file_name):
    ppdb = defaultdict(list)
    count = 0
    print "Loading paraphrases from " + file_name
    with open(file_name) as f:
        for line in f:
            data = line.split(" ||| ")
            source = data[1]
            target = data[2]
            ppdb[source].append(target)
            count += 1
            
    print str(count) + " paraphrases added to PPDB"
    return ppdb


def expand_tuple_for_phrase(t, phrase, ppdb, tuples):
    for paraphrase in ppdb[phrase]:
        t[1].append(paraphrase)

def expand_tuples_with_ppdb(ppdb, tuples):
    
    for t in tuples:
        phrase = t[1][0]
        expand_tuple_for_phrase(t, phrase, ppdb, tuples)
        for sub_phrase in phrase.split(' '):
            expand_tuple_for_phrase(t, sub_phrase, ppdb, tuples)

def construct_tuples(name, response):
    tuples = []
    if 'property' not in response: 
        return [] 

    for property in response['property']:
        simple_prop = str(property.split('/')[-1])
        simple_prop = sanitize_relation(simple_prop)
        
        for value in response['property'][property]['values']:
            if not value['text']:
                continue 
            arg = value['text'].encode("utf8") 
            if response['property'][property]['valuetype'] == 'compound':
                arg = sanitize_compound_arg(arg)     
           
            tuples.append((name, [simple_prop], str(arg)))
            
    return tuples

def allow_tuple(tuple):
     
    disallowed_relations = ['key', 'type', 'creator', 'image', 'timestamp', 'guid', 'attribution']
    # If t contains '/' we skip.
    # These are indicative of a Freebase topic-topic link, url, or other meta-data.
    if not tuple[2] or '/' in tuple[2]:
        return False
    if tuple[1][0] in disallowed_relations:
        return False
    if 'notable' in tuple[1][0]:
        return False
    
    return True

def sanitize_tuples(tuples, log_blocked=False):
    allowed_tuples = []
    blocked_tuples = []
    for t in tuples:
        if allow_tuple(t):
            allowed_tuples.append(t)
        else:
            blocked_tuples.append(t)
       
    if (log_blocked):
        print "== Start Blocked Tuples =="
        print_tuples(blocked_tuples)
        print "== End Blocked Tuples == \n"

    return allowed_tuples

def random_freebase_topic():
    service_url = 'http://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'random',
        'rnlimit': '1',
        'rnnamespace': '0'
    }
    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())
    title = response['query']['random'][0]['title']
    
    try:
        encoded_title = urllib.quote_plus(title).replace('+', '_')
    except:
        print "Error parsing title: " + title
        return None
    
    topic_name = title
    topic_id = "/wikipedia/en/" + encoded_title
    return (topic_name, topic_id)

# query=None defaults to random
# used_ppdb defines whether each relation should be expanded with ppdb
# min_tuples is used when no query is set, topics will be randomly selected until min_tuples is satisfied.
def extract_freebase_tuples(query=None, used_ppdb=False, min_tuples=6, ppdb=None):
    api_key = open(".api_key").read()
    tuples = []
    
    while True:
        
        if query:
            search_result = search_api_request(api_key, query)
            topic = extract_first_result(search_result)
        else:
            topic = random_freebase_topic()
            if not topic:
                continue
            
        response = topic_api_request(api_key, topic)
        tuples = construct_tuples(topic[0], response)
        tuples = sanitize_tuples(tuples, log_blocked=False)
        
        # Only if topics are randomly selected do we continue until min_tuples
        if query or len(tuples) >= min_tuples:
            break
    
    if used_ppdb:
        expand_tuples_with_ppdb(ppdb, tuples)
    
    return tuples

# ============== BEGIN SCRIPT ==============

#ppdb = construct_ppdb_from_file("ppdb-1.0-s-all")

# Example query for "Obama" with PPDB turned off
#tuples = extract_freebase_tuples(query="Obama", used_ppdb=False, ppdb=ppdb)

# Example query for "Stanford" with PPDB expansion turned on
# tuples = extract_freebase_tuples(query="Stanford", used_ppdb=True, ppdb=ppdb)

# Example random query, requiring 10 relation tuples (pre-PPDB) 
#tuples = extract_freebase_tuples(query=None, used_ppdb=False, min_tuples=10, ppdb=ppdb)

#print_tuples(tuples)

