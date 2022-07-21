from algoliasearch.search_client import SearchClient
from clean_utils import rmhtml
import json
import re
import keys

client = SearchClient.create(keys.KEY["ID"], keys.KEY["access"])
index = client.init_index('asknature_searchable_posts')
query = '' 
res = index.browse_objects({'query': query, 'attributesToRetrieve': ['reference_sources', 'summary']})

def clean_source(input):
    input = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", input)
    input = rmhtml.strip_tags(input)
    return input

data = {'summary': [], "source_excerpt": []}
with open('AskNature-Summarize/data/asknature-data.json', 'w') as fp:
    for hit in res:
        if 'reference_sources' in hit and 'source_excerpt' in hit['reference_sources'] and ''.join(hit['reference_sources']['source_excerpt']) != '':
            data['summary'].append(''.join(hit['summary']))
            data['source_excerpt'].append(clean_source(''.join(hit['reference_sources']['source_excerpt'])))
    json.dump(data, fp)


