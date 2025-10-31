"""
All data from raw nodes were collected with use of DeepSeek AI
chatbot. You can see system promt in ./promt.txt
"""

import json

tags = set()

with open('./think_python_raw.json', 'r') as input:
    data = json.load(input)
    
for node in data:
    n_tags = set()
    for t in node['tags']:
        tags.add(t.lower()
                     .strip("_")
                     .replace(" ", "_")
                     .replace("-", "_"))
        n_tags.add(t.lower()
                     .strip("_")
                     .replace(" ", "_")
                     .replace("-", "_"))
    node['name'] = node['name'].lower().replace(" ", "_").replace("-", "_")
    node['summary'] = node['summary'].lower()
    node['content'] = node['content'].lower()
    node['source'] = "https://allendowney.github.io/ThinkPython/"
    
with open("../nodes.json", "w+") as out:
    json.dump(data, out)    

print(f"Total themes:\t{len(data)}")
print(f"Total tags:\t{len(tags)}")
print(f"Extracted tags:\t{tags}")
