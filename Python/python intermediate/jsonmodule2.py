import json


with open('usadata.json','r') as f:
    data=json.load(f)
    for state in data['states']:
        print(state['name'],state['population'])
        del state['population']
with open('usadata2.json','w') as nf:
     json.dump(data,nf,indent=2)
