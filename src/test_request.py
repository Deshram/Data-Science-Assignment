import requests
import os
import json

#TO-DO: Chnage the directory below to directory which contains only sample articles (atleast 20)
files = os.listdir('663_webhose-2015-09-new_20170904095535')

articles = []
for i in files[20:40]:
    with open('663_webhose-2015-09-new_20170904095535/'+i, 'rb') as f:
        file = json.load(f)
        articles.append(file['text'])

response = requests.post("http://127.0.0.1:5000/nc_keyword_extraction", json = {'data':articles})

print(response.json())