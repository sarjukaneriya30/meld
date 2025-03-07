import utils_v2
from utils_v2 import process_manual
import os, json
from flask import Flask


# initial setup when tech manual files are deployed the first time
if 'ExtractedManual.json' not in os.listdir('./'):
    print('New text manual. Extracting text...', flush=True)
    from processTM import process_manual

    tm_file = 'LAV 25 Tech Manual.pdf'
    process_manual(tm_file)
else:      
    None

# import json with extracted tech manual 
with open('ExtractedManual.json', 'r') as file:
    TMdata = json.load(file)
    
# convert keys to numbers
TMdata = {int(k):v for k, v in TMdata.items()}

#establish search session
searchSession = utils_v2.Engine(TMdata)

app = Flask(__name__)

@app.route('/search_engine', methods=['POST'])
def run_search():
    return searchSession.search()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7055, debug=True)
    