import numpy as np
import pandas as pd
import re, json, nltk
import os
import itertools
from itertools import groupby
from flask import Flask, request, jsonify

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
nltk.download('punkt')

# initial setup when tech manual files are deployed the first time
if 'ExtractedManual.json' not in os.listdir('/app'):
    print('New text manual. Extracting text...', flush=True)
    from processTM import process_manual

    tm_file = 'LAV 25 Tech Manual.pdf'
    process_manual(tm_file)
else:
    print("None", flush=True)
    None

# import json with extracted tech manual 
with open('ExtractedManual.json', 'r') as file:
    TMdata = json.load(file)
    
# convert keys to numbers
TMdata = {int(k):v for k, v in TMdata.items()}

'''
Parse tech manual json
Does not need to run more than once per occurrance
'''
# parse Chapter 3: Troubleshooting data 
tdata = ''
for page in range(80,93):
    tdata = tdata + '\n' + TMdata[page]

# split by malfunction, e.g., '1.', '2.'
# will separate the number from the malfunction then rejoin
tdata = re.split('(\d{,3}\s\.)', tdata)
for idx, item in enumerate(tdata):
    if re.search('\d\s\.', item):
        tdata[idx] = tdata[idx] + tdata[idx+1] # put number and value back together
        tdata.pop(idx+1) # remove duplicate

### add code to provide search results for a more general search; i.e., text that is not in the troubleshooting chapter

# filter by results in troubleshooting chapter v. other chapters
ntdata = {}
for page in itertools.chain(range(19,81), range(92, list(TMdata.keys())[-1] + 1)):
    ntdata[page] = TMdata[page]
'''if other chapters, return search results as pg number; parent chapter number; 
    results (full section containing search results including header)
'''
# identify the other chapters and the pages they span 
chapters_dict = {}

ch4_sp = 92
chapters_dict = {'4': {'title': 'GENERAL MAINTENANCE PROCEDURES', 'page span': [ch4_sp]}}

for page, text in ntdata.items():
    chpts = re.findall('[\n]+\d-\d{1,2}(CHAPTER \d[\n].*[\n])' , ntdata[page])

    if len(chpts) > 0:
        chpts = chpts[0].strip().split('\n')

        chapters_dict[chpts[0][-1]] = {}
        chapters_dict[chpts[0][-1]]['title'] = chpts[-1]
        chapters_dict[chpts[0][-1]]['page span'] = [page]

        if (chpts[0][-1] != '1') and (chpts[0][-1] != '4'):
            chapters_dict[str(int(chpts[0][-1]) - 1)]['page span'] += [page-1]

        if chpts[0][-1] == '3':
            chapters_dict['3']['page span'] += [ch4_sp-1]

    pat = re.findall('[\n][A-Z]-\d(APPENDIX A)[\n]' , ntdata[page])
    if len(pat) > 0:
        chapters_dict[max(chapters_dict.keys())]['page span'] += [page-1]


chapters_dict = dict(sorted(chapters_dict.items()))

# identify all of the sections within each chapter from the table of contents
tocdata = ''
for page in list(range(2,9)):
    tocdata += '\n' + TMdata[page]

sections = np.unique(np.asarray([x.replace('.', '').strip() for x in re.findall('(\.\s*\d-\d{1,2})' , tocdata)]))
section_dict = {}

for key, group in itertools.groupby(sorted(sections), key=lambda x: x[0]):
    section_dict[key] = sorted([int(v.split('-')[-1]) for v in list(group)])

for ch in section_dict.keys():
    chapter = section_dict[ch]
    section_sets= [list(y) for i, y in groupby(zip(chapter, chapter[1:]), key = lambda x: (x[1] - x[0]) == 1)]
    longest_set = max(section_sets, key = len)

    sections = []  
    for elem in longest_set:
        sections.append(elem[0])
        sections.append(elem[1])   

    sections = list(set(sections))
    sections.sort()
    section_dict[ch] = sections

last_chapter = max(section_dict.keys()) #last section in document before appendix e.g. 7-9
last_section = max(section_dict[last_chapter])

#find the sections, names, and chapters
full_dict = {} 

for chapter, sections in section_dict.items():

    #get the start and end pages of the given chapter
    page_start = chapters_dict[chapter]['page span'][0]
    page_end = chapters_dict[chapter]['page span'][1]

    #get the text in the chapter by page
    ch_data = {}
    for page in range(page_start, page_end+1):
        ch_data[page] = TMdata[page]

    #find the sections and the associated page ranges
    full_dict[chapter] = {'chapter details': chapters_dict[chapter]}
    full_dict[chapter]['section details'] = {}

    for section in sections:

        soi = chapter + '-' + str(section) #section of index e.g. 1-1, 1-2, etc.
        full_dict[chapter]['section details'][soi] = {}

        #check document for the section
        for page, text in ntdata.items():
            pat = re.findall('[\n]+(\d*-\d{0,4}-*\d{0,3}\.)*\s+(.*[A-Z](\.\s\d*)?)\.?[\n]?' , ntdata[page])
            if len(pat) > 0: #a section exists on this page
                section_number = pat[0][0].replace('.', '').strip()#e.g. 7-4
                hyphen_index = section_number.rfind('-')
                s_num = section_number[hyphen_index-1:]

                if s_num == soi: #section number is the section of interest
                    section_name = pat[0]
                    full_dict[chapter]['section details'][soi]['title'] = section_name[1] + section_name[2].strip()
                    full_dict[chapter]['section details'][soi]['page span'] = [page]

                    break

        if (str(section) != '1'):
            full_dict[chapter]['section details'][chapter + '-' + str(int(section -1))]['page span'] += [page-1]

        if (int(section) == max(sections)) :
            full_dict[chapter]['section details'][soi]['page span'] += [chapters_dict[chapter]['page span'][-1]]        


def id_section(d, section):
    '''
        inputs: 
            d - dictionary whose keys are page numbers and values is the text on the page
            section - reference that was listed in the troubleshooting chapter (e.g. "see Paragraph xx.x")
        returns:
            page, section and text associated with the references listed in the troubleshooting chapter
        
    '''
    reference_dict = {'pages': [], 
                      'chapter': [], 
                      'section': [], 
                      'text': []}
    for page, value in d.items():
        #e.g. 7-4 Engine Tuneup
        format_section = section.replace('and', ';').replace('or', ';').replace('(see Paragraph', '').replace(')', '').strip().split(';')[0] 
        section_number = format_section.split(" ")[0]
        #removes the section number e.g. Engine Tuneup
        section_name = ''.join(x + ' ' for x in format_section.split() if x.isalpha()).strip() 
        
        #match the format \n section \n
        section_match = re.findall('[\n ]+\d-\d{1,2}.*' + section_name + '[\S.][\n]', ntdata[page]) 
        
        if len(section_match)>0:
            for res in section_match:
                page_temp = full_dict[section_number[0]]['section details'][section_number]['page span']
                reference_dict['pages'] += [str(page_temp[0]) + '-' + str(page_temp[1])]
                #reference_dict['pages'] += [page]
                reference_dict['chapter'] += ['Chapter ' + str(format_section[0]) + '. ' + chapters_dict[str(format_section[0])]['title']]
                reference_dict['section'] += [format_section]
                reference_dict['text'] += [value]
                return reference_dict
    return None

app = Flask(__name__)

@app.route('/search_engine', methods=['POST'])
def searchEngine():
    
    # possibly may need to uncomment line below during deployment
    input_data = request.json

    # initialize output structure
    output = {}

    search_term = input_data['Query']
    output['Query'] = search_term
    print('Conducting search for query: {}...'.format(search_term), flush=True)
    
    # initialize output length and list of results
    output['ResultCount'] = 0
    output['Results'] = []
    
    # stem search term
    search_word = word_tokenize(search_term) # Tokenize by Word
    stemmed_search_word = ' '.join([stemmer.stem(word) for word in search_word])
    print('Stemming complete: {}'.format(stemmed_search_word), flush=True)

    # search through sections of the troubleshooting chapter 
    print('Looking for result in troubleshooting chapter...', flush=True)
    for section in tdata:

        tok_sentence = word_tokenize(section)
        
        # stem entire section of the chapter
        stemmed_sentence = ' '.join([stemmer.stem(word) for word in tok_sentence])
        
        # search through stemmed section for result
        search_results = re.search(stemmed_search_word, stemmed_sentence, flags=re.IGNORECASE)
        
        if search_results: # if a result is found in the stemmed section, parse that section
            
            # return first page where result is found
            result_page = None
            for page in range(80,93): # range of pages for the troubleshooting chapter
                # search for term in 
                if re.search('smoke', TMdata[page], flags=re.IGNORECASE):
                    result_page = page
                    break
            
            # parse the issues and resolutions in the section
            # this creates a list
            results = re.split('\s(\w\.)|\.(\w\.)', section)

            # remove null pieces created bc of the split
            results = [r for r in results if r != None]
            for idx, result in enumerate(results):
                if re.search('^\w{1}\.', result): # find only the letters
                    results[idx] = results[idx] + results[idx+1] # put bullet and value back together
                    results.pop(idx+1) # remove duplicate
            
            # strip newlines & whitespace
            results = [r.strip().replace('\n', ' ') for r in results]
            
            # pair up issues + resolutions
            all_starting_letters = list(dict.fromkeys([re.findall('(\w\.)', x)[0] for x in results if re.findall('(\w\.)', x)])) # remove duplicates, too
            matching_solutions_data = {} # store matches
            
            title = re.split('\d\s\.', results[0])[1].strip()
            
            specific_match = ''
            section_match_count = 0
            for idx, result in enumerate(results):
                # find specific sentence matching search term
                if re.search(stemmed_search_word, result, flags=re.IGNORECASE):
                    # if multiple matches, return the first one
                    if not specific_match:
                        specific_match = result
                        section_match_count += 1
                    else:
                        section_match_count += 1
                        
                if re.search('^\w{,2}\.', result):
                    # if starts with a word character or two...(i.e., a., b.)
                    # match values with the same starting letter
                    for letter in all_starting_letters:
                        if re.search('^{}'.format(letter), result):
                            # strip letters from matches
                            result = result.lstrip(letter).strip()
                            if letter not in matching_solutions_data.keys():
                                matching_solutions_data[letter] = {'cause': result}
                            else:
                                matching_solutions_data[letter]['resolution'] = result  
                                
                            # add in references to other sections    
                            pat = r'(\(.+?\))'  
                            refs = re.findall(pat, result) 
                            if len(refs) > 0: # identify if references are present
                                for ref in refs: # find page, section, and text of reference
                                    try:
                                        matching_solutions_data[letter]['references'] = id_section(ntdata, ref)
                                    except:
                                        continue
                                        
            if section_match_count > 1:
                specific_match = specific_match + '... and {} additional matches within this section.'.format(section_match_count)
                        
            print('Result found in section: {}'.format(title), flush=True)
        
            # format text for result body
            textbody = ''
            for k1, v1 in matching_solutions_data.items():
                textbody = textbody + '------ Identified Cause/Remedy ------'
                textbody = textbody + '\n' + 'Probable cause: {}'.format(v1['cause'])
                textbody = textbody + '\n' + 'Possible remedy: {}'.format(v1['resolution'])
                if 'references' in v1.keys() and v1['references']:
                    textbody = textbody + '\n' + 'Additional reference material available below:'
                    textbody = textbody + '\n' + 'Details available in {}'.format(''.join(v1['references']['chapter']))
                    textbody = textbody + '\n' + '{}'.format(''.join(v1['references']['section']))
                    textbody = textbody + '\n' + 'See pages {} for more.'.format(''.join(v1['references']['pages']))
                    textbody = textbody + '\n' + '---'
                    textbody = textbody + '\n' + '{}'.format(''.join(v1['references']['text']))
                textbody = textbody + '\n'
            
            resultRow = {
                'ResultID': len(output['Results']) + 1, # number of result - ultimately should be numbered by match confidence
                'SearchMatch': specific_match,
                'PDFLocation': result_page,
                'PDFURL': 'coming-soon',
                'PDFChapter': '3. Troubleshooting',
                'SearchCategory': 'Troubleshooting',
                'Result_Heading': title,
                'Result_Body': textbody
            }
            output['ResultCount'] += 1
            output['Results'].append(resultRow)
    
    if not output['Results']:
        print('No troubleshooting results found. Initiating general search...', flush=True)
    elif output['Results']:
        print('{} troubleshooting results found; conducting general search anyways...'.format(len(output['Results'])), flush=True)
    
    '''
    GENERAL SEARCH CODE
    '''

    return jsonify({"result": output}, 200)  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7050, debug=True)