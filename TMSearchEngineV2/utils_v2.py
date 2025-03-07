import numpy as np
import pandas as pd
import re, json
import os
import itertools
from itertools import groupby
from PyPDF2 import PdfReader
import faiss
import pickle 
from pathlib import Path
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify


def process_manual(file_path):
    '''
    Run this function the first time a tech manual is processed to extract text.
    This process takes approximately 2 minutes. 
    Input: path to PDF, including pdf filename
    Output: JSON file
    '''
    reader = PdfReader(file_path)
    results = {}
    print('Converting PDF to searchable data structure...')
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        results[idx] = text
    print('Conversion complete. Writing to local file...')
    with open('ExtractedManual.json', 'w') as file:
        json.dump(results, file)
    print('Done.')
    return 'PDF extraction complete.'



class Engine:

    def __init__(self, TMdata):
        self.TMdata = TMdata

        '''
        Parse tech manual json
        Does not need to run more than once per occurrance
        '''

        # parse Chapter 3: Troubleshooting data 
        self.tdata = ''
        for page in range(80,93):
            self.tdata = self.tdata + '\n' + self.TMdata[page]

        # split by malfunction, e.g., '1.', '2.'
        # will separate the number from the malfunction then rejoin
        self.tdata = re.split('(\d{,3}\s\.)', self.tdata)
        for idx, item in enumerate(self.tdata):
            if re.search('\d\s\.', item):
                self.tdata[idx] = self.tdata[idx] + self.tdata[idx+1] # put number and value back together
                self.tdata.pop(idx+1) # remove duplicate

        ### add code to provide search results for a more general search; i.e., text that is not in the troubleshooting chapter

        # filter by results in troubleshooting chapter v. other chapters
        self.ntdata = {}
        for page in itertools.chain(range(19,81), range(92, list(self.TMdata.keys())[-1] + 1)):
            self.ntdata[page] = self.TMdata[page]
        
        '''if other chapters, return search results as pg number; parent chapter number; 
            results (full section containing search results including header)
        '''
        # identify the other chapters and the pages they span 
        self.chapters_dict = {}

        ch4_sp = 92
        self.chapters_dict = {'4': {'title': 'GENERAL MAINTENANCE PROCEDURES', 'page span': [ch4_sp]}}

        for page, text in self.ntdata.items():
            chpts = re.findall('[\n]+\d-\d{1,2}(CHAPTER \d[\n].*[\n])' , self.ntdata[page])

            if len(chpts) > 0:
                chpts = chpts[0].strip().split('\n')

                self.chapters_dict[chpts[0][-1]] = {}
                self.chapters_dict[chpts[0][-1]]['title'] = chpts[-1]
                self.chapters_dict[chpts[0][-1]]['page span'] = [page]

                if (chpts[0][-1] != '1') and (chpts[0][-1] != '4'):
                    self.chapters_dict[str(int(chpts[0][-1]) - 1)]['page span'] += [page-1]

                if chpts[0][-1] == '3':
                    self.chapters_dict['3']['page span'] += [ch4_sp-1]

            pat = re.findall('[\n][A-Z]-\d(APPENDIX A)[\n]' , self.ntdata[page])
            if len(pat) > 0:
                self.chapters_dict[max(self.chapters_dict.keys())]['page span'] += [page-1]


        self.chapters_dict = dict(sorted(self.chapters_dict.items()))

        # identify all of the sections within each chapter from the table of contents
        self.tocdata = ''
        for page in list(range(2,9)):
            self.tocdata += '\n' + self.TMdata[page]

        sections = np.unique(np.asarray([x.replace('.', '').strip() for x in re.findall('(\.\s*\d-\d{1,2})' , self.tocdata)]))
        self.section_dict = {}

        for key, group in itertools.groupby(sorted(sections), key=lambda x: x[0]):
            self.section_dict[key] = sorted([int(v.split('-')[-1]) for v in list(group)])

        for ch in self.section_dict.keys():
            chapter = self.section_dict[ch]
            section_sets= [list(y) for i, y in groupby(zip(chapter, chapter[1:]), key = lambda x: (x[1] - x[0]) == 1)]
            longest_set = max(section_sets, key = len)

            sections = []  
            for elem in longest_set:
                sections.append(elem[0])
                sections.append(elem[1])   

            sections = list(set(sections))
            sections.sort()
            self.section_dict[ch] = sections

        last_chapter = max(self.section_dict.keys()) #last section in document before appendix e.g. 7-9
        last_section = max(self.section_dict[last_chapter])

        #consolidate the sections, names, and chapters
        self.full_dict = {} 

        for chapter, sections in self.section_dict.items():

            #get the start and end pages of the given chapter
            page_start = self.chapters_dict[chapter]['page span'][0]
            page_end = self.chapters_dict[chapter]['page span'][1]

            #get the text in the chapter by page
            ch_data = {}
            for page in range(page_start, page_end+1):
                ch_data[page] = self.TMdata[page]

            #find the sections and the associated page ranges
            self.full_dict[chapter] = {'chapter details': self.chapters_dict[chapter]}
            self.full_dict[chapter]['section details'] = {}

            for section in sections:

                soi = chapter + '-' + str(section) #section of index e.g. 1-1, 1-2, etc.
                self.full_dict[chapter]['section details'][soi] = {}

                #check document for the section
                for page, text in self.ntdata.items():
                    pat = re.findall('[\n]+(\d*-\d{0,4}-*\d{0,3}\.)*\s+(.*[A-Z](\.\s\d*)?)\.?[\n]?' , self.ntdata[page])
                    if len(pat) > 0: #a section exists on this page
                        section_number = pat[0][0].replace('.', '').strip()#e.g. 7-4
                        hyphen_index = section_number.rfind('-')
                        s_num = section_number[hyphen_index-1:]

                        if s_num == soi: #section number is the section of interest
                            section_name = pat[0]
                            self.full_dict[chapter]['section details'][soi]['title'] = section_name[1] + section_name[2].strip()
                            self.full_dict[chapter]['section details'][soi]['page span'] = [page]

                            break

                if (str(section) != '1'):
                    self.full_dict[chapter]['section details'][chapter + '-' + str(int(section -1))]['page span'] += [page-1]

                if (int(section) == max(sections)) :
                    self.full_dict[chapter]['section details'][soi]['page span'] += [self.chapters_dict[chapter]['page span'][-1]]        
        

        ######################
        #data for KNN troubleshooting search
        self.ts_res = [] 
        chapter = '3'

        for section in self.full_dict[chapter]['section details'].keys():
            text = ''
            for page in range(self.full_dict[chapter]['section details'][section]['page span'][0], self.full_dict[chapter]['section details'][section]['page span'][1]+1):
                text += self.TMdata[page]
            self.ts_res += [[chapter, self.full_dict[chapter]['chapter details']['title'],section, self.full_dict[chapter]['section details'][section]['title'], str(self.full_dict[chapter]['section details'][section]['page span'][0]) + '-' + str(self.full_dict[chapter]['section details'][section]['page span'][-1]), text]]

        self.ts_tdf = pd.DataFrame(self.ts_res, columns= ['chapter', 'chapter title', 'section', 'section title', 'section pages', 'text'])
        #format dataframe
        self.ts_tdf['section title'] = self.ts_tdf['section title'].replace(to_replace='(\n)',value=' ',regex=True)

        self.ts_tdf.text = self.ts_tdf.text.replace(to_replace=r"\bTM 8A192D-OI\b", value = '', regex=True)
        self.ts_tdf.text = self.ts_tdf.text.replace(to_replace='-',value=' ',regex=True)
        self.ts_tdf.text = self.ts_tdf.text.replace(to_replace='  ',value='',regex=True)                #remove double white space
        self.ts_tdf.text = self.ts_tdf.text.apply(lambda x:x.strip())  # Ltrim and Rtrim of whitespace
        self.ts_tdf.text = self.ts_tdf.text.apply(lambda x:x.lower())

        self.ts_txt = self.ts_tdf.loc[1]['text'].split('table 3 2. troubleshooting\nmalfunction probable cause possible remedy')[1]
        self.ts_txt1 = [x for x in re.split('\n\d{1,2}\s\. ',self.ts_txt) if x != '']



        #set up dictionary to identify page number to malfunction in troubleshooting chapter
        self.matching_solutions_data = {}
        for i in range(len(self.ts_txt1)):
            results = [x for x in re.split('\s(\w\.)|\.(\w\.)', self.ts_txt1[i]) if x != None]

            for idx, result in enumerate(results):
                if re.search('^\w{1}\.', result): # find only the letters
                    results[idx] = results[idx] + results[idx+1] # put bullet and value back together
                    results.pop(idx+1) # remove duplicate
            
            self.all_starting_letters = list(dict.fromkeys([re.findall('(\w\.\s)', x)[0] for x in results if re.findall('(\w\.\s)', x)])) # remove duplicates, too

            # strip newlines & whitespace
            results = [r.strip().replace('\n', ' ') for r in results]
            
            self.matching_solutions_data[i] = {'malfunction': results[0]}

            for result in results:
                for letter in self.all_starting_letters:
                    if re.search('^{}'.format(letter), result):
                        # strip letters from matches
                        result = result.lstrip(letter).strip()
                        if letter not in self.matching_solutions_data[i].keys():
                            self.matching_solutions_data[i][letter] = {'cause': result}
                        else:
                            self.matching_solutions_data[i][letter]['resolution'] = result  
                            
                        # add in references to other sections    
                        pat = r'(\(.+?\))'  
                        refs = re.findall(pat, result) 
                        if len(refs) > 0: # identify if references are present
                            for ref in refs: # find page, section, and text of reference
                                try:
                                    self.matching_solutions_data[i][letter]['references'] = self.id_section(self.ntdata, ref)
                                except:
                                    continue

        #create a dictionary to identify what page the malfunction within the troubleshooting section starts at
        self.malf_dict = {}
        for i in range(80,91):
            nums = re.findall('\n(\d{1,2})\s\. ',self.TMdata[i])
            for num in nums:
                self.malf_dict[int(num)-1] = i


        #numbered malfunction data paired with cause and resolution
        self.ts_temp_df = pd.DataFrame.from_dict(self.matching_solutions_data,orient='index')
        self.ts_temp_df['Index'] = list(range(len(self.ts_temp_df)))

        #KNN Vector search classifier
        model = SentenceTransformer('all-mpnet-base-v2') 

        ################
        #Create embeddings of each of the malfunctions from the troubleshooting section for KNN search of input query
        
        self.ts_sects = self.ts_temp_df['malfunction'] 
        self.ts_embeddings = model.encode(self.ts_sects.to_list(), show_progress_bar=True)
        #calculate and rank embeddings
        self.ts_embeddings = np.array([embedding for embedding in self.ts_embeddings]).astype("float32")

        self.ts_index = faiss.IndexFlatL2(self.ts_embeddings.shape[1])
        self.ts_index = faiss.IndexIDMap(self.ts_index)
        self.ts_index.add_with_ids(self.ts_embeddings, self.ts_temp_df['Index'].values)



        ######################
        #data for general search 
        
        #dataframe to hold all chapters and sections for general search (excludes Troubleshooting chapter)
        self.gs_res = [] 
        for chapter in self.full_dict.keys():
            if chapter != '3':
                for section in self.full_dict[chapter]['section details'].keys():
                    text = ''
                    for page in range(self.full_dict[chapter]['section details'][section]['page span'][0], self.full_dict[chapter]['section details'][section]['page span'][1]+1):
                        text += self.ntdata[page]
                    self.gs_res += [[chapter, self.full_dict[chapter]['chapter details']['title'],section, self.full_dict[chapter]['section details'][section]['title'], str(self.full_dict[chapter]['section details'][section]['page span'][0]) + '-' + str(self.full_dict[chapter]['section details'][section]['page span'][-1]), text]]

        self.gs_tdf = pd.DataFrame(self.gs_res, columns= ['chapter', 'chapter title', 'section', 'section title', 'section pages', 'text'])
        #format dataframe
        self.gs_tdf['section title'] = self.gs_tdf['section title'].replace(to_replace='(\n)',value=' ',regex=True)
        self.gs_tdf.text = self.gs_tdf.text.replace(to_replace=r"\bTM 8A192D-OI\b", value = '', regex=True)
        self.formatted_gs_text = self.gs_tdf.text.replace(to_replace='(\n)',value='\n\n',regex=True) 
        self.gs_tdf.text = self.gs_tdf.text.replace(to_replace='(\n)',value=' ',regex=True) 
        self.gs_tdf.text = self.gs_tdf.text.replace(to_replace='[!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~]',value=' ',regex=True) #remove punctuation except
        self.gs_tdf.text = self.gs_tdf.text.replace(to_replace='-',value=' ',regex=True)
        self.gs_tdf.text = self.gs_tdf.text.replace(to_replace='\s+',value=' ',regex=True)    #remove new line
        self.gs_tdf.text = self.gs_tdf.text.replace(to_replace='  ',value='',regex=True)                #remove double white space
        self.gs_tdf.text = self.gs_tdf.text.apply(lambda x:x.strip())  # Ltrim and Rtrim of whitespace
        self.gs_tdf.text = self.gs_tdf.text.apply(lambda x:x.lower())
        #set dataframe indexes for model purposes
        self.gs_tdf['Index'] = list(range(len(self.gs_tdf)))

        ################
        #Create embeddings of each of the sections for KNN search of input query
        
        #improved results when combining chapter title and section to section text for embeddings
        self.gs_sects = self.gs_tdf['chapter title'] + ' ' + self.gs_tdf['section title'] + ' ' + self.gs_tdf['text']
        self.gs_embeddings = model.encode(self.gs_sects.to_list(), show_progress_bar=True)

        #calculate and rank embeddings
        self.gs_embeddings = np.array([embedding for embedding in self.gs_embeddings]).astype("float32")
        self.gs_index = faiss.IndexFlatL2(self.gs_embeddings.shape[1])
        self.gs_index = faiss.IndexIDMap(self.gs_index)
        self.gs_index.add_with_ids(self.gs_embeddings, self.gs_tdf['Index'].values)
        
    def get_hyperlink(self,page):
        '''
        given a page number, function returns the hyperlink to the manual
        '''
        url_base = 'https://sharedaccount.blob.rack01.usmc5gsmartwarehouse.com/d365/LAV%2025%20Tech%20Manual.pdf?sp=r&st=2023-09-25T18:34:32Z&se=2050-09-26T02:34:32Z&sv=2019-02-02&sr=b&sig=wJ7FeRspNYR3kdHO2X0O4yBolQe%2FsNJJpz3n3Th9%2B3I%3D'
        modified_url = url_base + '#page={}'.format(page)
        
        return modified_url

    def get_chapter_number(self, page):
        '''
        Function returns chapter number + title given a page;
        e.g., input '81' returns '3. Troubleshooting'
        '''
        for ch, v in self.full_dict.items():
            ch_pgs = v['chapter details']['page span']
            if (page >= ch_pgs[0]) and (page <= ch_pgs[1]):
                return ch + '. ' + v['chapter details']['title']

        return 'Invalid page number'
    
    def id_section(self, d, section):
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
            section_match = re.findall('[\n ]+\d-\d{1,2}.*' + section_name + '[\S.][\n]', self.ntdata[page]) 
            
            if len(section_match)>0:
                for res in section_match:
                    page_temp = self.full_dict[section_number[0]]['section details'][section_number]['page span']
                    reference_dict['pages'] += [str(page_temp[0]) + '-' + str(page_temp[1])]
                    #reference_dict['pages'] += [page]
                    reference_dict['chapter'] += ['Chapter ' + str(format_section[0]) + '. ' + self.chapters_dict[str(format_section[0])]['title']]
                    reference_dict['section'] += [format_section]
                    reference_dict['text'] += [value]
                    return reference_dict
        return None

    def search(self):
        
        input_data =  request.json

        # initialize output structure
        output = {}

        search_term = input_data['Query']
        output['Query'] = search_term
        print('Conducting search for query: {}...'.format(search_term), flush=True)
        
        # initialize output length and list of results
        output['ResultCount'] = 0
        output['Results'] = []

        # search through sections of the troubleshooting chapter 
        print('Looking for result in troubleshooting chapter...', flush=True)

        #KNN Vector search classifier
        model = SentenceTransformer('all-mpnet-base-v2') 

        #encode search term
        embed_query= model.encode([search_term])
        embed_query = np.array([eq for eq in embed_query]).astype("float32")

        self.ts_D, self.ts_I = self.ts_index.search(np.array([embed_query][0]), k = 5) #return top 5 result

        ts_threshold = 1
        res_ind = self.ts_I[0][np.where(self.ts_D[0] < ts_threshold)[0]]


        if len(res_ind) > 0:
            fs_res = self.ts_temp_df.iloc[res_ind.tolist()]
            for ri in res_ind:
                title = fs_res.loc[ri]['malfunction'].replace('.','')

                tb_page = self.malf_dict[fs_res.loc[ri]['Index']]

                #add results to output
                res = fs_res.loc[[ri]][fs_res.loc[[ri]].columns[~fs_res.loc[[ri]].isnull().all()]].drop(['Index'], axis=1)

                res_pairs = res[[x for x in res.columns if x != 'malfunction']].to_dict('records')[0]

                # format text for result body
                textbody = ''
                for k1, v1 in res_pairs.items():
                    
                    textbody = textbody + '------ Identified Cause/Remedy ------'
                    textbody = textbody + '\n' + 'Probable cause: {}'.format(v1['cause'])
                    textbody = textbody + '\n' + 'Possible remedy: {}'.format(v1['resolution'])
                    
                    #references
                    s_refs = ['-'.join(x.split(' ')) for x in re.findall('paragraph (\d \d{1,2})', v1['resolution'])]  
                    if len(s_refs) > 0:
                        textbody = textbody + '\n' + 'Additional reference material available below:'
                        for s_ref in s_refs:
                            textbody = textbody + '\n' + 'Details available in chapter {}'.format(': '.join([s_ref[0],self.full_dict[s_ref[0]]['chapter details']['title']] ))
                            textbody = textbody + '\n' + '{}'.format(': '.join([s_ref, self.full_dict[s_ref[0]]['section details'][s_ref]['title'].replace('\n', '').replace('.', '')]))
                            textbody = textbody + '\n' + 'See pages {} for more.'.format(str(self.full_dict[s_ref[0]]['section details'][s_ref]['page span'][0]) + '-' + str(self.full_dict[s_ref[0]]['section details'][s_ref]['page span'][1]))
                            textbody = textbody + '\n' + '---Section Preview---'
                            textbody = textbody + '\n' + self.get_hyperlink(self.full_dict[s_ref[0]]['section details'][s_ref]['page span'][0]) #'{}'.format(ntdata[full_dict[s_ref[0]]['section details'][s_ref]['page span'][0]])
                        
                    textbody = textbody + '\n'

                resultRow = {
                    'ResultID': len(output['Results']) + 1, # number of result - ultimately should be numbered by match confidence
                    'SearchMatch': title,
                    'PDFLocation': tb_page,
                    'PDFURL': self.get_hyperlink(tb_page),
                    'PDFChapter': '3. Troubleshooting',
                    'SearchCategory': 'Troubleshooting',
                    'Result_Heading': 'malfunction: ' + title,
                    'Result_Body': textbody
                }
                output['ResultCount'] += 1
                output['Results'].append(resultRow)

        if not output['Results']:
            print('No troubleshooting results found. Initiating general search...', flush=True)
        elif output['Results']:
            print('{} troubleshooting results found; conducting general search anyways...'.format(len(output['Results'])), flush=True)
            
        '''
        GENERAL SEARCH CODE HERE
        '''
        
        #KNN Vector search classifier
        self.gs_D, self.gs_I = self.gs_index.search(np.array([embed_query][0]), k = 10) #return top 10 results

        gs_threshold = 1.4
        
        res_ind = self.gs_I[0][np.where(self.gs_D[0] < gs_threshold)[0]]
        if len(res_ind) > 0:
            #dataframe of top 10 results
            fs_res = self.gs_tdf.iloc[res_ind.tolist()][['chapter', 'chapter title', 'section', 'section title', 'section pages', 'text']]
            fs_res['section title'] = fs_res['section title'].apply(lambda x: x.replace('.', '').strip())
            
            print('{} general search results found'.format(len(res_ind)), flush=True)
            
            #add results to output
            for ind, val in fs_res.iterrows():
                resultRow = {
                'ResultID': len(output['Results']) + 1, # number of result - ultimately should be numbered by match confidence
                'SearchMatch': 'general search related to ' + search_term,
                'PDFLocation': fs_res.loc[ind]['section pages'],
                'PDFURL': self.get_hyperlink( fs_res.loc[ind]['section pages'].split('-')[0]),
                'PDFChapter': fs_res.loc[ind]['chapter'] + '. ' + fs_res.loc[ind]['chapter title'],
                'SearchCategory':  fs_res.loc[ind]['chapter title'],
                'Result_Heading':  fs_res.loc[ind]['section title'],
                'Result_Body': '------ Identified Section ------' + '\n' + self.formatted_gs_text.loc[ind]
                }
                output['ResultCount'] += 1
                output['Results'].append(resultRow)
        else:
            print('No general results found.', flush=True)

           
        
        return jsonify({"result": output}, 200)  