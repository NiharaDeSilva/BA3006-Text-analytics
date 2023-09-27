# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

# Sample document for extracting NEs from
text = """
Bayern Munich, or FC Bayern, is a German sports club based in Munich, 
Bavaria, Germany. It is best known for its professional football team, 
which plays in the Bundesliga, the top tier of the German football 
league system, and is the most successful club in German football 
history, having won a record 26 national titles and 18 national cups. 
FC Bayern was founded in 1900 by eleven football players led by Franz John. 
Although Bayern won its first national championship in 1932, the club 
was not selected for the Bundesliga at its inception in 1963. The club 
had its period of greatest success in the middle of the 1970s when, 
under the captaincy of Franz Beckenbauer, it won the European Cup three 
times in a row (1974-76). Overall, Bayern has reached ten UEFA Champions 
League finals, most recently winning their fifth title in 2013 as part 
of a continental treble. 
"""

import nltk
from normalization import parse_document
import pandas as pd

# Tokenize sentences using our helper functions in normalization.py
sentences = parse_document(text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]


# POS tag the sentences and use nltk's Named Entity Chunker
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
ne_chunked_sents = [nltk.ne_chunk(tagged) for tagged in tagged_sentences]
# Extract all named entities
named_entities = []
for ne_tagged_sentence in ne_chunked_sents:
    for tagged_tree in ne_tagged_sentence:
        if hasattr(tagged_tree, 'label'):
                entity_name = ' '.join(c[0] for c in tagged_tree.leaves())
                entity_type = tagged_tree.label()
                named_entities.append((entity_name, entity_type))

# Get unique named entities - why do this?
named_entities = list(set(named_entities))
# Store named entities in a (pandas) data frame (for pretty printing)
entity_frame = pd.DataFrame(named_entities, 
                            columns=['Entity Name', 'Entity Type'])

# Display results as a table
print("Entity extraction using NLTK's NE Recognizer:")
print(entity_frame)   
print()





# Repeat the process above for the stanford NE Recognizer (Java program wrapped in NLTK)
from nltk.tag import StanfordNERTagger
import os
# Set java path in environment variables
java_path = r'/usr/bin/java'
os.environ['JAVAHOME'] = java_path

# Load stanford NER by pointing to the pre-trained English model and the code (jar file)
sn = StanfordNERTagger('/Users/arw/Documents/Work/stanford-ner-2014-08-27/classifiers/english.all.3class.distsim.crf.ser.gz',
                       path_to_jar='/Users/arw/Documents/Work/stanford-ner-2014-08-27/stanford-ner.jar')

# First POS-tag the sentences                       
ne_annotated_sentences = [sn.tag(sent) for sent in tokenized_sentences]

# Then extract the named entities
named_entities = []
for sentence in ne_annotated_sentences:
    temp_entity_name = ''
    temp_named_entity = None
    for term, tag in sentence:
        # Get terms with NE tags
        if tag != 'O':
            temp_entity_name = ' '.join([temp_entity_name, term]).strip()
            temp_named_entity = (temp_entity_name, tag)
        else:
            if temp_named_entity:
                named_entities.append(temp_named_entity)
                temp_entity_name = ''
                temp_named_entity = None

# Extract the unique named entities - why?
named_entities = list(set(named_entities))
# Store named entities in a (pandas) data frame as before
entity_frame = pd.DataFrame(named_entities, 
                            columns=['Entity Name', 'Entity Type'])

# Display the results in a table form as before
print("Entity extraction using StanfordNER's NE Recognizer:")
print(entity_frame)

# Compare the results of the 2 NER systems

                      