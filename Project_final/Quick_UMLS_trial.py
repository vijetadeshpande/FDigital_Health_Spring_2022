import spacy
from quickumls import QuickUMLS

#
path_dir = r'/Users/vijetadeshpande/Downloads/BioNLP Lab/Datasets/KG/UMLS/QuickUMLS'

#
matcher = QuickUMLS(quickumls_fp = path_dir)

#
text = "The ulna has dislocated posteriorly from the trochlea of the humerus."
text_processed = matcher.match(text, best_match=True, ignore_syntax=False)


from quickumls.spacy_component import SpacyQuickUMLS

# common English pipeline
nlp = spacy.load('en_core_web_sm')

quickumls_component = SpacyQuickUMLS(nlp, 'PATH_TO_QUICKUMLS_DATA')
nlp.add_pipe(quickumls_component)

doc = nlp('Pt c/o shortness of breath, chest pain, nausea, vomiting, diarrrhea')

for ent in doc.ents:
    print('Entity text : {}'.format(ent.text))
    print('Label (UMLS CUI) : {}'.format(ent.label_))
    print('Similarity : {}'.format(ent._.similarity))
    print('Semtypes : {}'.format(ent._.semtypes))