import pandas as pd
import spacy
from quickumls import QuickUMLS
from Project_1.UMedLS import UMedLS
from copy import deepcopy
import os

class AhsanSBDH(UMedLS):

    def __init__(self, dir_data = r'/Users/vijetadeshpande/Documents/GitHub/MIMIC-SBDH'):
        UMedLS.__init__(self)

        self.keyword = pd.read_csv(os.path.join(dir_data, 'MIMIC-SBDH-keywords.csv'))
        self.occurrence = pd.read_csv(os.path.join(dir_data, 'MIMIC-SBDH.csv'))

        return

    def get_sbdh_entities(self, text, mimic_row_identifier):

        """
        Here we make use of the previously published dataset, MIMIC-SBDH, that lists SBDH factors annotated for >7k
        MIMIC notes. The published data has two files,
            self.occurrence: binary matrix for presence of SBDH factors in annotated 7k notes
            self.keyword: location of the SBDH factors present in a particular note

        :param text: a medical note
        :param mimic_row_identifier: an identifier to search in the data published by Ahsan et.al.
        :return:
        """

        #
        try:
            kws = self.keyword.loc[self.keyword.loc[:, 'row_id'] == mimic_row_identifier, :]
        except:
            print('row_id not found in the keyword file')
            return []

        #
        entities = []
        for row in kws.index:
            start = int(kws.loc[row, 'start'])
            end = int(kws.loc[row, 'end'])
            ent_type = kws.loc[row, 'sbdh']
            ngram = text[start:end]

            #
            entity = {'start': start,
                      'end': end,
                      'ngram': ngram,
                      'term': None,
                      'cui': None,
                      'similarity': None,
                      'entity_type': ent_type,
                      }
            entities.append(deepcopy(entity))

        return entities

    def umls_search(self, term, table = 'MRCONSO'):

        res = self.search_term(table, term)

        return res

    def link_entities(self, text, entities, reference_set = []):

        """
        :param text: medical record text
        :param entities: entities for which we need to find cui
        :param reference_set: already found entities in 'text' via quickUMLS
        :return: entities (with added keys for cuis and other stuff)
        """

        # collect all terms identified by quickUMLS
        terms_found_by_quickumls, ngram_to_idx = [], {}
        for idx, ent in enumerate(reference_set):
            terms_found_by_quickumls.append(ent['ngram'])
            ngram_to_idx[ent['ngram']] = idx

        #
        for idx, ent in enumerate(entities):
            if not ent['ngram'] in terms_found_by_quickumls:
                #TODO: current UMLS class code does not handle approximate search facility. It needs exact match.
                #res = self.umls_search('MRCONSO', ent['ngram'])
                #ent['cui'] = res['cui']
                #ent['term'] = None
                #ent['similarity'] = res['similarity']
                pass
            else:
                ent['cui'] = reference_set[ngram_to_idx[ent['ngram']]]['cui']
                ent['term'] = reference_set[ngram_to_idx[ent['ngram']]]['term']
                ent['similarity'] = reference_set[ngram_to_idx[ent['ngram']]]['similarity']

                # pop from the reference set to avoid repetition
                reference_set.pop(ngram_to_idx[ent['ngram']])

        return entities, reference_set


class QuickUMLSPass(AhsanSBDH):

    def __init__(self, dir_quickumls = r'/Users/vijetadeshpande/Downloads/BioNLP Lab/Datasets/KG/UMLS/QuickUMLS'):
        AhsanSBDH.__init__(self)

        self.matcher = QuickUMLS(quickumls_fp = dir_quickumls)

    def get_umls_entities(self, text):

        entities = self.matcher.match(text, best_match=True, ignore_syntax=False)

        return entities

    def get_top_match(self, matches):

        match_top = {}
        sim = 0
        for match in matches:
            if match['similarity'] == 1:
                return match

            if match['similarity'] > sim:
                match_top = match
                sim = match['similarity']

        return match_top

    def reduce_matches(self, ents_in):

        ents_out = []
        for ent in ents_in:
            top_match = self.get_top_match(ent)
            ents_out.append(top_match)

        return ents_out

    def get_entities(self, text, mimic_row_identifier):

        # get all type of entities in the text
        ents_ = self.get_umls_entities(text)
        ents_ = self.reduce_matches(ents_)
        ents_sbdh = self.get_sbdh_entities(text, mimic_row_identifier)

        # link all entities in the UMLS
        ents_sbdh, ents_ = self.link_entities(text, ents_sbdh, ents_)

        # merge both entities
        #entities = ents_ + ents_sbdh
        entities = {'sbdh': ents_sbdh, 'umls': ents_}

        return entities

    def annotate_document(self, text, mimic_row_identifier):

        # find entities and their locaiton
        entities = self.get_entities(text, mimic_row_identifier)

        return {'text': text, 'entities': entities}


#region Test the code

#data = pd.read_csv(r'/Users/vijetadeshpande/Downloads/BioNLP Lab/Datasets/EHR/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv')
#print(data.columns)
#print(data.head())

#record = data.loc[data.loc[:, 'ROW_ID'] == 5, 'TEXT']
text = 'Admission Date:  [**2190-5-16**]     Discharge Date:  [**2190-5-22**]\n\nDate of Birth:   [**2139-4-22**]     Sex:  F\n\nService:  CARDIOTHORACIC\n\nHISTORY OF PRESENT ILLNESS:  This 51 year-old female was\nadmitted to an outside hospital with chest pain and ruled in\nfor myocardial infarction.  She was transferred here for a\ncardiac catheterization.\n\nPAST MEDICAL HISTORY:  Hypertension, fibromyalgia,\nhypothyroidism, NASH and noninsulin dependent diabetes.\n\nPAST SURGICAL HISTORY:  Hysterectomy and cholecystectomy.\n\nSOCIAL HISTORY:  She smokes a pack per day.\n\nMEDICATIONS ON ADMISSION:  Hydrochlorothiazide, Alprazolam,\nUrsodiol and Levoxyl.\n\nShe was hospitalized with Aggrastat, nitroglycerin and\nheparin as she ruled in for myocardial infarction.\n\nALLERGIES:  No known drug allergies.\n\nCardiac catheterization showed left anterior descending\ncoronary artery diagonal 80% lesion, circumflex 90% lesion\nand 90% lesion of the right coronary artery with a normal\nejection fraction.  She was transferred from [**Hospital3 68**]\nto [**Hospital1 69**] for cardiac\ncatheterization.  The results as above.  After\ncatheterization she was referred to cardiothoracic surgery\nand was seen by Dr. [**First Name8 (NamePattern2) **] [**Last Name (NamePattern1) 70**] and Dr. [**First Name4 (NamePattern1) 71**] [**Last Name (NamePattern1) 72**].\nPreoperative laboratories showed a sodium of 141, K 4.2,\nchloride 105, CO2 24, BUN 12, creatinine 0.6 with a blood\nsugar of 156.  White count 8.9, hematocrit 44.2, platelet\ncount 201,000.  PT 13, PTT 26 with an INR of 1.2.  CK was\n1511 on [**5-16**].  She was also followed by Dr. [**Last Name (STitle) 73**] of\ncardiology and agreed to participate in both the Cariporide\nand Dermabond studies through cardiac surgery.  The patient\nwas taken to the Operating Room on [**5-18**] and underwent\ncoronary artery bypass grafting times four with a left\ninternal mammary coronary artery to the left anterior\ndescending coronary artery, saphenous vein graft to right\nposterior descending coronary artery, saphenous vein graft to\ndiagonal two and a saphenous vein graft to the obtuse\nmarginal by Dr. [**Last Name (STitle) 70**].\n\nThe patient was transferred to the Cardiothoracic Intensive\nCare Unit in stable condition.  On postoperative day number\none there were no events overnight. The patient was\nextubated and was on a neo-synephrine drip at 0.3 micrograms\nper kilo per minute with the Cariporide infusing.\nNitroglycerin had been turned off.  Postoperative hematocrit\nwas 30 with a K of 4.2 and a blood sugar of 139.  CPK trended\ndown to 357 and 379 with an MB of 15 to 16.  The patient was\nin sinus rhythm in the 80s with a stable blood pressure.  She\nwas alert and oriented.  Her lungs were clear bilaterally.\nHeart was regular rate and rhythm.  Her abdomen was benign.\nHer extremities were within normal limits.  She was\nneurologically stable.  Her chest tubes were pulled on\npostoperative day number three.  She continued on\nperioperative antibiotics and was transferred out to the\nfloor.\n\nShe was seen by physical therapy for evaluation.  On\npostoperative day two she had no events overnight.  She had a\ntemperature max of 100.6.  Her JP drain from her leg site was\nremoved as was her Foley.  Her Lopresor was increased to 50\nb.i.d.  She began to ambulate and was out of bed.  She had\ndecreased at the bases, but was otherwise hemodynamically\nstable.  Her dressings were clean, dry and intact.  She was\nseen by case management to determine the need for rehab.  Her\npacing wires were discontinued on postoperative day three.\nShe continued to advance her ambulation.  She had decreased\nbreath sounds a the bases again on postoperative day three,\nbut was stable and continuing to increase her physical\ntherapy.  Her incision was were clean, dry and intact.  Pain\nwas managed with Percocet and Motrin.  She was sating 92% on\nroom air on postoperative day number four the day of\ndischarge with a temperature max of 99.3, blood pressure\n136/71, heart rate 93.  She was alert, oriented and had been\nambulating well.  Her lungs were clear bilaterally.  Her\nexamination was otherwise benign.\n\nHer laboratories on the 9th showed a white count of 13.6,\nhematocrit 28.7, platelet count 153,000, BUN 15, creatinine\n0.5, sodium 141, glucose 100, K 3.8, magnesium 1.7 for which\nshe received 2 grams of repletion.  Calcium 1.08 for which\nshe received 2 grams of repletion.  She was discharged to\nhome on postoperative day four [**5-22**].\n\nDISCHARGE MEDICATIONS:  Lasix 20 mg po q.d. times one week,\nK-Ciel 20 milliequivalents po q day times one week.  Colace\n100 mg po q.d., Zantac 150 mg po b.i.d., enteric coated\naspirin 325 mg po q day, Levoxyl 0.25 mg po q day, Lopressor\n75 mg po b.i.d., Nicoderm 14 patch q.d., Xanax 2 mg q 4 to 6\nhours prn, Ursodiol dosage not specified.  The patient was\ninstructed to return to preoperative dose.  Percocet one to\ntwo tabs po prn q 4 to 6 hours.\n\nThe patient was afebrile.  Incisions were healing well.\n\nDISCHARGE DIAGNOSES:\n1.  Hypertension.\n2.  Status post coronary artery bypass grafting times four.\n3.  Fibromyalgia.\n4.  Hypothyroidism.\n5.  Noninsulin dependent diabetes mellitus.\n6.  Question NASH.\n\nShe was also instructed to follow up with her primary care\nphysician [**Last Name (NamePattern4) **]. [**Last Name (STitle) 74**] in two weeks and follow up with Dr.\n[**Last Name (STitle) 70**] in the office in six weeks for postop follow up.\nAgain, the patient was discharged home on [**2190-5-22**].\n\n\n\n\n\n\n\n\n                          [**First Name11 (Name Pattern1) **] [**Initials (NamePattern4) **] [**Last Name (NamePattern4) **], M.D.  [**MD Number(1) 75**]\n\nDictated By:[**Last Name (NamePattern1) 76**]\n\nMEDQUIST36\n\nD:  [**2190-7-7**]  08:16\nT:  [**2190-7-7**]  11:56\nJOB#:  [**Job Number 77**]\n'#record.values[0]
qu_pass = QuickUMLSPass()
all_ents = qu_pass.annotate_document(text, 5)

#endregion