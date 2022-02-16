import mysql.connector
import pprint
import pandas as pd
import numpy as np


class UMedLS():

    def __init__(self,
                 host='172.16.34.1',
                 port='3307',
                 user='umls',
                 password='umls',
                 database='umls2020'):
        # connect to the server and save mysql object at attribute
        self.connection = mysql.connector.connect(host=host, port=port, user=user, password=password, database=database)
        self.cursor = self.connection.cursor()

        return

    def to_df(self, query_res):
        df = pd.DataFrame(0, index=np.arange(len(query_res['rows'])), columns=query_res['columns'])
        df.loc[:, :] = query_res['rows']

        return df

    def run_query(self, query, to_pandas=True):
        cur = self.cursor
        cur.execute(query)
        query_results = cur.fetchall()
        col_names = [col[0] for col in cur.description]
        # pprint.pprint(col_names)
        # pprint.pprint(query_result)

        res = {'rows': query_results, 'columns': col_names}
        if to_pandas:
            res = self.to_df(res)

        return res

    def find_disease(self, table, disease):
        query = """
select * 
  from 
    SAMPLE_TABLE
  where 
    STR='DISEASE' 
    and LAT='ENG' 
    and SUPPRESS='N';
    """
        # search
        query = query.replace('DISEASE', disease)
        query = query.replace('SAMPLE_TABLE', table)
        results = self.run_query(query)

        return results

    def find_cui(self, table, cui):
        query = """
select distinct STR 
  from SAMPLE_TABLE
  where 
    CUI='SAMPLE_CUI' 
    and LAT='ENG' 
    and SUPPRESS='N'; 
    """
        # search
        query = query.replace('SAMPLE_CUI', cui)
        query = query.replace('SAMPLE_TABLE', table)
        results = self.run_query(query)

        return results

    def extract_parents(self, table, source_cui):
        query = """
select *
  from SAMPLE_TABLE
  where 
    CUI1='SOURCE_CUI'  
    and REL='PAR'
    """
        # search
        query = query.replace('SAMPLE_CUI', source_cui)
        query = query.replace('SAMPLE_TABLE', table)
        results = self.run_query(query)

        return results

    def extract_children(self, table, source_cui):
        query = """
select *
  from SAMPLE_TABLE
  where 
    CUI1='SOURCE_CUI'  
    and REL='CHD' 
    """
        # search
        query = query.replace('SOURCE_CUI', source_cui)
        query = query.replace('SAMPLE_TABLE', table)
        results = self.run_query(query)

        return results

    def dfs(self, ref_node, cur_node, level):
        if level > 40:
            return ''

        #if cur_node == ref_node:
        #    print(level)
        #    print(3 < level <= 40)


        if (3 < level <= 40) and (cur_node == ref_node):
            return cur_node

        # if above two conditions don't satisfy then explore the graph

        # get child and parent nodes
        children = self.extract_children('MRREL', cur_node).loc[:, 'CUI2'].values.tolist()
        parents = self.extract_parents('MRREL', cur_node).loc[:, 'CUI2'].values.tolist()

        # explore every child node
        while not children == []:
            child = children.pop(0)
            cycle = self.dfs(ref_node, child, level + 1)

            # if we find a cycle then return
            if cycle != '':
                print(cycle)
                return cur_node + ' ' + cycle

        # explore every parent node
        while not parents == []:
            parent = parents.pop(0)
            cycle = self.dfs(ref_node, parent, level+1)

            # if we find a cycle then return
            if cycle != '':
                return cur_node + ' ' + cycle

        return ''


# Test code
umls_obj = UMedLS()

# try searching
#search_d = umls_obj.find_disease('MRCONSO', 'breast cancer')
#search_cui = umls_obj.find_cui('MRCONSO', 'C0678222')
#search_par = umls_obj.extract_parents('MRREL', 'C0006826')
#search_chd = umls_obj.extract_children('MRREL', 'C0006826')

# Test functions
cui = 'C2939428'  # Amoeba genus
path_ = umls_obj.dfs(cui, cui, 0)
print(path_)

# list of starting nodes for checking cycles
queries = []