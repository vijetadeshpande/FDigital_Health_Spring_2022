# Project-1: Identifying hierarchical circles in UMLS

To run the final code, please exceute following in the terminal
python main.py


The main.py file calls
	1. UMedLS: a class written to perform following operations
		1. connect to server
		2. find CUI in UMLS
		3. find a concept in UMLS
		4. find parents of a CUI
		5. find children of a CUI
		6. find hierarchical circle corresponding to a starting CUI node

In addition, the directory has a notebook interface in the file detect_cycles.ipynb