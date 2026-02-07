# Clinical NLP: Bridging the Patient-Provider Linguistic Gap
## Project Overview
This repository documents the steps taken to produce the paper exploring the semantic and linguistic disparities between formal medical literature (PubMed Central) and informal patient narratives (Reddit). By developing a custom Named Entity Recognition (NER) and normalization pipeline, this project quantifies how patients describe symptoms compared to how they are documented in clinical research.

The codes were used as a part of a Master’s thesis in Data Science and Business Analytics, focusing on Pancreatic Cancer symptoms—a domain where early symptom detection is critical for survival rates. The paper can be found [here](https://www.academia.edu/146227609/Bridging_the_Gap_Extracting_and_Analyzing_Symptom_Entities_in_Online_Health_Forums_and_Medical_Research_Literature?source=swp_share). 

🚧 **In Active Development** 
Notebooks 6 is still undergoing internal update and cleanup, please do not try to run it. Automated scripts are still in the process of cleanup and will be available soon.

### Tech Stack

- Models: BioBERT, BioMed-RoBERTa (Fine-tuned via Hugging Face)

- Frameworks: spaCy, Scikit-learn (TF-IDF, K-Means), LDA, HuggingFace

- Data Ingestion: PRAW (Reddit API), NCBI Entrez (PubMed API), HuggingFace Dataset

- Medical Ontology: UMLS Metathesaurus (Mapping to Concept Unique Identifiers - CUIs)

- Database: MongoDB Atlas (NoSQL storage for extracted entities)

- Environment: Google Colab Pro+ (GPU Accelerated)

### Methodology & Pipeline
- Corpus Acquisition: Automated extraction of 8,000+ Reddit submissions and 250+ biomedical articles, creating a 600,000-token cross-genre corpus.

- NER Modeling: Evaluated three frameworks (spaCy, BioBERT, and BioMed-RoBERTa). BioMed-RoBERTa emerged as the top performer with an F1-score of 0.81.

- Entity Normalization: Leveraged MetaMapLite and the UMLS API to normalize informal language (e.g., "yellow eyes") to formal medical concepts (e.g., Jaundice).

- Statistical Analysis: Performed Chi-Square tests for homogeneity to validate that the symptoms identified in patient forums significantly differ in context and frequency from clinical literature.

<img width="1277" height="524" alt="image" src="https://github.com/user-attachments/assets/895313f3-d523-423a-85bf-5d9c14262f8f" />


### Key Research Findings
- Symptom Nuance: Patients focus heavily on the quality of life and sensory aspects of symptoms, whereas literature focuses on pathophysiological markers.

- Communication Gap: Identified a 13% disparity in the "Goodness of Fit" between datasets, suggesting that clinical diagnostic criteria may miss early-stage symptoms frequently discussed in peer-to-peer forums.

- Clustering: Unsupervised K-Means clustering successfully grouped symptoms into clinical "themes," aiding in the identification of phantom or under-reported symptoms.

### Repository Structure
- notebooks/: Contains the Jupyter notebooks for data cleaning, model training, and UMLS mapping.

- data_samples/: Sample CSVs of normalized entities and CUI mappings.

- src/: Utility scripts for API connections (Reddit/NCBI).

**How to Run**
[Not advisible for now as the repo is undergoing internal updates] 
- Secrets: Set up ENTREZ_EMAIL, UMLS_API_KEY, MONGODB URI in the Google Colab Secrets manager or in .env file (see .env.examples for format and requires keys)
- Execution: Open the notebooks in Google Colab to leverage GPU acceleration for the transformer models, or locally if GPU is available.
- Dependencies: Run pip install -r requirements.txt to install dependencies
