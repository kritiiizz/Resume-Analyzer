Resume Analyzer - Task A preprocessing

Use Python 3.11. Extract the ZIP and run these commands in the code folder.

python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_pipeline.py --source ../raw/Resume.csv --out task_a_data --handoff

This writes raw_no_header.csv, light.csv, classical.csv, stemmed.csv and
lemmatized.csv under processed/, each with ID, Category, split and text.
The split/ folder holds train.csv, val.csv and test.csv with ID and Category.
Every processed text file retains all source IDs.

run_pipeline.py is the entry point. The preprocessing rules are in
src/preprocess.py. The split and duplicate handling is in src/splits.py.

Light cleaning repairs known text-encoding artifacts and recognizes
standard, explicitly labeled, and some international phone formats.

Use --skip-lemmatization to run without the spaCy English model and omit
lemmatized.csv. For the detailed audit outputs, run without --handoff.
