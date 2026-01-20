from collections import defaultdict
import os
import uuid
import time
import json
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
import subprocess
import tempfile

mml_dir = "C:/mml/metamaplite.bat"

#function to extract annotated entities from control_ds
def extract_problem_entity_spans(tokens, tags):
    spans = []
    current_span = []

    for token, tag in zip(tokens, tags):
        if tag == 1:  # B-PROBLEM
            current_span = [token]
        elif tag == 2:  # I-PROBLEM
            if current_span:
                current_span.append(token)
        elif tag == 3:  # E-PROBLEM
            if current_span:
                current_span.append(token)
                spans.append(" ".join(current_span))
                current_span = []
        elif tag == 4:  # S-PROBLEM (single-token span)
            spans.append(token)
            current_span = []
        else:  # O or any non-PROBLEM tag
            if current_span:
                spans.append(" ".join(current_span))
                current_span = []

    # Catch any unfinished span
    if current_span:
        spans.append(" ".join(current_span))

    return spans

#function to run MetaMapLite on untokenized sentences [all label]
def run_metamap(text, input_path, output_path):
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(text)

    subprocess.run(
        [mml_dir, input_path, "--overwrite"],
        cwd="C:\\mml",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Try reading as latin-1 if utf-8 fails
            with open(output_path, "r", encoding="latin-1") as f:
                return f.read()
    else:
        return None


#function to parse metamap output (.mmi file) for all phrases identified and get semantic type
def extract_metamap_phrases(mmi_text):
    semtype_dict = defaultdict(list)
    for line in mmi_text.strip().split("\n"):
        parts = line.split('|')
        if len(parts) < 7:
            continue
        phrase = parts[3].strip().lower()
        sem_type = parts[5].strip().lower()
        semtype_dict[sem_type].append(phrase)
    
    # remove duplicates within each semantic type
    for key in semtype_dict:
        semtype_dict[key] = list(set(semtype_dict[key]))
    
    return dict(semtype_dict)

def get_matched_phrase_sem_type(annotated_spans, mm_phrases_by_semtype):
    matched = []
    for semtype, phrases in mm_phrases_by_semtype.items():
        for phrase in phrases:
            for tag in annotated_spans:
                if phrase in tag or tag in phrase:
                    matched.append({"phrase": phrase, "semtype": semtype})
                    break  # avoid duplicates
    return matched

#function to process row
def process_row(row):
    tokens = row["sentence"]
    tags = row["tags"]
    text = " ".join(tokens)

    annotated_spans = extract_problem_entity_spans(tokens, tags)
    tmp_id = str(uuid.uuid4())
    input_file = os.path.join(tempfile.gettempdir(), f"{tmp_id}.txt")
    output_file = os.path.join(tempfile.gettempdir(), f"{tmp_id}.mmi")

    mmi_text = run_metamap(text, input_file, output_file)

    if mmi_text:
        mm_phrases_by_semtype = extract_metamap_phrases(mmi_text)
        matched_phrases = get_matched_phrase_sem_type(annotated_spans, mm_phrases_by_semtype)
        y_pred = 1 if matched_phrases else 0
    else:
        matched_phrases = []
        y_pred = 0

    y_true = 1 if annotated_spans else 0

    #clean up
    for f in [input_file, output_file]:
        if os.path.exists(f):
            os.remove(f)

    return {
        "sentence": text,
        "annotated_spans": annotated_spans,
        "matched_metamap": matched_phrases,
        "y_true": y_true,
        "y_pred": y_pred
    }