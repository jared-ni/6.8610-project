import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from huggingface_hub import login
import spacy
from deep_translator import GoogleTranslator
from datasets import load_dataset
import pandas as pd
import csv
from itertools import combinations
import jieba
import sacrebleu
from tqdm import tqdm

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
if not openai_api_key or not huggingface_api_key:
    raise ValueError("API_KEYs are not set in the .env file.")

# Extract entities from text
def extract_entities(nlp, text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

# Translate entities to a target language
def translate_entities(entities, target_lang):
    translations = [GoogleTranslator(source='auto', target=target_lang).translate(entity) for entity in entities]
    return translations

# # Load datasets into pandas DataFrames
llama_token = huggingface_api_key.strip()
login(llama_token)
def load_law_dataset():
    ds = load_dataset("casehold/casehold", "all")
    train_df = pd.DataFrame(ds['train'])
    test_df = pd.DataFrame(ds['test'])
    validation_df = pd.DataFrame(ds['validation'])
    law_dataset = pd.concat([train_df, test_df, validation_df], ignore_index=True)['citing_prompt']
    return law_dataset

def load_medical_dataset():
    ds = load_dataset("zhengyun21/PMC-Patients")
    train_df = pd.DataFrame(ds['train'])
    medical_dataset = train_df['patient']
    return medical_dataset

# Combine datasets
def load_all_datasets():
    law_dataset = load_law_dataset()
    medical_dataset = load_medical_dataset()
    return [law_dataset, medical_dataset]

# Calculate JTC score
def calculate_JTC(translations, text, entities):
    jtc_score = 0
    n = len(entities)
    entity_counts = 0

    for entity, translated_entity in entities.items():
        # Count occurrences of the entity in the original text
        c = text.count(entity)
        entity_counts += c
        if c == 0 or not translated_entity:
            continue  # Skip entities that are not present or have no translation
        jargon_inconsistency = K_HYPERPARAMETER * c

        for translation in translations:
            # Count occurrences of the translated entity in the translation
            t = translation.count(translated_entity)
            # Calculate penalty for mismatched occurrences
            jargon_inconsistency -= t
        
        # Update the JTC score
        jtc_score += jargon_inconsistency

    # Normalize and invert the score
    normalized_score = jtc_score / max(K_HYPERPARAMETER * entity_counts, 1)
    return 1 - normalized_score

# calculates the Jaccard similarity between the k translations
def calculate_jaccard(translations, target_lang):
    jaccard_scores = []
    if target_lang == "Simplified Chinese":
        translations = [set(jieba.lcut(translation)) for translation in translations]
    else:
        translations = [set(translation.split()) for translation in translations]
    for t1, t2 in combinations(translations, 2):
        # Calculate Jaccard similarity for the pair
        intersection = len(t1.intersection(t2))
        union = len(t1.union(t2))
        jaccard_score = intersection / union if union > 0 else 0
        jaccard_scores.append(jaccard_score)
    # Return the average Jaccard similarity
    return sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0

# chrF++ ⇒ google translate serves as a reference ground truth
def calculate_chrf(ground_truth_translation, translations, n_value=6):
    ground_truth_translations = [ground_truth_translation] * len(translations)
    # sacrebleu.corpus_chrf calculates chrF++ directly
    chrf = sacrebleu.corpus_chrf(
        translations,  # List of translated texts
        ground_truth_translations,  # List of reference texts
        beta=2  # Default beta for F-score weighting
    )
    return chrf.score


# Initialize the models using GPT-4o-mini and spacy
llm_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
spacy_model = spacy.load('../en_core_sci_sm-0.5.4/en_core_sci_sm/en_core_sci_sm-0.5.4')
K_HYPERPARAMETER = 3


def run_pipeline(target_lang, dataset_index=0, results_file="results.csv"):
    datasets = load_all_datasets()
    # Google Translate abbreviations
    lang_abbrs = {"Simplified Chinese": "zh-CN", "French": "fr"}
    
    cur_dataset = [datasets[dataset_index]]
    # law[0] and medical[1] datasets
    for dataset in cur_dataset:
        progress_bar = tqdm(dataset[:500], desc=f"Processing {dataset_index}", unit="text")
        for text in progress_bar:
            text = " ".join(text.split()[:50])  # Truncate to the first 30 words
            named_entities = list(set(extract_entities(spacy_model, text)))
            # Translate named entities to the target language
            named_entities_translations = translate_entities(named_entities, lang_abbrs[target_lang])
            named_entity_mapping = {e: t for e, t in zip(named_entities, 
                                                         named_entities_translations)}
            ground_truth_translation = GoogleTranslator(source='auto', target=lang_abbrs[target_lang]).translate(text)


            # regular text translation
            regular_translations = []
            for k in range(K_HYPERPARAMETER):
                prompt = f"Please return only the answer and nothing else.\
                           Translate the following text to {target_lang}: {text}."
                response = llm_model.invoke(prompt)
                regular_translations.append(response.content)
            
            # LEAP text translation
            leap_translations = []
            for k in range(K_HYPERPARAMETER):
                prompt = f"Please return only the answer and nothing else.\
                    Translate the following text to {target_lang}\
                    using these mappings {str(named_entities_translations)}: {text}."
                response = llm_model.invoke(prompt)
                leap_translations.append(response.content)
            
            # JTC Metric
            regular_jtc_score = calculate_JTC(regular_translations, text,
                                              named_entity_mapping)
            leap_jtc_score = calculate_JTC(leap_translations, text, 
                                           named_entity_mapping)
        
            # Jaccard Similarities
            regular_jaccard_score = calculate_jaccard(regular_translations, target_lang)
            leap_jaccard_score = calculate_jaccard(leap_translations, target_lang)
            
            # chrF++ Metric
            regular_chrf = calculate_chrf(ground_truth_translation, regular_translations)
            leap_chrf = calculate_chrf(ground_truth_translation, leap_translations)

            with open(results_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([regular_jtc_score, leap_jtc_score, 
                                 regular_jaccard_score, leap_jaccard_score,
                                 regular_chrf, leap_chrf])
            
for target_lang in ["French"]:
    for dataset_index in [0, 1]:
        run_pipeline(target_lang=target_lang,
                    dataset_index=dataset_index, # 0 is Law, 1 is Medical
                    results_file=f"gpt4o_{target_lang}_{dataset_index}_results.csv")
