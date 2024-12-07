import os
import csv
from datasets import load_dataset
import pandas as pd
import spacy
from deep_translator import GoogleTranslator
from tqdm import tqdm
from llamaapi import LlamaAPI
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

# Access the Llama API token
LLAMA_API_TOKEN = os.getenv("LLAMA_API_TOKEN")

# Flags for which LLMs to use
USE_LLAMA = True
USE_MISTRAL = False
USE_FALCON = False

# Define max_length multiplier for LLM prompts
MAX_LENGTH_MULTIPLIER = 2

# Define k hyperparameter
K_HYPERPARAMETER = 3

# Load datasets into pandas DataFrames
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

# Load SpaCy model
def load_spacy_model(model_path='../en_core_sci_sm-0.5.4/en_core_sci_sm/en_core_sci_sm-0.5.4'):
    return spacy.load(model_path)

# Extract entities from text
def extract_entities(nlp, text):
    doc = nlp(text)
    return list(set([ent.text for ent in doc.ents]))

# Translate entities to a target language
def translate_entities(entities, target_lang):
    translations = [GoogleTranslator(source='auto', target=target_lang).translate(entity) for entity in entities]
    return translations

# Initialize the Llama API
llama = LlamaAPI(LLAMA_API_TOKEN)

def llama_translate(prompt, max_tokens, model="llama3.3-70b"):
    """Function to interact with the Llama API for translation."""
    api_request_json = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_token": max_tokens,
        "temperature": 0.01,
        "top_p": 0.3,
        "frequency_penalty": 0.5
    }
    response = llama.run(api_request_json)
    # Parse and return the generated text
    result = response.json()
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"].strip()
    return ""

# Calculate JTC score
def calculate_JTC(translations, text, entities):
    jtc_score = 0
    n = len(entities)

    for entity, translated_entity in entities.items():
        # Count occurrences of the entity in the original text
        c = text.count(entity)
        if c == 0 or not translated_entity:
            continue  # Skip entities that are not present or have no translation

        for translation in translations:
            # Count occurrences of the translated entity in the translation
            t = translation.count(translated_entity)
            # Calculate penalty for mismatched occurrences
            penalty = abs(c - t) / max(c, 1)  # Avoid division by zero
            jtc_score += penalty

    # Normalize and invert the score
    normalized_score = jtc_score / max(n, 1)

    return 1 - normalized_score

# TODO: calculate Jaccard Similarity

# TODO: calculate chrf++ score

def run_pipeline(target_lang, results_file):
    # Load datasets
    datasets = load_all_datasets()
    datanames = ["Law", "Medical"]

    # Load NLP model
    nlp = load_spacy_model()

    # Google Translate abbreviations
    lang_abbrs = {"Simplified Chinese": "zh-CN", "French": "fr"}

    for dataname, dataset in zip(datanames, datasets):
        progress_bar = tqdm(dataset[:10], desc=f"Processing {dataname}", unit="text")
        for text in progress_bar:
            text = " ".join(text.split()[:50])  # Truncate to the first 50 words
            entities = extract_entities(nlp, text)

            # Regular translations
            regular_translations = []
            for _ in range(K_HYPERPARAMETER):
                prompt = f"Translate the following text to {target_lang}: {text}"
                max_length = len(text) * MAX_LENGTH_MULTIPLIER
                regular_translations.append(llama_translate(prompt, max_length))

            # LEAP translations
            leap_translations = []
            translated_entities = translate_entities(entities, lang_abbrs[target_lang])
            entity_mapping = {e: t for e, t in zip(entities, translated_entities)}
            for _ in range(K_HYPERPARAMETER):
                prompt = f"Translate the following text to {target_lang} using these mappings {str(entities)}: {text}"
                max_length = len(text) * MAX_LENGTH_MULTIPLIER
                leap_translations.append(llama_translate(prompt, max_length))

            # Calculate JTC scores
            print("TEXT: ", text)
            print("ENTITY MAPPING: ", entity_mapping)
            print("REGULAR TRANSLATIONS: ", regular_translations)
            print("LEAP TRANSLATIONS: ", leap_translations)
            regular_jtc_score = calculate_JTC(regular_translations, text, entity_mapping)
            leap_jtc_score = calculate_JTC(leap_translations, text, entity_mapping)
            with open(results_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([dataname, regular_jtc_score, leap_jtc_score])


if __name__ == "__main__":
    run_pipeline("Simplified Chinese", "llama_chinese_translations.csv")
    run_pipeline("French", "llama_french_translations.csv")