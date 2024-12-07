import csv
from datasets import load_dataset
import pandas as pd
import spacy
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
def load_spacy_model(model_path='en_core_sci_sm'):
    return spacy.load(model_path)

# Extract entities from text
def extract_entities(nlp, text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

# Translate entities to a target language
def translate_entities(entities, target_lang):
    translations = [GoogleTranslator(source='auto', target=target_lang).translate(entity) for entity in entities]
    return translations

# Calculate JTC score
def calculate_JTC(translations, text, entities):
    jtc_score = 0
    n = len(entities)

    for entity, translated_entity in entities.items():
        # Count how many times the entity appears in the text
        c = text.count(entity)
        if c == 0:  
            continue

        for translation in translations:
            # Count how many times the translated_entity appears in the translation
            t = translation.count(translated_entity)
            # Add to the score based on the proportion of appearances
            jtc_score += abs(c - t) / c 

    # Normalize the score by the number of entities
    jtc_score = 1 - (jtc_score / n if n > 0 else 0)

    return jtc_score

def run_pipeline(target_lang, results_file):
    # Load datasets
    datasets = load_all_datasets()

    # Load NLP model
    nlp = load_spacy_model()

    # Google Translate abbreviations
    lang_abbrs = {"Simplified Chinese": "zh-CN", "French": "fr"}

    # Load LLMs
    # llama_tokenizer, llama_model = load_llama_model() if USE_LLAMA else (None, None)
    # mistral_tokenizer, mistral_model = load_mistral_model() if USE_MISTRAL else (None, None)
    # falcon_tokenizer, falcon_model = load_falcon_model() if USE_FALCON else (None, None)

    for dataset in datasets:
        for i, text in enumerate(dataset[:10]):  # Iterate through the first 10 entries for testing
            text = " ".join(text.split()[:30])  # Truncate to the first 30 words
            entities = extract_entities(nlp, text)

            for llm_name, tokenizer, model, is_active in [
                ("Llama", llama_tokenizer, llama_model, USE_LLAMA),
                ("Mistral", mistral_tokenizer, mistral_model, USE_MISTRAL),
                ("Falcon", falcon_tokenizer, falcon_model, USE_FALCON),
            ]:
                if not is_active:
                    continue

                # Regular translations
                regular_translations = []
                for k in range(K_HYPERPARAMETER):
                    prompt = f"Translate the following text to {target_lang}: {text}"
                    max_length = len(prompt) * MAX_LENGTH_MULTIPLIER
                    regular_translations.append(
                        llama_generate_text(tokenizer, model, prompt, max_length)
                        )

                # LEAP translations
                leap_translations = []
                translated_entities = translate_entities(entities, lang_abbrs[target_lang])
                entity_mapping = {e: t for e, t in zip(entities, translated_entities)}
                for k in range(K_HYPERPARAMETER):
                    prompt = f"Translate the following text to {target_lang} using these mappings {str(entities)}: {text}"
                    max_length = len(prompt) * MAX_LENGTH_MULTIPLIER
                    leap_translations.append(
                        llama_generate_text(tokenizer, model, prompt, max_length)
                    )

                # Calculate JTC scores
                print("TEXT: ", text)
                print("ENTITY MAPPING: ", entity_mapping)
                print("REGULAR TRANSLATIONS: ", regular_translations)
                print("LEAP TRANSLATIONS: ", leap_translations)
                regular_jtc_score = calculate_JTC(regular_translations, text, entity_mapping)
                leap_jtc_score = calculate_JTC(leap_translations, text, entity_mapping)
                with open(results_file, mode='a', newline='', encoding='utf-8') as file:
                  writer = csv.writer(file)
                  writer.writerow([regular_jtc_score, leap_jtc_score])


if __name__ == "__main__":
    run_pipeline("Simplified Chinese", "chinese_translations.csv")
    run_pipeline("French", "french_translations.csv")