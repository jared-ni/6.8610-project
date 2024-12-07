import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from huggingface_hub import login
import spacy
from deep_translator import GoogleTranslator
from datasets import load_dataset
import pandas as pd
import csv

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

def calculate_JTC(translations, text, entities):
    jtc_score = 0
    n = len(entities)
    for entity, translated_entity in entities.items():
        if translated_entity is None:
                continue
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


# Initialize the models using GPT-4o-mini and spacy
llm_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
spacy_model = spacy.load('en_core_sci_sm-0.5.4/en_core_sci_sm/en_core_sci_sm-0.5.4')
K_HYPERPARAMETER = 3


def run_pipeline(target_lang, results_file):
    datasets = load_all_datasets()
    # Google Translate abbreviations
    lang_abbrs = {"Simplified Chinese": "zh-CN", "French": "fr"}

    # law[0] and medical[1] datasets
    for dataset in datasets:
        for i, text in enumerate(dataset[:10]):
            text = " ".join(text.split()[:50])  # Truncate to the first 30 words
            print(f"Text {i+1}: {text}")
            named_entities = list(set(extract_entities(spacy_model, text)))
            print("Named entities:", named_entities)
            # Translate named entities to the target language
            named_entities_translations = translate_entities(named_entities, lang_abbrs[target_lang])
            print("Translations:", named_entities_translations)
            named_entity_mapping = {e: t for e, t in zip(named_entities, 
                                                         named_entities_translations)}


            # regular text translation
            regular_translations = []
            for k in range(1):
                prompt = f"Translate the following text to {target_lang}: {text}"
                response = llm_model.invoke(prompt)
                print("\nGenerated Response:", response.content)
                regular_translations.append(response.content)
            
            # LEAP text translation
            leap_translations = []
            for k in range(1):
                prompt = f"Translate the following text to {target_lang}\
                    using these mappings {str(named_entities_translations)}: {text}"
                response = llm_model.invoke(prompt)
                print("\nGenerated Response (LEAP):", response.content)
                leap_translations.append(response.content)
            
            regular_jtc_score = calculate_JTC(regular_translations, text,
                                              named_entity_mapping)
            leap_jtc_score = calculate_JTC(leap_translations, text, 
                                           named_entity_mapping)
            
            with open(results_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([regular_jtc_score, leap_jtc_score])

run_pipeline("Simplified Chinese", "results.csv")
