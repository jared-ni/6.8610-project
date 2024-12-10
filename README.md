# LEAP: Learned Entity Alignment Prompting for Consistent Jargon Translation

Maintaining translation consistency—particularly for proper nouns and domain-specific terminology (jargon) — is a significant challenge in machine translation. Inconsistent translations, especially in fields that require high language accuracy (such as justice and medicine), can lead to confusion, lack of clarity, and incorrect interpretations.

We introduce **Learned Entity Alignment Prompting (LEAP)**, a multi-step framework de- signed to enhance translation consistency in large language models (LLMs). LEAP utilizes named entity recognition and a translation oracle to construct domain-specific entity mappings, which are integrated into model prompts to ensure consistent translations across queries.

To evaluate consistency, we propose a novel metric, **Jargon Translation Consistency (JTC)**, alongside established metrics such as Jaccard Similarity and chrF++. Experiments conducted on domain-specific datasets (legal and medical) demonstrate that LEAP significantly improves consistency metrics compared to regular prompting methods by up to **2.1x**.

### Running the Pipeline
To run the LEAP pipeline for ChatGPT4o and translate from English to Chinese:
```
python chatopenai_pipeline_chinese.py
```
To run the LEAP pipeline for ChatGPT4o and translate from English to French:
```
python chatopenai_pipeline_chinese.py
```
To run the LEAP pipeline for Mistral and translate from English to French:
```
python mistral_pipeline_french.py
```
To run the LEAP pipeline for Mistral and translate from English to Chinese:
```
python mistral_pipeline_chinese.py
```
To run the LEAP pipeline for Claude: run the Python notebook ```claude_pipeline.ipynb```
