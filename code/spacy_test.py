import spacy
import scispacy

model_path = 'en_core_sci_sm-0.5.4/en_core_sci_sm/en_core_sci_sm-0.5.4'
nlp = spacy.load(model_path)

text = "A 51-year-old Caucasian woman presented with fever and profound neutropenia (48 neutrophils/uL). \
Her clinical history included non-Hodgkin lymphoma, in remission following treatment with allogeneic bone marrow transplantation, quiescent chronic graft-versus-host disease, and squamous cell carcinoma of the skin metastatic to cervical lymph nodes. \
Medications included atenolol, topical clobetasol, Ditropan (oxybutynin), prophylactic voriconazole, prophylactic valganciclovir, Soriatane (acitretin), and Carac (fluorouracil) cream. \
The bone marrow was hypocellular without metastatic cancer or myelodysplasia. \
Neutropenia did not respond to stopping medications that have been associated with neutropenia (valganciclovir, voriconazole and Soriatane) or treatment with antibiotics or granulocyte colony stimulating factor. \
Blood tests revealed absence of antineutrophil antibodies, normal folate and B12 levels, moderate zinc deficiency and severe Vitamin B6 deficiency. \
Replacement therapy with oral Vitamin B6 restored blood vitamin levels to the normal range and corrected the neutropenia. \
Her cervical adenopathy regressed clinically and became negative on scintography following Vitamin B6 therapy and normalization of the blood neutrophil count."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)