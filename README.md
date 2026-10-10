# HAC vs NAI — Web app (Streamlit)

Clinical decision support: 10-variable ElasticNet model (M10, no ACTHst, no clinical signs)
to distinguish hyperadrenocorticism (HAC) from non-adrenal illness (NAI) in dogs,
from LDDST + routine biology. Bilingual EN/FR, UK/US units.

- Entry point: `app.py`
- Model: `modele_M10_logregEN.joblib` + `metadata_M10.json`
  (legacy F3 kept as fallback: `modele_F3_logregEN.joblib` + `metadata.json`)
- Methods: `material_and_methods.txt`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Posit Connect Cloud, public)

Publish from GitHub with framework **Streamlit**, entry point `app.py`.
No secrets required.

## Important

- Research / second-opinion tool — correlate with physical exam, history and imaging.
- Do not upload identifiable patient data to the public instance.
- Raw training data (`DataSet0…`, `dataset_clean.csv`) is **never** committed to this repo.
