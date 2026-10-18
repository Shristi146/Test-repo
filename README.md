# Multimodal Interview Assessment System (improved)

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository or a branch for this version.
2. Upload these files to the repository root:
   - `streamlit_app.py`
   - `requirements.txt`
   - `packages.txt`
   - `runtime.txt`
3. In Streamlit Community Cloud, create/redeploy the app and select `streamlit_app.py` as the main file.
4. First launch may take several minutes because Whisper and the multilingual sentiment model download model weights.
5. If deployment fails, open **Manage app → Logs** and check the first package/model error.

## Notes

- Whisper `base` is the default compromise for CPU and memory. `small` may exceed resource limits or be slow on Streamlit Community Cloud.
- Facial analysis is optional and off by default. DeepFace/TensorFlow dependencies can make installation heavy; if deployment fails due to them, remove `deepface` and `tf-keras` from `requirements.txt` and keep the feature disabled, or move facial analysis to a separate service.
- The multilingual sentiment model is a separate download and is used for descriptive sentiment only. Sentiment is not used to determine answer quality.
- Content coverage is only calculated when the user provides a checklist. It uses transparent phrase matching, not semantic grading.
- WPM uses estimated speech-active time. The active-speech estimate is approximate and can vary with background noise and pauses.
- Do not use the generated indicators to rank candidates or make employment decisions.
- This is an improved prototype, not a scientifically validated assessment model. Validate on manually reviewed recordings before reporting accuracy claims.
