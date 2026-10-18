# Multimodal Interview Assessment System

## Deploy on Streamlit Community Cloud

1. Extract this ZIP file.
2. Upload the four deployment files (`streamlit_app.py`, `requirements.txt`, `packages.txt`, and `runtime.txt`) to the root of a GitHub repository.
3. In Streamlit Community Cloud, deploy that repository and select `streamlit_app.py` as the main file.
4. The first deployment can take several minutes because Whisper and the multilingual sentiment model download model weights.
5. If deployment fails, open **Manage app → Logs** and check the first error.

## Included features

- Video upload and audio extraction
- Whisper speech transcription with selectable model and language
- Estimated speaking pace and active speech time
- Basic filler-word counting and keyword hints
- Optional descriptive sentiment estimate
- Optional facial-expression sampling (off by default)
- Optional checklist-based content coverage estimate
- One PDF practice report and one JSON export

## Important limitations

This is a learning prototype, not a scientifically validated assessment system. The speech metrics and checklist coverage are approximate. Sentiment and facial-expression labels are not measures of answer quality, confidence, honesty, competence, or employability. Do not use this app to rank, reject, or make employment decisions about candidates. Obtain permission before uploading recordings.

Facial analysis uses DeepFace/TensorFlow dependencies and can make cloud installation slower or less reliable. If deployment fails due to those dependencies, remove `deepface` and `tf-keras` from `requirements.txt` and leave facial analysis disabled; the rest of the app is designed to continue without it.
