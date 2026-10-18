import os
import re
import json
import time
import tempfile
import subprocess
from collections import Counter

import cv2
import numpy as np
import streamlit as st
import whisper
from fpdf import FPDF

# Optional / expensive components are imported only when needed.
try:
    import librosa
except Exception:
    librosa = None

try:
    from deepface import DeepFace
except Exception:
    DeepFace = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None


st.set_page_config(
    page_title="Multimodal Interview Assessment",
    page_icon="🎤",
    layout="wide",
)

st.title("🎤 Multimodal Interview Assessment System")
st.caption(
    "Practice feedback for interview recordings. Scores are estimates for learning, "
    "not validated measures of employability or grounds for hiring decisions."
)

with st.expander("⚠️ Limitations, privacy and responsible use"):
    st.markdown(
        """
        - **Not a hiring decision tool.** Automated scores can be wrong and should not be
          used to rank or reject real candidates.
        - **Transcription:** accuracy varies with noise, accents, code-switching, and language.
          Always review the transcript before trusting downstream results.
        - **Sentiment:** sentiment is not answer quality. A negative or neutral tone can still
          contain an excellent answer. Sentiment is shown as a weak descriptive signal only.
        - **Facial expressions:** apparent emotion is not a reliable measure of confidence,
          honesty, competence, or personality. Face-analysis results are optional and may be
          biased by lighting, pose, camera quality, and skin tone.
        - **Content scoring:** without a question and a reference checklist, content coverage
          cannot be judged meaningfully. This app uses a transparent checklist when you provide one.
        - **Privacy:** recordings are processed by the deployed app. Do not upload recordings
          without permission. Avoid uploading sensitive or identifying material.
        """
    )


@st.cache_resource(show_spinner=False)
def load_whisper(model_name: str):
    return whisper.load_model(model_name, device="cpu")


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    if pipeline is None:
        return None
    # Multilingual model; first use downloads model weights and may take time.
    return pipeline(
        "text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        device=-1,
        truncation=True,
    )


def run_command(args):
    """Run a subprocess safely and raise a readable error if it fails."""
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "Unknown error")[-1800:]
        raise RuntimeError(detail)
    return result


def extract_audio(video_path: str, audio_path: str):
    # Mono 16 kHz WAV is suitable for speech recognition and speech-activity estimates.
    run_command([
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", audio_path
    ])


def estimate_speech_seconds(audio_path: str, fallback_duration: float) -> float:
    """Estimate active speech duration using non-silent audio intervals."""
    if librosa is None:
        return fallback_duration
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        if len(y) == 0:
            return fallback_duration
        intervals = librosa.effects.split(y, top_db=28, frame_length=2048, hop_length=512)
        active = sum(max(0, end - start) for start, end in intervals) / sr
        # Avoid implausible near-zero duration caused by noise-gate settings.
        if active <= 0:
            return fallback_duration
        return min(float(fallback_duration), float(active))
    except Exception:
        return fallback_duration


def get_audio_duration(audio_path: str) -> float:
    try:
        result = run_command([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ])
        return max(0.0, float(result.stdout.strip()))
    except Exception:
        if librosa is not None:
            try:
                return float(librosa.get_duration(path=audio_path))
            except Exception:
                pass
        return 0.0


def transcribe_audio(audio_path: str, model_name: str, language_choice: str):
    model = load_whisper(model_name)
    options = {"fp16": False, "verbose": False}
    if language_choice != "Auto-detect":
        options["language"] = language_choice
    result = model.transcribe(audio_path, **options)
    return result.get("text", "").strip(), result.get("language", "unknown")


def analyze_sentiment(text: str):
    if not text.strip():
        return {"label": "Unavailable", "score": 0.0, "note": "No transcript was produced."}
    try:
        classifier = load_sentiment_pipeline()
        if classifier is None:
            return {
                "label": "Unavailable",
                "score": 0.0,
                "note": "Transformers could not be imported; sentiment was skipped."
            }
        # Keep inference bounded for long transcripts.
        result = classifier(text[:5000])[0]
        return {
            "label": str(result.get("label", "unknown")).upper(),
            "score": float(result.get("score", 0.0)),
            "note": "Multilingual sentiment estimate; not a measure of answer quality."
        }
    except Exception as exc:
        return {
            "label": "Unavailable",
            "score": 0.0,
            "note": f"Sentiment model unavailable: {str(exc)[:180]}"
        }


FILLERS_BY_LANGUAGE = {
    "english": ["um", "uh", "erm", "you know", "sort of", "kind of", "i mean"],
    "hindi": ["मतलब", "अम्म", "उह", "यानी"],
    "spanish": ["eh", "este", "o sea", "pues"],
    "french": ["euh", "ben", "en fait"],
    "german": ["äh", "ähm", "also"],
    "portuguese": ["é", "tipo", "né"],
}


def count_fillers(text: str, language_name: str):
    candidates = FILLERS_BY_LANGUAGE.get(language_name.lower(), FILLERS_BY_LANGUAGE["english"])
    lower = text.lower()
    counts = {}
    for filler in candidates:
        if re.search(r"[^\w\s]", filler):
            count = lower.count(filler)
        else:
            count = len(re.findall(r"(?<!\w)" + re.escape(filler) + r"(?!\w)", lower))
        if count:
            counts[filler] = count
    return counts


def extract_keywords(text: str, language_code: str):
    # English NLP keywords only; don't pretend English keyword extraction works across languages.
    if language_code != "en":
        return []
    try:
        import spacy
        nlp = load_spacy()
        doc = nlp(text[:100000])
        words = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in {"NOUN", "PROPN", "VERB"}
            and not token.is_stop
            and len(token.lemma_) > 2
        ]
        return [word for word, _ in Counter(words).most_common(20)]
    except Exception:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        stop = {"that", "this", "with", "from", "have", "were", "what", "when", "your", "about"}
        return [word for word, _ in Counter(w for w in words if w not in stop).most_common(20)]


@st.cache_resource(show_spinner=False)
def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Avoid downloading a large model at runtime; fall back to simple tokenization.
        raise RuntimeError("English spaCy model en_core_web_sm is not installed.")


def analyze_video_emotions(video_path: str, sample_every_seconds: float, max_samples: int):
    if DeepFace is None:
        return [], "DeepFace could not be imported. Facial analysis was skipped."
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], "The video could not be opened for frame analysis."

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    step = max(1, int(fps * sample_every_seconds))
    emotions = []
    sampled = 0
    frame_index = 0
    try:
        while sampled < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                sampled += 1
                try:
                    result = DeepFace.analyze(
                        img_path=frame,
                        actions=["emotion"],
                        enforce_detection=True,
                        detector_backend="opencv",
                        silent=True,
                    )
                    if result:
                        emotion_map = result[0].get("emotion", {})
                        dominant = result[0].get("dominant_emotion")
                        confidence = float(emotion_map.get(dominant, 0))
                        if dominant and confidence >= 0:
                            emotions.append(dominant.lower())
                except Exception:
                    # No face / detector error on a sample is not an emotion label.
                    pass
            frame_index += 1
    finally:
        cap.release()

    if not emotions:
        return [], "No usable face/emotion samples were found. This is not scored as failure."
    return emotions, None


def score_content(answer: str, expected_points_text: str):
    """Transparent checklist coverage score; no checklist means not scored."""
    points = [p.strip() for p in re.split(r"[\n,;]+", expected_points_text) if p.strip()]
    if not points:
        return None, [], []
    normalized = answer.lower()
    matched, missing = [], []
    for point in points:
        # Match a meaningful phrase rather than exact token only.
        terms = [t.lower() for t in re.findall(r"\b[\w'-]+\b", point) if len(t) > 2]
        if terms and sum(term in normalized for term in terms) / len(terms) >= 0.6:
            matched.append(point)
        else:
            missing.append(point)
    score = 100.0 * len(matched) / len(points)
    return score, matched, missing


def score_speech(filler_count: int, wpm: float):
    # Pace and fillers are separate transparent practice metrics, not personality measures.
    if 110 <= wpm <= 160:
        pace_score = 100.0
    elif 90 <= wpm < 110 or 160 < wpm <= 180:
        pace_score = 75.0
    elif 70 <= wpm < 90 or 180 < wpm <= 200:
        pace_score = 50.0
    else:
        pace_score = 25.0
    total_words = max(1, filler_count)
    filler_rate = filler_count / total_words
    filler_score = max(0.0, 100.0 - 15.0 * filler_count)
    return 0.7 * pace_score + 0.3 * filler_score, pace_score, filler_score


def build_recommendations(transcript, wpm, filler_result, content_score, missing_points, language_code):
    recommendations = []
    if wpm < 90:
        recommendations.append("Your estimated speaking pace is slow. Try shorter pauses between ideas, while keeping your words clear.")
    elif wpm > 180:
        recommendations.append("Your estimated speaking pace is fast. Practise pausing between key points and recording again.")
    elif wpm > 0:
        recommendations.append("Your estimated speaking pace is within a commonly used practice range. Confirm it sounds natural by listening to the recording.")
    if sum(filler_result.values()) >= 3:
        most_common = sorted(filler_result.items(), key=lambda item: item[1], reverse=True)[:3]
        recommendations.append("Reduce repeated fillers such as " + ", ".join(f"'{w}' ({n})" for w, n in most_common) + ". Replace them with a short pause.")
    elif transcript:
        recommendations.append("Filler words detected were limited. Review the transcript because automatic filler detection can miss disfluencies.")
    if content_score is not None:
        if missing_points:
            recommendations.append("Checklist points to cover more clearly: " + ", ".join(missing_points[:5]) + ".")
        else:
            recommendations.append("All checklist points were detected by the simple text-matching rubric. Verify meaning manually; matching words does not prove a strong answer.")
    else:
        recommendations.append("Add expected answer points or a rubric to get a more meaningful content-coverage estimate.")
    if language_code != "en":
        recommendations.append("Keyword extraction is disabled for this language in this version; do not interpret missing keywords as a weak answer.")
    return recommendations


def generate_pdf(report: dict):
    """Generate a PDF report while keeping every paragraph inside page margins."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.add_page()

    def add_paragraph(text, line_height=6):
        # Reset X before every multi_cell: FPDF may otherwise leave X at the right edge.
        safe_text = str(text).encode("latin-1", "replace").decode("latin-1")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, line_height, safe_text, new_x="LMARGIN", new_y="NEXT")

    def add_heading(text, size=12):
        pdf.set_font("Helvetica", "B", size)
        pdf.set_x(pdf.l_margin)
        pdf.cell(pdf.epw, 8, str(text), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_x(pdf.l_margin)
    pdf.cell(pdf.epw, 12, "Interview Practice Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    add_paragraph("Practice feedback only. Not validated for hiring or candidate ranking.")
    pdf.ln(2)

    add_heading("Summary", 13)
    pdf.set_font("Helvetica", "", 10)
    for key, value in report.get("summary", {}).items():
        add_paragraph(f"{key}: {value}")
    pdf.ln(2)

    for section, lines in report.get("sections", {}).items():
        add_heading(section, 12)
        pdf.set_font("Helvetica", "", 10)
        for line in lines:
            add_paragraph(line)
        pdf.ln(2)

    return bytes(pdf.output())


st.subheader("1. Upload and configure")
uploaded_video = st.file_uploader("Upload an interview video", type=["mp4", "mov", "mkv", "webm"])
col1, col2, col3 = st.columns(3)
with col1:
    model_name = st.selectbox(
        "Transcription model",
        ["base", "tiny", "small"],
        index=0,
        help="base is a practical CPU default. small may be more accurate but uses more RAM and takes longer on Community Cloud."
    )
with col2:
    language_choice_name = st.selectbox(
        "Spoken language",
        ["Auto-detect", "English", "Hindi", "Spanish", "French", "German", "Portuguese", "Italian", "Japanese", "Chinese", "Korean", "Arabic", "Russian"],
        index=0,
    )
with col3:
    run_face_analysis = st.checkbox("Run optional facial-expression analysis", value=False)

language_codes = {
    "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de",
    "Portuguese": "pt", "Italian": "it", "Japanese": "ja", "Chinese": "zh", "Korean": "ko",
    "Arabic": "ar", "Russian": "ru",
}
language_choice = "Auto-detect" if language_choice_name == "Auto-detect" else language_codes[language_choice_name]

st.subheader("2. Optional content rubric")
st.caption("Enter expected concepts separated by commas or new lines. Leave blank to skip content scoring.")
expected_points = st.text_area(
    "Expected answer points",
    placeholder="Example: customer needs, proposed solution, measurable result, lesson learned",
    height=90,
)

if uploaded_video is not None:
    st.video(uploaded_video)
    if st.button("Analyze interview", type="primary", use_container_width=True):
        video_path = None
        audio_path = None
        started = time.time()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
                video_file.write(uploaded_video.getbuffer())
                video_path = video_file.name
            audio_path = os.path.splitext(video_path)[0] + ".wav"

            with st.status("Analyzing recording…", expanded=True) as status:
                st.write("Extracting audio…")
                extract_audio(video_path, audio_path)
                duration = get_audio_duration(audio_path)
                if duration <= 0:
                    raise RuntimeError("No usable audio was found in the uploaded video.")

                st.write(f"Transcribing with Whisper {model_name}…")
                transcript, detected_language = transcribe_audio(audio_path, model_name, language_choice)
                if not transcript:
                    raise RuntimeError("The transcription model returned an empty transcript.")

                speech_seconds = estimate_speech_seconds(audio_path, duration)
                word_count = len(re.findall(r"\b[\w'-]+\b", transcript, flags=re.UNICODE))
                wpm = (word_count / speech_seconds * 60.0) if speech_seconds > 0 else 0.0

                detected_code = detected_language if detected_language else "unknown"
                code_to_language = {
                    "en": "english", "hi": "hindi", "es": "spanish", "fr": "french",
                    "de": "german", "pt": "portuguese"
                }
                filler_language = (
                    language_choice_name.lower()
                    if language_choice_name != "Auto-detect"
                    else code_to_language.get(detected_code, "english")
                )
                filler_result = count_fillers(transcript, filler_language)
                keywords = extract_keywords(transcript, detected_code)
                sentiment = analyze_sentiment(transcript)
                content_score, matched_points, missing_points = score_content(transcript, expected_points)
                speech_score, pace_score, filler_score = score_speech(sum(filler_result.values()), wpm)

                emotions, emotion_note = ([], None)
                if run_face_analysis:
                    st.write("Sampling facial expressions (optional)…")
                    emotions, emotion_note = analyze_video_emotions(video_path, 2.0, 60)

                # Do not create a misleading single overall score: component validity differs.
                recommendations = build_recommendations(
                    transcript, wpm, filler_result, content_score, missing_points, detected_code
                )
                status.update(label="Analysis complete", state="complete", expanded=False)

            st.subheader("3. Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("Words transcribed", f"{word_count:,}")
            m2.metric("Estimated speaking pace", f"{wpm:.0f} WPM")
            m3.metric("Estimated speech-active time", f"{speech_seconds:.1f} sec")

            st.markdown("#### Transcript")
            st.info(f"Detected language: **{detected_code}**. Please review the transcript before using the other metrics.")
            st.text_area("Editable transcript for review", value=transcript, height=220)

            st.markdown("#### Speech practice metrics")
            s1, s2, s3 = st.columns(3)
            s1.metric("Pace practice indicator", f"{pace_score:.0f}/100")
            s2.metric("Filler control indicator", f"{filler_score:.0f}/100")
            s3.metric("Combined speech practice indicator", f"{speech_score:.0f}/100")
            st.caption("These are transparent heuristic indicators, not validated measures of communication skill.")
            st.write(f"**Filler words detected:** {filler_result if filler_result else 'None detected'}")
            st.write(f"**Keyword hints:** {', '.join(keywords) if keywords else 'Not available for this language or NLP model.'}")

            st.markdown("#### Sentiment (descriptive only)")
            st.write(f"**Result:** {sentiment['label']}" + (f" ({sentiment['score']:.0%} model confidence)" if sentiment["score"] else ""))
            st.caption(sentiment["note"])

            st.markdown("#### Content coverage")
            if content_score is None:
                st.warning("No content score calculated. Add a checklist of expected answer points before analysis to enable this section.")
            else:
                st.metric("Checklist coverage estimate", f"{content_score:.0f}/100")
                st.write("**Detected points:** " + (", ".join(matched_points) if matched_points else "None"))
                st.write("**Points to review:** " + (", ".join(missing_points) if missing_points else "None"))
                st.caption("This is simple phrase matching, not semantic grading. It can miss paraphrases and can give credit for words used without the intended meaning.")

            st.markdown("#### Optional facial-expression sampling")
            if run_face_analysis:
                if emotions:
                    counts = Counter(emotions)
                    total = len(emotions)
                    st.write(f"Usable face samples: {total}")
                    st.bar_chart({name: count for name, count in counts.items()})
                    st.caption("Do not interpret these labels as confidence, honesty, competence, or suitability.")
                else:
                    st.warning(emotion_note or "No reliable face samples were available.")
            else:
                st.write("Skipped. Facial-expression analysis is off by default because it is error-prone and can be biased.")

            st.markdown("#### Actionable recommendations")
            for recommendation in recommendations:
                st.write(f"- {recommendation}")

            report = {
                "summary": {
                    "Transcription model": model_name,
                    "Detected language": detected_code,
                    "Words transcribed": word_count,
                    "Speaking pace": f"{wpm:.1f} WPM",
                    "Speech-active time": f"{speech_seconds:.1f} seconds",
                    "Sentiment": sentiment["label"],
                    "Content checklist coverage": "Not calculated" if content_score is None else f"{content_score:.1f}/100",
                },
                "sections": {
                    "Transcript": [transcript],
                    "Speech practice metrics": [
                        f"Pace indicator: {pace_score:.1f}/100",
                        f"Filler control indicator: {filler_score:.1f}/100",
                        f"Combined speech practice indicator: {speech_score:.1f}/100",
                        f"Detected fillers: {filler_result}",
                    ],
                    "Sentiment": [sentiment["label"], sentiment["note"]],
                    "Content coverage": [
                        f"Score: {'Not calculated' if content_score is None else f'{content_score:.1f}/100'}",
                        f"Detected points: {matched_points}",
                        f"Points to review: {missing_points}",
                    ],
                    "Recommendations": recommendations,
                    "Facial-expression sampling": [
                        "Skipped" if not run_face_analysis else (emotion_note or f"Samples: {dict(Counter(emotions))}")
                    ],
                },
            }
            st.download_button(
                "📄 Download practice report (PDF)",
                data=generate_pdf(report),
                file_name="interview_practice_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Download transcript and results (JSON)",
                data=json.dumps({"transcript": transcript, "report": report}, ensure_ascii=False, indent=2),
                file_name="interview_assessment_results.json",
                mime="application/json",
                use_container_width=True,
            )
            st.caption(f"Analysis time: {time.time() - started:.1f} seconds.")
        except Exception as exc:
            st.error("Analysis could not be completed.")
            st.code(str(exc))
            st.info("If this is a deployment error, check the Streamlit Cloud app logs. Model downloads may take time on first use.")
        finally:
            for path in (video_path, audio_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
else:
    st.info("Upload a video to begin. For best results, use a clear recording with audible speech and permission from the person recorded.")
