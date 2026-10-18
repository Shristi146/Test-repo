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
    layout="centered",
)

st.markdown("""
<style>
.block-container {max-width: 980px; padding-top: 2.2rem; padding-bottom: 3rem;}
h1 {letter-spacing: -0.035em; line-height: 1.15;}
[data-testid="stMetric"] {background: rgba(120,130,150,.08); padding: 14px 16px; border-radius: 12px;}
div[data-testid="stAlert"] {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)
st.title("🎤 Multimodal Interview Assessment")
st.caption("A practice tool for reviewing an interview recording. It gives feedback on speech, key answer points, and optional facial-expression labels—not a hiring verdict.")

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
        raw = classifier(text[:5000], top_k=None)
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        scores = {}
        for item in raw or []:
            label = str(item.get("label", "unknown")).strip().lower()
            score = float(item.get("score", 0.0))
            if "positive" in label:
                key = "Positive"
            elif "negative" in label:
                key = "Negative"
            elif "neutral" in label:
                key = "Neutral"
            else:
                key = label.title()
            scores[key] = score
        if not scores:
            return {"label": "Unavailable", "score": 0.0, "scores": {}, "note": "The model returned no sentiment labels."}
        best = max(scores, key=scores.get)
        return {"label": best, "score": scores[best], "scores": scores,
                "note": "Text sentiment is a rough language-model estimate. It does not measure answer quality, confidence, or interview performance."}
    except Exception as exc:
        return {
            "label": "Unavailable",
            "score": 0.0,
            "note": f"Sentiment model unavailable: {str(exc)[:180]}"
        }
FILLERS_BY_LANGUAGE = {
    "english": [
        "um", "uh", "erm", "umm", "uhh",
        "you know", "i mean", "kind of", "sort of",
        "basically", "actually", "literally"
    ],
    "hindi": [
        "मतलब", "अम्म", "उह", "यानी"
    ],
    "spanish": [
        "eh", "este", "o sea", "pues"
    ],
    "french": [
        "euh", "ben", "en fait"
    ],
    "german": [
        "äh", "ähm", "also"
    ],
    "portuguese": [
        "é", "tipo", "né"
    ],
}


def count_fillers(text: str, language_name: str):
    """
    Count possible filler words and phrases in a transcript.

    This is an approximate transcript-based count. It cannot reliably
    distinguish a filler from a word used intentionally in a sentence.
    """
    language = language_name.lower().strip()
    candidates = FILLERS_BY_LANGUAGE.get(
        language,
        FILLERS_BY_LANGUAGE["english"]
    )

    text_lower = text.lower()
    counts = {}

    # Check longer phrases first to reduce overlapping matches.
    for filler in sorted(candidates, key=len, reverse=True):
        pattern = (
            r"(?<!\w)"
            + re.escape(filler.lower())
            + r"(?!\w)"
        )

        count = len(re.findall(pattern, text_lower))

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


def score_speech(filler_count: int, wpm: float, word_count: int):
    """
    Calculate approximate practice indicators for speaking pace
    and detected filler frequency.

    These are heuristic indicators, not scientifically validated
    measures of interview performance.
    """

    # Speaking pace indicator
    if word_count <= 0 or wpm <= 0:
        pace_score = 0.0
    elif 110 <= wpm <= 160:
        pace_score = 100.0
    elif 90 <= wpm < 110 or 160 < wpm <= 180:
        pace_score = 75.0
    elif 70 <= wpm < 90 or 180 < wpm <= 200:
        pace_score = 50.0
    else:
        pace_score = 25.0

    # Detected filler frequency per 100 transcript words
    filler_rate = (
        filler_count / max(1, word_count)
    ) * 100

    # Filler-control practice indicator
    if word_count <= 0:
        filler_score = 0.0
    elif filler_rate <= 1:
        filler_score = 100.0
    elif filler_rate <= 3:
        filler_score = 85.0
    elif filler_rate <= 5:
        filler_score = 70.0
    elif filler_rate <= 8:
        filler_score = 50.0
    else:
        filler_score = 30.0

    # A practice indicator only; not an overall interview score.
    combined_score = (
        0.7 * pace_score
        + 0.3 * filler_score
    )

    return (
        combined_score,
        pace_score,
        filler_score,
        filler_rate
    )


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


def _pdf_safe_text(value):
    """Convert arbitrary text to a safe, printable representation for built-in Helvetica."""
    text = str(value).replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2026", "...")
    text = text.encode("latin-1", "replace").decode("latin-1")
    return "".join(ch if ch in "\n\r\t" or ord(ch) >= 32 else " " for ch in text)


def _pdf_multicell(pdf, text, line_height=6):
    """
    Safely write wrapped text into the PDF.

    Converts unsupported characters for built-in Helvetica and
    splits long unbroken tokens to reduce FPDF wrapping errors.
    """
    text = _pdf_safe_text(text)

    usable_width = (
        pdf.w
        - pdf.l_margin
        - pdf.r_margin
    )

    if usable_width <= 0:
        raise ValueError("PDF has no usable horizontal writing space.")

    wrapped_lines = []

    for original_line in text.splitlines() or [""]:
        words = original_line.split()
        output_words = []

        for word in words:
            # Split tokens that are wider than the available line.
            chunks = []

            while (
                word
                and pdf.get_string_width(word) > usable_width
            ):
                cut = 1

                while (
                    cut < len(word)
                    and pdf.get_string_width(word[:cut + 1])
                    <= usable_width
                ):
                    cut += 1

                chunks.append(word[:cut])
                word = word[cut:]

            if word:
                chunks.append(word)

            output_words.extend(chunks or [""])

        wrapped_lines.append(" ".join(output_words))

    pdf.set_x(pdf.l_margin)

    pdf.multi_cell(
        w=usable_width,
        h=line_height,
        text="\n".join(wrapped_lines)
    )
    
class InterviewReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(90, 90, 90)
            self.cell(0, 7, "MULTIMODAL INTERVIEW ASSESSMENT | PRACTICE REPORT", align="L", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(210, 210, 210)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 6, f"Practice feedback only - Page {self.page_no()}", align="C")


def generate_pdf(report: dict):
    pdf = InterviewReportPDF(format="A4")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(auto=True, margin=17)
    pdf.add_page()

    pdf.set_fill_color(32, 43, 61)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 19)
    pdf.cell(0, 15, "Interview Practice Report", align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(70, 70, 70)
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 9)
    _pdf_multicell(pdf, "Purpose: practice feedback for interview recordings. The indicators are estimates for learning, not validated measures of employability and not grounds for hiring decisions.", 5)
    pdf.ln(4)

    def section_heading(title):
        pdf.set_fill_color(232, 237, 244)
        pdf.set_text_color(32, 43, 61)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _pdf_safe_text(title), fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    section_heading("Summary")
    pdf.set_text_color(35, 35, 35)
    pdf.set_font("Helvetica", "", 10)
    for key, value in report.get("summary", {}).items():
        _pdf_multicell(pdf, f"{key}: {value}", 5.5)
    pdf.ln(3)

    for section, lines in report.get("sections", {}).items():
        section_heading(section)
        pdf.set_text_color(35, 35, 35)
        pdf.set_font("Helvetica", "", 10)
        for line in lines:
            _pdf_multicell(pdf, f"- {line}", 5.5)
        pdf.ln(3)

    output = pdf.output()
    return bytes(output) if isinstance(output, (bytes, bytearray)) else output.encode("latin-1")

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

st.subheader("2. What should a good answer include? (optional)")
st.caption("If you know the interview question, list a few important points a strong answer should mention. The app checks whether related words appear; it cannot reliably judge meaning or answer quality. Leave blank if you do not have a checklist.")
expected_points = st.text_area(
    "Key points to look for",
    placeholder="Example: explain the problem, describe what you did, share the result, explain what you learned",
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
                filler_count = sum(filler_result.values())
                speech_score, pace_score, filler_score, filler_rate = score_speech(filler_count, wpm, word_count)

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
            st.text_area("Transcript for review", value=transcript, height=220, help="Review the automatic transcription carefully. This display does not change the analysis already calculated in this run.")

            st.markdown("#### Speaking feedback")
            s1, s2, s3 = st.columns(3)
            s1.metric("Speaking pace", f"{wpm:.0f} words/min")
            s2.metric("Filler phrases found", f"{filler_count}")
            s3.metric("Filler frequency", f"{filler_rate:.1f} per 100 words")
            st.caption("Filler detection is approximate: the app checks a small list of phrases and may miss some or count a phrase used intentionally.")
            if word_count < 50:
                st.info("This transcript is short, so the practice scores are only rough estimates. Try a longer answer for more useful feedback.")
            else:
                p1, p2, p3 = st.columns(3)
                p1.metric("Pace practice indicator", f"{pace_score:.0f}/100")
                p2.metric("Filler control indicator", f"{filler_score:.0f}/100")
                p3.metric("Combined practice indicator", f"{speech_score:.0f}/100")
            st.write("**Detected filler phrases:** " + (", ".join(f"{word} ({count})" for word, count in filler_result.items()) if filler_result else "None detected"))
            with st.expander("Keyword hints (optional technical detail)"):
                st.write(", ".join(keywords) if keywords else "Not available for this language or NLP model.")

            st.markdown("#### Tone of the words (rough estimate)")
            st.write(f"**Most likely label:** {sentiment['label']}")
            sentiment_scores = sentiment.get("scores", {})
            if sentiment_scores:
                sentiment_cols = st.columns(min(3, len(sentiment_scores)))
                for idx, (label, score) in enumerate(sorted(sentiment_scores.items())):
                    sentiment_cols[idx % len(sentiment_cols)].metric(label, f"{score:.0%}")
            st.caption(sentiment["note"])

            st.markdown("#### Key points in your answer")
            if content_score is None:
                st.info("Not checked. Add a few key points above before analysis if you want the app to check whether your answer mentions them. This is optional.")
            else:
                st.metric("Checklist points detected", f"{len(matched_points)} of {len(matched_points) + len(missing_points)}")
                if matched_points:
                    st.write("**Points that seem to be mentioned**")
                    for point in matched_points:
                        st.write(f"✅ {point}")
                if missing_points:
                    st.write("**Points to consider adding**")
                    for point in missing_points:
                        st.write(f"○ {point}")
                st.caption("This is a simple word-match check, not a grade. It can miss paraphrases and cannot confirm whether a point was explained correctly.")

            st.markdown("#### Optional facial-expression labels")
            if run_face_analysis:
                if emotions:
                    counts = Counter(emotions)
                    total = len(emotions)
                    st.write(f"Frames with a detected face: {total}")
                    for label, count in counts.most_common():
                        st.write(f"**{label.title()}** — {count} samples ({count / total:.0%})")
                    st.caption("These are labels assigned to a small sample of video frames. They can be wrong and do not tell us whether someone is confident, honest, competent, or suitable for a job.")
                else:
                    st.warning(emotion_note or "No usable face samples were found. This is not a negative result.")
            else:
                st.write("Not run. Facial-expression analysis is optional and off by default because it can be inaccurate and biased.")

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
                        f"Filler frequency: {filler_rate:.1f} per 100 words",
                        f"Detected fillers: {filler_result}",
                    ],
                    "Sentiment": [
                        f"Most likely label: {sentiment['label']}",
                        "Label estimates: " + ", ".join(f"{k} {v:.0%}" for k, v in sentiment.get("scores", {}).items()),
                        sentiment["note"],
                    ],
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
            # Report export is isolated: a PDF formatting problem must not erase analysis results.
            try:
                pdf_bytes = generate_pdf(report)
                st.download_button(
                    "📄 Download practice report (PDF)",
                    data=pdf_bytes,
                    file_name="interview_practice_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as pdf_exc:
                st.warning("The analysis completed, but the PDF report could not be generated. Your on-screen results are still available.")
                st.caption(f"PDF export detail: {pdf_exc}")
            st.download_button(
        
                "Download transcript (TXT)",
                data=transcript.strip() + "\n",
                file_name="interview_transcript.txt",
                mime="text/plain; charset=utf-8",
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
