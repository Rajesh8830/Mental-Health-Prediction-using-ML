import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from deepface import DeepFace
import google.generativeai as genai
from dotenv import load_dotenv
import os
import traceback
import re # Import regular expressions for parsing sentiment

# --- Constants ---

# Mental Health Resources (India/Andhra Pradesh Focus - Publicly Available)
# Using Nellore, Andhra Pradesh, India context
INDIA_MENTAL_HEALTH_HELPLINES = {
    "KIRAN (National Helpline - Ministry of Social Justice and Empowerment)": "1800-599-0019",
    "Vandrevala Foundation": "9999666555 (24x7)",
    "Fortis Stress Helpline": "+91-8376804102"
    # Note: Finding specific, reliable, public District Mental Health Programme (DMHP)
    # contact numbers for Nellore can be difficult. Focusing on national/state-level
    # and well-known NGO helplines is often more reliable for an AI.
    # Users should be guided to check local government health department websites too.
}

GENERAL_SEARCH_ADVICE_INDIA = """
You can also:
* Contact your General Practitioner (GP) or local primary health centre for guidance.
* Search online for mental health professionals (Psychiatrists, Psychologists) in Andhra Pradesh using terms like "psychologist Nellore", "psychiatrist Andhra Pradesh". Look for directories on reputable healthcare platforms.
* Check the website of the Andhra Pradesh Department of Health and Family Welfare for potential resources or directories.
* Explore reputable online therapy platforms available in India.
"""

# PHQ-9 Questions and Options (same as before)
PHQ9_QUESTIONS = [
    {"id": "q1", "text": "Little interest or pleasure in doing things"},
    {"id": "q2", "text": "Feeling down, depressed, or hopeless"},
    {"id": "q3", "text": "Trouble falling or staying asleep, or sleeping too much"},
    {"id": "q4", "text": "Feeling tired or having little energy"},
    {"id": "q5", "text": "Poor appetite or overeating"},
    {"id": "q6", "text": "Feeling bad about yourself - or that you are a failure or have let yourself or your family down"},
    {"id": "q7", "text": "Trouble concentrating on things, such as reading the newspaper or watching television"},
    {"id": "q8", "text": "Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual"},
    {"id": "q9", "text": "Thoughts that you would be better off dead, or of hurting yourself in some way"}
]
PHQ9_OPTIONS = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}
PHQ9_OPTION_LIST = list(PHQ9_OPTIONS.keys())

# --- Updated PHQ-9 Interpretation Function ---
def get_phq9_interpretation(score):
    """Returns interpretation text and a severity category."""
    if score is None:
        return "Score not calculated yet.", "unknown"

    if 0 <= score <= 4:
        severity = "none"
        text = f"Score: {score}. Depression Severity: None-minimal."
        action = "At this level, specific action might not be needed, but continue monitoring your feelings."
    elif 5 <= score <= 9:
        severity = "mild"
        text = f"Score: {score}. Depression Severity: Mild."
        action = "Suggested Action: Consider watchful waiting; repeating the PHQ-9 at follow-up might be helpful. Monitor your symptoms."
    elif 10 <= score <= 14:
        severity = "moderate"
        text = f"Score: {score}. Depression Severity: Moderate."
        action = "Suggested Action: This score suggests moderate symptoms. It would be beneficial to consider seeking support, such as counseling or talking to a doctor. Follow-up is recommended."
    elif 15 <= score <= 19:
        severity = "moderately_severe"
        text = f"Score: {score}. Depression Severity: Moderately Severe."
        action = "Suggested Action: This score indicates significant symptoms. It is strongly recommended to seek professional help, such as active treatment with pharmacotherapy and/or psychotherapy. Please consult a healthcare provider soon."
    elif 20 <= score <= 27:
        severity = "severe"
        text = f"Score: {score}. Depression Severity: Severe."
        action = "Suggested Action: This score suggests severe symptoms. It is very important to seek professional help immediately. Options include pharmacotherapy and/or psychotherapy. Please consult a healthcare provider as soon as possible."
    else:
        return "Invalid score.", "unknown"

    return f"{text} {action}", severity


# --- Setup (same as before) ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("🔴 GOOGLE_API_KEY not found.")
    st.stop()
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
except Exception as e:
    st.error(f"🔴 Failed to configure Gemini: {e}")
    st.stop()

# Emotion Capture Function (same as before)
def capture_emotion():
    # ... (function code remains identical to previous version) ...
    """Capture image from webcam and analyze emotion"""
    cap = cv2.VideoCapture(0) # Use 0 for default camera

    if not cap.isOpened():
        st.error("Could not access camera")
        return None, None, None

    countdown_placeholder = st.empty()
    for i in range(3, 0, -1):
        countdown_placeholder.markdown(f"## Capturing in {i}... Look at the camera")
        time.sleep(1)
    countdown_placeholder.empty()

    ret, frame = cap.read()
    cap.release() # Release camera immediately after capture

    if not ret:
        st.error("Failed to capture image")
        return None, None, None

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    try:
        frame_np = np.array(image)
        analysis = DeepFace.analyze(frame_np, actions=['emotion'], enforce_detection=False)

        if isinstance(analysis, list) and len(analysis) > 0:
            dominant_emotion = analysis[0]['dominant_emotion']
            emotion_score = analysis[0]['emotion'][dominant_emotion]
            return image, dominant_emotion, f"{emotion_score:.1f}%"
        else:
             st.warning("Could not detect face or extract emotion reliably.")
             return image, "Unknown", "N/A"
    except ValueError as ve:
        st.warning(f"Face detection failed: {str(ve)}. Trying without strict enforcement.")
        return image, "Unknown", "N/A"
    except Exception as e:
        st.error(f"Emotion detection encountered an error: {str(e)}")
        return image, "Unknown", "N/A"

# --- Session State Initialization (same as before) ---
# Conversation and emotion state
if 'conversation' not in st.session_state:
    st.session_state.conversation = [
        {"role": "assistant", "content": "Hello! How are you feeling today? Click 'Analyze My Expression' if you'd like, then tell me what's on your mind."}
    ]
# ... (rest of session state initializations remain the same) ...
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'detected_emotion' not in st.session_state:
    st.session_state.detected_emotion = None
if 'emotion_confidence' not in st.session_state:
    st.session_state.emotion_confidence = None
if 'capture_requested' not in st.session_state:
     st.session_state.capture_requested = False
# PHQ-9 state
if 'phq9_active' not in st.session_state:
    st.session_state.phq9_active = False
if 'phq9_current_q' not in st.session_state:
    st.session_state.phq9_current_q = 0
if 'phq9_answers' not in st.session_state:
    st.session_state.phq9_answers = {}
if 'phq9_score' not in st.session_state:
    st.session_state.phq9_score = None
if 'phq9_asked_to_start' not in st.session_state:
    st.session_state.phq9_asked_to_start = False
if 'show_phq9_offer' not in st.session_state:
     st.session_state.show_phq9_offer = False


# --- Main App UI ---
st.title("🤖 AI Mental Health Assistant")
st.caption("Remember: I am an AI assistant, not a healthcare professional. This is not a substitute for professional medical advice, diagnosis, or treatment. Current location context: Nellore, AP, India.")

# Display conversation history (same as before)
for message in st.session_state.conversation:
    role = "assistant" if message["role"] == "assistant" else message["role"]
    with st.chat_message(role):
        st.markdown(message["content"])

# Display captured image if available (same as before)
if st.session_state.captured_image:
    st.image(
        st.session_state.captured_image,
        caption=f"Detected emotion: {st.session_state.detected_emotion} (Confidence: {st.session_state.emotion_confidence})",
        width=200
    )

# --- Emotion Capture Button (same as before) ---
if not st.session_state.capture_requested and not st.session_state.phq9_active :
    if st.button("✨ Analyze My Expression"):
        st.session_state.capture_requested = True
        # ... (rest of button logic is the same) ...
        with st.spinner("Analyzing your expression... Please wait."):
            img, emotion, confidence = capture_emotion()
            if img:
                st.session_state.captured_image = img
                st.session_state.detected_emotion = emotion
                st.session_state.emotion_confidence = confidence
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": f"Thanks! I noticed your expression seems to be '{emotion}'. Now, feel free to tell me what's on your mind."
                })
            else:
                 st.session_state.conversation.append({
                     "role": "assistant",
                     "content": "I couldn't capture your expression clearly. No worries, just tell me how you're feeling."
                 })
            st.rerun()

# --- PHQ-9 Questionnaire Section ---
if st.session_state.phq9_active:
    st.markdown("---")
    st.subheader("PHQ-9 Questionnaire")
    st.markdown("Over the *last 2 weeks*, how often have you been bothered by any of the following problems?")

    current_q_index = st.session_state.phq9_current_q
    if current_q_index < len(PHQ9_QUESTIONS):
        # ... (Question display and radio button logic remains the same) ...
        question = PHQ9_QUESTIONS[current_q_index]
        st.markdown(f"{current_q_index + 1}. {question['text']}")
        answer = st.radio(
            "Select an option:",
            options=PHQ9_OPTION_LIST,
            key=f"phq9_q_{question['id']}",
            horizontal=True,
            label_visibility="collapsed"
        )
        if st.button("Next Question", key=f"phq9_next_{question['id']}"):
            st.session_state.phq9_answers[question['id']] = PHQ9_OPTIONS[answer]
            st.session_state.phq9_current_q += 1
            # Special check for Q9 (suicidal ideation) - immediate high severity flag
            if question['id'] == 'q9' and PHQ9_OPTIONS[answer] > 0:
                 st.session_state.phq9_high_alert_q9 = True # Flag this specific concern
            st.rerun()

    else:
        # --- PHQ-9 Completion and Severity Check ---
        st.session_state.phq9_score = sum(st.session_state.phq9_answers.values())
        interpretation_text, severity_level = get_phq9_interpretation(st.session_state.phq9_score)

        st.success("Questionnaire Completed!")
        st.markdown(interpretation_text)
        st.warning("*Disclaimer:* This questionnaire is a screening tool and *not* a substitute for a professional diagnosis. Please consult a healthcare provider for any health concerns.")

        # Construct the final message based on severity
        final_message = f"Thank you for completing the PHQ-9. {interpretation_text}\n\n"

        # Check for high alert on Q9 specifically
        high_alert_q9 = st.session_state.get('phq9_high_alert_q9', False)
        if high_alert_q9:
             final_message += "*Given your answer to question 9, it's especially important to talk to someone right away. Please reach out to a helpline or a professional immediately.*\n\n"

        # Append resource info based on severity level
        if severity_level in ['moderate', 'moderately_severe', 'severe'] or high_alert_q9:
            final_message += "*Based on these results, seeking professional support is strongly recommended.*\n\n"
            final_message += "*Who can help?*\n"
            final_message += "* *Psychiatrists:* Medical doctors specializing in mental health, can prescribe medication.\n"
            final_message += "* *Psychologists/Therapists/Counselors:* Provide talk therapy and coping strategies.\n"
            final_message += "* *General Practitioners (GPs):* Can be a first point of contact for assessment and referral.\n\n"

            final_message += "*How to find help (Nellore / Andhra Pradesh / India context):*\n"
            if INDIA_MENTAL_HEALTH_HELPLINES:
                final_message += "*Helplines (available across India):*\n"
                for name, number in INDIA_MENTAL_HEALTH_HELPLINES.items():
                    final_message += f"* *{name}:* {number}\n"
                final_message += "(These helplines offer immediate support and guidance.)\n\n"

            final_message += "*Other ways to find professionals:*\n"
            final_message += GENERAL_SEARCH_ADVICE_INDIA + "\n"
            final_message += "*Please remember, reaching out is a sign of strength. You don't have to go through this alone.*\n\n"
        elif severity_level == 'mild':
             final_message += "Continue to monitor how you're feeling. If things don't improve or get worse, talking to a GP or counselor could be helpful.\n\n"
        else: # None severity
             final_message += "It's good you're keeping track of your feelings. Continue practicing self-care.\n\n"

        final_message += "How are you feeling after reflecting on these questions?"

        st.session_state.conversation.append({
            "role": "assistant",
            "content": final_message
        })

        # Reset PHQ-9 state
        st.session_state.phq9_active = False
        st.session_state.phq9_asked_to_start = False # Allow re-offer later if needed
        st.session_state.show_phq9_offer = False
        st.session_state.phq9_answers = {}
        st.session_state.phq9_high_alert_q9 = False # Reset Q9 alert
        # Keep score in state for potential future reference in session? Optional.
        # st.session_state.phq9_score = None
        st.session_state.phq9_current_q = 0

        st.rerun()

# --- Offer PHQ-9 (same logic as before) ---
if st.session_state.show_phq9_offer and not st.session_state.phq9_active:
    st.markdown("---")
    st.info("Based on our conversation, it sounds like you might be going through a tough time. Would you be interested in answering a few standard questions (PHQ-9) to help understand these feelings better? It's completely optional.")
    col1, col2, col3 = st.columns([1,1,5])
    with col1:
        if st.button("Yes, start"):
            # ... (logic for starting PHQ-9 is the same) ...
            st.session_state.phq9_active = True
            st.session_state.phq9_current_q = 0
            st.session_state.phq9_answers = {}
            st.session_state.phq9_score = None
            st.session_state.show_phq9_offer = False
            st.session_state.phq9_asked_to_start = True
            st.session_state.phq9_high_alert_q9 = False # Ensure Q9 flag is reset at start

            st.session_state.conversation.append({
                "role": "assistant",
                "content": "Okay, let's begin the PHQ-9 questionnaire. Please answer based on how you've felt over the *last 2 weeks*."
            })
            st.rerun()
    with col2:
         if st.button("No, not now"):
             # ... (logic for declining is the same) ...
             st.session_state.show_phq9_offer = False
             st.session_state.phq9_asked_to_start = True

             st.session_state.conversation.append({
                 "role": "assistant",
                 "content": "Okay, no problem at all. We can just continue chatting. What else is on your mind?"
             })
             st.rerun()
    st.markdown("---")


# --- User Input Section ---
chat_input_disabled = st.session_state.phq9_active
user_input = st.chat_input("Type your message here...", disabled=chat_input_disabled, key="user_chat_input")

if user_input and not chat_input_disabled:
    st.session_state.conversation.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        try:
            detected_emotion_context = st.session_state.get('detected_emotion', 'not captured')
            emotion_confidence_context = st.session_state.get('emotion_confidence', 'N/A')

            # --- MODIFIED PROMPT FOR LONGER RESPONSE & INTEGRATION ---
            prompt = f"""
            You are a compassionate, empathetic, and supportive AI mental health assistant. Your goal is to engage in a thoughtful conversation.

            Context:
            - User's message: "{user_input}"
            - Detected facial emotion (if available): {detected_emotion_context} (Confidence: {emotion_confidence_context})

            Your task:
            1.  Acknowledge the user's message directly.
            2.  Show genuine empathy and understanding for the feelings expressed in their text.
            3.  If a facial emotion was detected (and isn't 'Unknown' or confidence too low - consider anything below 40% low), gently integrate it with the text. For example: "I hear you're feeling [feeling from text], and I also noticed your expression seemed [detected emotion], that sounds really tough." or "It sounds like [situation from text], and seeing that your expression looked [detected emotion] adds another layer to that." If emotion wasn't captured or contradicts the text strongly, focus primarily on the text.
            4.  Validate their feelings. Let them know it's okay to feel this way.
            5.  Offer brief, general reflections or supportive statements. Avoid platitudes. You could gently explore a part of what they said.
            6.  *Generate a longer, more detailed response (aim for 4-6 sentences).*
            7.  End with an open-ended, encouraging question to invite further sharing or reflection. Avoid questions that can be answered with just "yes" or "no".
            8.  *Crucially, do NOT give medical advice, diagnosis, or specific treatment recommendations.* Maintain a supportive, listening role.

            ---
            *Sentiment Analysis Task:* After crafting the response above, classify the sentiment of the user's original message ("{user_input}"). Output this classification on a new line at the VERY END of your entire output, in the format:
            SENTIMENT: [positive/negative/neutral]
            ---
            """

            # Get response from Gemini
            response = model.generate_content(prompt)
            full_llm_output = response.text

            # Parse Response and Sentiment (same logic as before)
            llm_response_text = full_llm_output
            sentiment = "neutral"
            sentiment_match = re.search(r"SENTIMENT:\s*(positive|negative|neutral)", full_llm_output, re.IGNORECASE)
            if sentiment_match:
                sentiment = sentiment_match.group(1).lower()
                llm_response_text = re.sub(r"SENTIMENT:\s*(positive|negative|neutral)", "", llm_response_text, flags=re.IGNORECASE).strip()
                # print(f"DEBUG: Detected Sentiment: {sentiment}")
            else:
                # print("DEBUG: Sentiment tag not found.")
                pass

            st.session_state.conversation.append({"role": "assistant", "content": llm_response_text})

            # Trigger PHQ-9 Offer Logic (same logic as before)
            if sentiment == 'negative' and not st.session_state.phq9_active and not st.session_state.phq9_asked_to_start:
                 st.session_state.show_phq9_offer = True
                 st.session_state.phq9_asked_to_start = True
            else:
                 st.session_state.show_phq9_offer = False

        except Exception as e:
            # ... (Error handling remains the same) ...
            error_details = traceback.format_exc()
            print(f"\n--- ERROR during Gemini API call ---")
            print(error_details)
            print(f"----------------------------------\n")
            st.error(f"API Error: Failed to get response from assistant. Details: {str(e)}")
            error_msg = "I'm currently unable to respond. Please check the connection or try again later."
            st.session_state.conversation.append({"role": "assistant", "content": error_msg})
            st.session_state.show_phq9_offer = False

    st.rerun()


# --- Reset Button (same logic as before) ---
if st.button("🔄 Start New Conversation"):
    # ... (Reset logic remains the same, ensures all relevant keys cleared) ...
    keys_to_reset = [
        'conversation', 'captured_image', 'detected_emotion', 'emotion_confidence',
        'capture_requested', 'phq9_active', 'phq9_current_q', 'phq9_answers',
        'phq9_score', 'phq9_asked_to_start', 'show_phq9_offer', 'phq9_high_alert_q9'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.conversation = [
        {"role": "assistant", "content": "Hello! How are you feeling today? Click 'Analyze My Expression' if you'd like, then tell me what's on your mind."}
    ]
    st.session_state.capture_requested = False
    st.session_state.phq9_active = False
    st.session_state.phq9_asked_to_start = False
    st.session_state.show_phq9_offer = False
    st.rerun()