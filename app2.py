import streamlit as st
from textblob import TextBlob
import google.generativeai as genai
from dotenv import load_dotenv
import os, string, datetime, time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
st.set_page_config(page_title="AI Mental Health Asst", layout="wide")
DOCTOR_INFO=[{"name":"Dr. Anya Sharma","specialty":"Psychiatrist","address":"123 Wellness Way, Bangalore","phone":"9876543210"},{"name":"Mr. Rohan Desai","specialty":"Clinical Psychologist","address":"456 Mindful Street, Bangalore","phone":"9876543211"},{"name":"Dr. Priya Singh","specialty":"Counseling Psychologist","address":"789 Serenity Ave, Bangalore","phone":"9876543212"}]
PHQ9_QUESTIONS=["Little interest or pleasure in doing things","Feeling down, depressed, or hopeless","Trouble falling or staying asleep, or sleeping too much","Feeling tired or having little energy","Poor appetite or overeating","Feeling bad about yourself — or that you are a failure or have let yourself or your family down","Trouble concentrating on things, such as reading the newspaper or watching television","Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual","Thoughts that you would be better off dead, or of hurting yourself in some way"]
PHQ9_RESPONSES={"Not at all":0,"Several days":1,"More than half the days":2,"Nearly every day":3}
RESPONSE_OPTIONS=list(PHQ9_RESPONSES.keys())
PHQ9_FINAL_QUESTION="If you checked off any problems, how difficult have these problems made it for you to do your work, take care of things at home, or get along with other people?"
PHQ9_FINAL_RESPONSES=["Not difficult at all","Somewhat difficult","Very difficult","Extremely difficult"]
SEVERE_THRESHOLD, MODERATELY_SEVERE_THRESHOLD, MODERATE_THRESHOLD, MILD_THRESHOLD = 20, 15, 10, 5
LOG_FILE = "interaction_log_arch.txt"; GEMINI_MODEL_NAME = 'gemini-1.5-pro-latest'
GEMINI_PROMPT_TEMPLATE="""You are an AI assistant... [Your full prompt here, same as before] ...consult a qualified healthcare professional... End with a clear disclaimer...
User's Context:
- Initial Statement: "{initial_statement}"
- Extracted Keywords from Initial Statement: {keywords_str}
- PHQ-9 Score: {score} / 27
- Assessed Severity Level (based on score): {severity}
- Score on Question 9 (Thoughts of self-harm): {q9_score} (0-3 scale)
- Reported Functional Difficulty: {functional_difficulty}
Instructions for Response: ... [Rest of instructions same as before] ...""" # Keep the full prompt template content

# --- Setup ---
load_dotenv(); API_KEY = os.getenv("GOOGLE_API_KEY")
model = None
if API_KEY:
    try: genai.configure(api_key=API_KEY); model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e: st.error(f"Error configuring Gemini API: {e}.")
else: st.error("⚠ *GOOGLE_API_KEY not found.* Please set it in your environment.")

@st.cache_resource
def load_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']; #print("Checking NLTK resources...")
    for resource in resources:
        try: nltk.download(resource, quiet=True)
        except Exception as e: st.error(f"Failed to download NLTK resource '{resource}'. Error: {e}")
    #print("NLTK resources check complete.")
if API_KEY and model: load_nltk_resources()
stop_words = set(stopwords.words('english')); lemmatizer = WordNetLemmatizer()

# --- Helper Functions ---
def analyze_sentiment(text): return TextBlob(text).sentiment.polarity
def log_interaction(log_data):
    log_entry = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
    for key, value in log_data.items():
        value_str = str(value)[:250] + "...[truncated]" if key == "response_given" and len(str(value)) > 500 else str(value)
        log_entry += f"  {key.replace('_', ' ').title()}: {value_str}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f: f.write(log_entry + "---\n")
    except Exception as e: print(f"Error writing log: {e}") # Keep one print for critical log errors
def get_severity_level(score, q9_score):
    if q9_score > 0 and score < MODERATE_THRESHOLD: return "Mild to Moderate (with safety concern)"
    if score >= SEVERE_THRESHOLD: return "Severe"
    if score >= MODERATELY_SEVERE_THRESHOLD: return "Moderately Severe"
    if score >= MODERATE_THRESHOLD: return "Moderate"
    if score >= MILD_THRESHOLD: return "Mild"
    return "Minimal"
def extract_keywords(text, max_keywords=10):
    if not text: return []
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    lemmas = [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word.isalnum() and word not in stop_words]
    return [word for word, freq in nltk.FreqDist(lemmas).most_common(max_keywords)]
def get_llm_suggestions_stream(score, q9_score, severity, functional_difficulty, initial_statement, keywords):
    if not model: yield "\n\n⚠ *Gemini Model Not Initialized.*"; return
    try:
        prompt = GEMINI_PROMPT_TEMPLATE.format(initial_statement=initial_statement or "Not provided", keywords_str=', '.join(keywords) if keywords else 'None', score=score, severity=severity, q9_score=q9_score, functional_difficulty=functional_difficulty)
        response_stream = model.generate_content(prompt, stream=True)
        for chunk in response_stream: time.sleep(0.05); yield chunk.text
    except Exception as e: yield f"\n\nSorry, error contacting AI model: {e}"; print(f"Gemini API Error: {e}") # Keep API error print

# --- Streamlit App UI and Logic ---
st.title("AI Mental Health Assistant")
if not API_KEY or not model: st.stop() # Stop if key missing or model failed init

st.markdown("Disclaimer: Not a substitute for professional advice. **In crisis? Contact emergency services (e.g., 112 India) or a hotline.")
st.markdown("---")

default_states = {'conversation_phase':"initial", 'phq9_answers':{}, 'question_index':0, 'score':0, 'q9_score':0, 'functional_difficulty_answer':None, 'current_selection':None, 'history':[{"role": "assistant", "content": "Hi there! How are you feeling today?"}], 'assessment_complete_message':None, 'initial_statement':"", 'extracted_keywords':[]}
for key, default_value in default_states.items(): st.session_state.setdefault(key, default_value)

chat_container = st.container()
with chat_container:
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            content = st.session_state.assessment_complete_message if message.get("is_assessment") and st.session_state.assessment_complete_message else message["content"]
            st.markdown(content, unsafe_allow_html=True)

def submit_answer():
    selected_option = st.session_state.current_selection
    if selected_option:
        score = PHQ9_RESPONSES[selected_option]; q_index = st.session_state.question_index
        st.session_state.score += score; st.session_state.phq9_answers[q_index] = score
        if q_index == 8: st.session_state.q9_score = score
        st.session_state.history.append({"role": "user", "content": f"Q{q_index + 1}: {selected_option}"})
        st.session_state.question_index += 1; st.session_state.current_selection = None
        if st.session_state.question_index >= len(PHQ9_QUESTIONS):
            st.session_state.conversation_phase = "asking_final_q"
            for i in range(len(PHQ9_QUESTIONS)): st.session_state.pop(f"phq9_q_{i}_displayed", None)

def submit_final_answer():
    selected_option = st.session_state.current_selection
    if selected_option:
        st.session_state.functional_difficulty_answer = selected_option
        st.session_state.history.append({"role": "user", "content": f"Difficulty: {selected_option}"})
        st.session_state.conversation_phase = "assessment_complete"; st.session_state.current_selection = None
        st.session_state.pop("final_q_displayed", None); st.session_state.pop("assessment_stream_started", None)

# --- State Machine ---
phase = st.session_state.conversation_phase

if phase == "initial":
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("You:", key="user_input_initial", placeholder="How are you feeling?")
        if st.form_submit_button("Send") and user_input:
            st.session_state.history.append({"role": "user", "content": user_input})
            sentiment = analyze_sentiment(user_input); log_data = {"user_input": user_input, "sentiment": f"{sentiment:.2f}"}
            if sentiment <= 0.05:
                st.session_state.initial_statement = user_input; st.session_state.extracted_keywords = extract_keywords(user_input)
                ai_response = "Thanks for sharing. To understand better, I'll ask some standard screening questions (PHQ-9). Ready?"
                st.session_state.update(conversation_phase="asking_phq9", question_index=0, score=0, q9_score=0, phq9_answers={}, functional_difficulty_answer=None, assessment_complete_message=None)
                st.session_state.pop('assessment_stream_started', None)
                log_data.update(state="Initial -> Starting PHQ-9", response_given="[Starting Questionnaire Intro]", extracted_keywords=st.session_state.extracted_keywords)
            else:
                st.session_state.initial_statement = ""; st.session_state.extracted_keywords = []
                ai_response = "That's good to hear! Feel free to share more."
                log_data.update(state="Initial (Positive)", response_given=ai_response)
            st.session_state.history.append({"role": "assistant", "content": ai_response}); log_interaction(log_data); st.rerun()

elif phase == "asking_phq9":
    q_index = st.session_state.question_index
    if q_index < len(PHQ9_QUESTIONS):
        question_text = PHQ9_QUESTIONS[q_index]; full_prompt = f"*Over the last 2 weeks, how often have you been bothered by:\n\n{q_index + 1}. {question_text}*"
        display_flag = f"phq9_q_{q_index}_displayed"
        if display_flag not in st.session_state:
            st.session_state.history.append({"role": "assistant", "content": full_prompt}); st.session_state[display_flag] = True; st.rerun()
        st.radio(label=f"Q {q_index+1}", options=RESPONSE_OPTIONS, key="current_selection", horizontal=False, label_visibility="collapsed")
        st.button("Submit Answer", on_click=submit_answer, key=f"submit_q_{q_index}")
    elif phase == "asking_phq9": st.session_state.conversation_phase = "asking_final_q"; st.rerun() # Transition safety net

elif phase == "asking_final_q":
    display_flag = "final_q_displayed"
    if display_flag not in st.session_state:
        st.session_state.history.append({"role": "assistant", "content": f"*Finally:* {PHQ9_FINAL_QUESTION}"}); st.session_state[display_flag] = True; st.rerun()
    st.radio(label="Difficulty Q", options=PHQ9_FINAL_RESPONSES, key="current_selection", horizontal=False, label_visibility="collapsed")
    st.button("Submit Final Answer", on_click=submit_final_answer, key="submit_final")

elif phase == "assessment_complete":
    if "assessment_stream_started" not in st.session_state:
        st.session_state["assessment_stream_started"] = True
        st.session_state.history.append({"role": "assistant", "content": "Generating suggestions...", "is_assessment": True}); st.rerun()
    elif st.session_state.assessment_complete_message is None:
         severity = get_severity_level(st.session_state.score, st.session_state.q9_score)
         llm_args = {"score": st.session_state.score, "q9_score": st.session_state.q9_score, "severity": severity, "functional_difficulty": st.session_state.functional_difficulty_answer, "initial_statement": st.session_state.initial_statement, "keywords": st.session_state.extracted_keywords}
         with st.chat_message("assistant"): full_response_text = st.write_stream(get_llm_suggestions_stream(**llm_args))
         doc_info_md = ""
         if st.session_state.score >= MODERATE_THRESHOLD or st.session_state.q9_score > 0:
             doc_info_md = "\n\n---\n*Finding Professional Help in Bangalore:\n_(Verify details)_\n" + "".join([f"- **{d['name']}* ({d['specialty']})\n  - Addr: {d['address']}, Ph: {d['phone']}\n" for d in DOCTOR_INFO])
         st.session_state.assessment_complete_message = full_response_text + doc_info_md
         st.session_state.history[-1]["content"] = "[AI Assessment & Suggestions Provided]" # Update placeholder
         log_data = {"state": "Assessment Complete", "phq9_score": st.session_state.score, "q9_score": st.session_state.q9_score, "functional_difficulty": st.session_state.functional_difficulty_answer, "assessed_severity": severity, "initial_statement_provided": bool(st.session_state.initial_statement), "extracted_keywords": st.session_state.extracted_keywords, "response_given": full_response_text}
         log_interaction(log_data); st.rerun()

# --- Footer ---
st.markdown("---")
if st.button("Start Over / Clear Chat"):
    keys_to_clear = [k for k in st.session_state.keys() if not k.startswith('_')]
    for key in keys_to_clear: del st.session_state[key]
    st.session_state.update(default_states); st.rerun()