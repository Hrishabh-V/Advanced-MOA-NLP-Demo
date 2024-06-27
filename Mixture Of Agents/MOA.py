import streamlit as st
from transformers import pipeline
import torch  # Import torch for GPU availability check
import json
# Define Agent class
class Agent:
    def __init__(self, task, model_name):
        self.task = task
        self.model = pipeline(task, model=model_name, device=0 if torch.cuda.is_available() else -1)

    def run(self, input_data):
        return self.model(input_data)

# Define MoAManager class
class MoAManager:
    def __init__(self):
        self.agents = {}

    def add_agent(self, task, model_name):
        self.agents[task] = Agent(task, model_name)

    def run_task(self, task, input_data):
        if task in self.agents:
            return self.agents[task].run(input_data)
        else:
            return "No agent available for this task."

# Initialize MoA Manager and add agents
moa = MoAManager()
moa.add_agent("summarization", "sshleifer/distilbart-cnn-6-6")
moa.add_agent("question-answering", "distilbert-base-uncased-distilled-squad")
moa.add_agent("text-generation", "gpt2")

# Streamlit Interface
st.title("Advanced Mixture of Agents (MoA) Demo")
st.sidebar.header("Configuration")
task = st.sidebar.selectbox("Choose a task", ["summarization", "question-answering", "text-generation"])

st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
    }
    .highlight {
        background-color: #f0f0f5;
        padding: 10px;
        border-radius: 5px;
        color: black; /* Set text color to black */
    }
</style>
""", unsafe_allow_html=True)

input_text = st.text_area("Input Text", height=200)

if task == "question-answering":
    question = st.text_input("Question", "")

if st.button("Run Task"):
    st.markdown('<p class="big-font highlight">Result:</p>', unsafe_allow_html=True)
    with st.spinner("Processing..."):
        if task == "summarization":
            result = moa.run_task(task, input_text)
            if result:
                st.write(result[0]['summary_text'])
            else:
                st.error("Error processing summarization task.")
        elif task == "question-answering":
            input_data = {"question": question, "context": input_text}
            result = moa.run_task(task, input_data)
            formatted_result = json.dumps(result, indent=4)
            st.code(formatted_result, language='json')
            
        elif task == "text-generation":
            result = moa.run_task(task, input_text)
            if result:
                st.write(result[0]['generated_text'])
            else:
                st.error("Error processing text-generation task.")

# Load data examples
st.sidebar.header("Example Data")
example_task = st.sidebar.selectbox("Choose example task", ["summarization", "question-answering", "text-generation"])
if example_task == "summarization":
    st.sidebar.markdown("""
    **Example Input:**
    ```
    Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. AI is continuously evolving to benefit many different industries. Machines are wired using a cross-disciplinary approach based on mathematics, computer science, linguistics, psychology, and more. John McCarthy, an American computer scientist, coined the term 'artificial intelligence' in 1956 at the Dartmouth Conference, where the discipline was born.
    ```
    """)
elif example_task == "question-answering":
    st.sidebar.markdown("""
    **Example Context:**
    ```
    Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. AI is continuously evolving to benefit many different industries. Machines are wired using a cross-disciplinary approach based on mathematics, computer science, linguistics, psychology, and more. John McCarthy, an American computer scientist, coined the term 'artificial intelligence' in 1956 at the Dartmouth Conference, where the discipline was born.
    ```
    **Example Question:**
    ```
    Who coined the term 'artificial intelligence' and when?
    ```
    """)
elif example_task == "text-generation":
    st.sidebar.markdown("""
    **Example Input:**
    ```
    Once upon a time,
    ```
    """)

