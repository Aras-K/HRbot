import os
import json
import re
import io
import base64
import matplotlib.pyplot as plt
import nltk

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import RetrievalQA
import faiss
import gradio as gr

###############################################################################
#                      Environment Setup & Basic Config                       #
###############################################################################
nltk.download('punkt')
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

###############################################################################
#                          Initialize Vector Store                            #
###############################################################################
# We use the sentence-transformers/all-mpnet-base-v2 model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create a FAISS index with 768-dim vectors for the above model
index = faiss.IndexFlatL2(768)
vector_store = FAISS(
    embedding_function=embeddings.embed_query,
    index=index,
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={}
)

def create_llm():
    """
    Creates a ChatOpenAI model (GPT-3.5-turbo) with zero temperature.
    Adjust 'model_name' to "gpt-4" if you have access.
    """
    return ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", max_tokens=2000)

# Build a RetrievalQA chain that uses our FAISS vector store
llm = create_llm()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

###############################################################################
#                              Utility Functions                              #
###############################################################################
def load_and_chunk_file(file_path, chunk_size=1000, chunk_overlap=100):
    """
    Loads a file (PDF, DOCX, TXT), splits it into chunks.
    Returns a list of text chunks.
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        # For .docx, .txt, etc.
        loader = UnstructuredFileLoader(file_path)

    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_texts = []
    for page in pages:
        chunks = text_splitter.split_text(page.page_content)
        chunked_texts.extend(chunks)
    return chunked_texts

def add_chunks_to_vector_store(chunks, metadata_tag=""):
    """
    Adds chunked texts to the FAISS vector store, optionally tagging with a candidate name.
    """
    tagged_chunks = []
    for chunk in chunks:
        text = f"{metadata_tag}\n{chunk}" if metadata_tag else chunk
        tagged_chunks.append(text)
    vector_store.add_texts(tagged_chunks)

def summarize_cv_text(long_text):
    """
    Summarizes the CV text focusing on key skills & qualifications.
    """
    prompt = (
        "Please provide a concise bullet-point summary of this CV text, focusing on "
        "key skills, experience, and standout qualifications:\n\n"
        f"{long_text}\n\n"
        "Keep it under 200 words."
    )
    response = qa_chain.run(prompt)
    return response

###############################################################################
#                Step 1: Summarize Each CV & Store the Chunks                #
###############################################################################
def process_cv_files(cv_files):
    """
    1) Load and chunk each CV
    2) Add to FAISS vector store
    3) Summarize the full CV
    Return { "Candidate 1": "...summary...", "Candidate 2": "...", ... }
    """
    summaries = {}
    for idx, file_obj in enumerate(cv_files, start=1):
        candidate_name = f"Candidate {idx}"
        temp_path = file_obj.name

        # Load & chunk
        chunks = load_and_chunk_file(temp_path)

        # Add to vector store with candidate name as a tag
        add_chunks_to_vector_store(chunks, metadata_tag=candidate_name)

        # Summarize entire CV
        combined_text = " ".join(chunks)
        short_summary = summarize_cv_text(combined_text)
        summaries[candidate_name] = short_summary

    return summaries

###############################################################################
#              Step 2: Read / Summarize Project Description                  #
###############################################################################
def process_project_file(project_file):
    """
    Loads the project file (PDF/DOCX/TXT) and returns the combined text.
    """
    file_path = project_file.name
    chunks = load_and_chunk_file(file_path, chunk_size=1000, chunk_overlap=100)
    project_text = " ".join(chunks)
    return project_text

###############################################################################
#  Step 3: LLM => (JSON + Compliance) from a Single Response (Two-Part Output)
###############################################################################
def rank_candidates_with_json_and_compliance(cv_summaries, project_text):
    """
    Calls the LLM to:
      1) Return JSON with candidate scores & explanations.
      2) Then provide 'Compliance with requirements' sections for each candidate,
         in the format:
             ### Candidate X
             Compliance with requirements:
             - Some Requirement: ✓
             - Another Requirement: ✗

    We'll parse these two parts separately:
      - The JSON for ranking
      - The compliance lines for charts
    """
    system_msg = (
        "You are an advanced language model designed to score and rank candidates. "
        "We have multiple candidates with short CV summaries, plus a project/job description."
    )

    user_msg = f"Project Description:\n{project_text}\n\n"
    user_msg += "Candidate Summaries:\n"
    for name, summary in cv_summaries.items():
        user_msg += f"{name}:\n{summary}\n\n"

    # Instruct the LLM carefully: return JSON first, then compliance text
    user_msg += (
        "First, return valid JSON ONLY in this format:\n"
        "{\n"
        "  \"candidates\": {\n"
        "    \"Candidate 1\": {\n"
        "      \"score\": 85,\n"
        "      \"explanation\": \"...\"\n"
        "    },\n"
        "    ...\n"
        "  }\n"
        "}\n"
        "No extra text before the opening '{' or after the closing '}'.\n"
        "Immediately after that JSON (on a new line), provide a 'Compliance with requirements' section "
        "for each candidate, in the format:\n\n"
        "### Candidate 1\n"
        "Compliance with requirements:\n"
        "- Requirement A: ✓\n"
        "- Requirement B: ✗\n\n"
        "- Requirement C: ✗\n\n\n"
        "There are multiple requirements."
        "They should be individually listed."
        "Do not combine or summarize them."
        "No disclaimers, no additional commentary. End of instructions."
    )

    direct_llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", max_tokens=2000)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    response = direct_llm.invoke(messages)
    full_text = response.content.strip()

    print("DEBUG: Full LLM response:\n", full_text)

    # 1) Extract the JSON part by searching for first '{' and the corresponding last '}'
    start_idx = full_text.find('{')
    end_idx = full_text.rfind('}')
    if start_idx == -1 or end_idx == -1:
        print("DEBUG: Could not find JSON braces in LLM response.")
        return None, None

    json_str = full_text[start_idx:end_idx+1].strip()
    compliance_str = full_text[end_idx+1:].strip()  # everything after the JSON

    print("DEBUG: Extracted JSON:\n", json_str)
    print("DEBUG: Extracted compliance text:\n", compliance_str)

    return json_str, compliance_str

###############################################################################
#          Step 4: Parse the JSON & Extract Top 10, Parse Compliance          #
###############################################################################
def parse_top_10_candidates(json_string):
    """
    Parses the JSON string to extract top 10 candidates by score.
    Returns a list of dicts like:
    [
      { 'name': 'Candidate 1', 'score': 85, 'explanation': '...' },
      ...
    ]
    """
    if not json_string:
        print("DEBUG: JSON string is empty or None.")
        return []

    try:
        data = json.loads(json_string)
        candidates = data["candidates"]

        candidate_list = [
            {"name": name, "score": info["score"], "explanation": info["explanation"]}
            for name, info in candidates.items()
        ]

        candidates_sorted = sorted(candidate_list, key=lambda x: x["score"], reverse=True)
        return candidates_sorted[:10]

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []
    except KeyError as e:
        print(f"Missing key in JSON response: {e}")
        return []

def parse_summary_for_requirements(long_text):
    """
    Searches for:
       ### Candidate X
       Compliance with requirements:
       - Some Requirement: ✓
       - Another Requirement: ✗
    Returns a dict:
    {
      "Candidate 4": [
         ("Requirement 1", 1),
         ("Requirement 2", 0)
      ],
      ...
    }
    Where 1 = ✓, 0 = ✗
    """
    candidate_sections = re.split(r"(?=###\s*(Candidate|Kandidat))", long_text, flags=re.IGNORECASE)
    candidate_data = {}

    for section in candidate_sections:
        section = section.strip()
        if not section.startswith("###"):
            continue

        cand_match = re.match(r"###\s*(Candidate|Kandidat)\s+(\d+)", section, re.IGNORECASE)
        candidate_name = f"Candidate {cand_match.group(2)}" if cand_match else "Unknown Candidate"

        # Look for "Compliance with requirements:" block
        comp_regex = r"(Compliance with requirements:|Samsvar med krav:)(.*?)(?:\n###|\Z)"
        comp_section_match = re.search(comp_regex, section, flags=re.IGNORECASE | re.DOTALL)
        if not comp_section_match:
            continue

        comp_section = comp_section_match.group(2).strip()
        requirements = []
        for line in comp_section.split("\n"):
            line = line.strip()
            # e.g., "- Requirement A: ✓"
            if line.startswith("- "):
                line = line[2:].strip()
                req_match = re.match(r"(.*?):\s*(✓|✗)", line)
                if req_match:
                    req_name = req_match.group(1).strip()
                    req_mark = req_match.group(2)
                    compliance = 1 if req_mark == "✓" else 0
                    requirements.append((req_name, compliance))

        candidate_data[candidate_name] = requirements

    return candidate_data

def generate_compliance_charts(candidate_data):
    """
    For each candidate, generate a chart (green/red dots) of their compliance.
    Returns a dict { candidate_name: <img base64> } for embedding in HTML.
    """
    chart_paths = {}
    for candidate, reqs in candidate_data.items():
        if not reqs:
            continue

        req_names = [r[0] for r in reqs]
        compliances = [r[1] for r in reqs]

        fig, ax = plt.subplots(figsize=(6, max(len(reqs) * 0.4, 2)))
        y_positions = range(len(reqs))

        for y_pos, compliance in zip(y_positions, compliances):
            color = 'green' if compliance == 1 else 'red'
            label_text = "Met" if compliance == 1 else "Not Met"
            ax.plot(0, y_pos, marker='o', color=color, markersize=10)
            ax.text(0.05, y_pos, label_text, va='center', fontsize=7, color=color)

        ax.set_yticks(list(y_positions))
        ax.set_yticklabels(req_names, fontsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_xticks([])

        ax.set_title(f'{candidate} Compliance with Requirements', fontsize=8, color='#34495E')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        chart_paths[candidate] = (
            f"<img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}' "
            f"alt='{candidate} chart'/>"
        )
        plt.close(fig)

    return chart_paths

###############################################################################
#      Step 5: Main Function to Show Top 10 + Compliance in Gradio App       #
###############################################################################
def process_and_rank_candidates_with_compliance(cv_files, project_file):
    """
    1) Summarize & store CVs
    2) Read project text
    3) LLM => JSON (ranking) + compliance lines
    4) Parse top 10 from JSON
    5) Parse compliance lines => generate charts
    6) Build final HTML
    """
    if not project_file:
        return "Please upload the project description."

    # 1) Summaries of each CV
    cv_summaries = process_cv_files(cv_files)

    # 2) Project text
    project_text = process_project_file(project_file)

    # 3) LLM => (json_str, compliance_str)
    json_str, compliance_str = rank_candidates_with_json_and_compliance(cv_summaries, project_text)
    if not json_str:
        return "<p style='color:red;'>Error: Could not get JSON from the LLM.</p>"

    # 4) Parse top 10
    top_10 = parse_top_10_candidates(json_str)

    # 5) Parse compliance lines & generate charts
    compliance_data = parse_summary_for_requirements(compliance_str)
    compliance_charts = generate_compliance_charts(compliance_data)

    # 6) Build final HTML
    result_html = "<h2>Top 10 Candidates</h2>\n"
    for i, cand in enumerate(top_10, start=1):
        name = cand["name"]
        score = cand["score"]
        explanation = cand["explanation"]
        result_html += f"<h4>{i}. {name} — Score: {score}</h4>\n"
        result_html += f"<p>{explanation}</p>\n"

        # Insert compliance chart if found
        if name in compliance_charts:
            result_html += compliance_charts[name]

        result_html += "<hr/>\n"

    return result_html

###############################################################################
#                              Gradio Interface                               #
###############################################################################
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center; color: #1842a8; font-family: Arial;'>
            CV Ranking & Compliance Visualization
        </h1>
        <p style='text-align: center;'>
            By Kogniti
        </p>
        """
    )

    with gr.Row():
        with gr.Column():
            cv_uploads = gr.File(
                label="Upload multiple CVs",
                file_types=[".pdf", ".docx", ".txt"],
                file_count="multiple"
            )
        with gr.Column():
            project_upload = gr.File(
                label="Upload Project Description",
                file_types=[".pdf", ".docx", ".txt"]
            )

    btn_rank = gr.Button("Find Top 10 Candidates")
    output_html = gr.HTML(label="Results")

    btn_rank.click(
        fn=process_and_rank_candidates_with_compliance,
        inputs=[cv_uploads, project_upload],
        outputs=output_html
    )

if __name__ == "__main__":
    demo.launch()
