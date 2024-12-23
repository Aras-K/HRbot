# Version with gpt-4o 
### This version integrates the ability to upload a couple of CVs, gives the user a choice to select which Cv he wants to process and generate the competence matrix based on that
# There is a function of similarity score integrated that shows which candidate matches better
# 2 languages are implemented 

import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import RetrievalQA
from docx import Document
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
import faiss
import tempfile
import os
from dotenv import load_dotenv
import nltk
import re
import matplotlib.pyplot as plt

nltk.download('punkt')

# Load API keys from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv('LANGCHAIN_ENDPOINT')
os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGCHAIN_PROJECT')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')

# Initialize tracer
client = Client()
tracer = LangChainTracer(client = client)

# Initialize embedding model and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
index = faiss.IndexFlatL2(768)  # Dimension 768 for this embedding model
vector_store = FAISS(
    embedding_function=embeddings.embed_query,
    index=index,
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={}
)

def clean_text(text):
    return " ".join(text.split()) 

def detect_language(text):
    try:
        lang = detect(text)
        if lang == "no":
            return "Norwegian"
        elif lang == "en":
            return "English"
        else: 
            return "Unknown"
    except Exception as e:
        return "Unknown"

def initialize_llm_and_chain():
    llm = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=2000)
    retriever = vector_store.as_retriever(search_kwargs={"k":5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, callbacks=[tracer])
    return llm, qa_chain

llm, qa_chain = initialize_llm_and_chain()

def process_file(file, candidate_name=None):
    file_extension = os.path.splitext(file.name)[1]
    with open(file.name, "rb") as f:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(f.read())
            temp_file_path = temp_file.name

    if file_extension == ".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension == ".docx":
        loader = UnstructuredFileLoader(temp_file_path)
    elif file_extension == ".txt":
        loader = UnstructuredFileLoader(temp_file_path)

    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    content = ""
    for page in pages:
        chunks = text_splitter.split_text(page.page_content)
        if candidate_name:
            chunks = [f"Candidate: {candidate_name}\n" + chunk for chunk in chunks]        
        content += " ".join(chunks)
        vector_store.add_texts(chunks)
    return content

def generate_comparative_summary(cv_contents, project_content):
    cv_language = detect_language(project_content)
    candidates_prompt =""
    for candidate_name, content in cv_contents.items():
        candidates_prompt += f"{candidate_name} CV: \n{content}\n\n"

    if cv_language == "Norwegian":
        prompt = (
                "Du er en avansert språkmodell designet for å sammenligne profesjonelle dokumenter. "
                "Du har fått flere kandidaters CV-er og en prosjektbeskrivelse. "
                "Du skal vurdere hver av kandidatene i forhold til prosjektets krav. "
                "Vennligst bruk navnene på kandidatene som du identifiserer fra CV-dokumentene nedenfor. "
                "Start med en kort (2–3 setninger) overordnet vurdering som identifiserer hvem som fremstår best egnet totalt sett, og hvorfor.\n\n"
                "Deretter, for hver kandidat (bruk overskriften `### Kandidat X`):\n"
                "- Under overskriften 'Samsvar med krav': "
                "  * List opp de sentrale kravene fra prosjektbeskrivelsen.\n"
                "  * Bruk en hake (✓) hvis kandidaten oppfyller kravet, og et kryss (✗) hvis ikke.\n"
                "  * Gi en kort begrunnelse (1–2 setninger) for hver vurdering.\n"
                "- Under overskriften 'Mangler / Forbedringsområder': "
                "  * List opp eventuelle mangler og gi et kort, konstruktivt forbedringsforslag.\n\n"
                "Etter at alle kandidater er analysert, lag en siste seksjon med overskriften 'Endelig konklusjon':\n"
                "- Angi tydelig hvilken kandidat som er mest egnet, og begrunn det kort.\n\n"
                "Bruk tydelig og profesjonell språkføring, kortfattede setninger, og Markdown-formatering for klarhet.\n\n"
                f"{candidates_prompt}"
                f"Prosjektbeskrivelse:\n{project_content}\n\n"
                "Gi en klar, strukturert og profesjonell analyse."
                "Husk å utføre analysen for hver enkelt kandidat med tittelen ### Kandidat X, og ikke bare for én kandidat."
            )
    else:
        prompt = (
                "You are an advanced language model designed to compare professional documents. "
                "You have received several candidates' CVs and a project description. "
                "You are to assess each of the candidates in relation to the project's requirements. "
                "Please use the names of the candidates as you identify them from the CV documents below. "
                "Start with a short (2–3 sentences) overall assessment that identifies who appears to be best suited overall, and why.\n\n"
                "Then, for each candidate (use the heading `### Candidate X`):\n"
                "- Under the heading 'Compliance with requirements': "
                "  * List the key requirements from the project description.\n"
                "  * Use a checkmark (✓) if the candidate meets the requirement, and a cross (✗) if not.\n"
                "  * Provide a brief justification (1–2 sentences) for each assessment.\n"
                "After all candidates are analyzed, create a final section with the heading 'Final Conclusion':\n"
                "- Clearly indicate which candidate is most suitable, and briefly justify it.\n\n"
                "Use clear and professional language, concise sentences, and Markdown formatting for clarity.\n\n"
                f"{candidates_prompt}"
                f"Project description:\n{project_content}\n\n"
                "Provide a clear, structured, and professional analysis."
                "Remember to perform the analysis for each individual candidate with the title ### Candidate X, and not just for one candidate."
        )
    raw_summary = qa_chain.run(prompt)
    return raw_summary

def process_and_generate_summary(cv_files, project_file, matrix_file=None):
    if not project_file:
        return "Please upload the project description."
    num_cv_files = len(cv_files)
    # Process CV files
    cv_contents = {}
    for i, cv_file in enumerate(cv_files):
        candidate_name = f"Candidate {i+1}" if num_cv_files > 1 else "Candidate"
        cv_content = process_file(cv_file, candidate_name=candidate_name)
        cv_contents[candidate_name] = cv_content

    project_content = process_file(project_file, candidate_name="Project")
    summary = generate_comparative_summary(cv_contents, project_content)
    return summary

# For similarity charts
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def calculate_similarity(text1, text2):
    embedding1 = semantic_model.encode(text1)
    embedding2 = semantic_model.encode(text2)
    similarity = util.cos_sim(embedding1, embedding2)
    return similarity.item()

def generate_similarity_chart(similarity, candidate_name):
    labels = ['Similarity', 'Difference']
    sizes = [similarity, 1 - similarity]
    colors = ['#00203fff', '#adefd1ff']
    explode = (0.1, 0) 

    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=250)
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, colors=colors,
                                      autopct='%1.1f%%', startangle=140,
                                      shadow=True, textprops={'fontsize': 8, 'color': '#F8F8EA'})
    ax.legend(wedges, labels,
              title="Categories",
              loc="center",
              bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True,
              ncol=len(labels),
              fontsize='xx-small',
              title_fontsize='xx-small')

    plt.title(f'Similarity for {candidate_name}', color='navy', fontweight='bold', fontsize=10)
    ax.axis('equal')
    file_path = os.path.join(os.getcwd(), f'{candidate_name}_similarity_chart.png')
    plt.savefig(file_path, bbox_inches='tight')  
    plt.close(fig)
    return file_path

def match_cv_to_project_and_display_charts(cv_files, project_file):
    project_content = process_file(project_file)
    results = []
    for i, cv_file in enumerate(cv_files):
        cv_content = process_file(cv_file)
        similarity_score = calculate_similarity(cv_content, project_content)
        chart_path = generate_similarity_chart(similarity_score, f"Candidate {i+1}")
        results.append(chart_path)
    return results

# ---- New Functions for Parsing and Visualizing Requirements ----



def parse_summary_for_requirements(summary_text):
    # Try to parse candidate sections in both English and Norwegian:
    # English: ### Candidate X
    # Norwegian: ### Kandidat X
    # We'll match either 'Candidate' or 'Kandidat'
    candidate_sections = re.split(r"(?=###\s*(Candidate|Kandidat))", summary_text, flags=re.IGNORECASE)

    candidate_data = {}
    print("Parsing summary for requirements...")
    for section in candidate_sections:        section = section.strip()
        if not section.startswith("###"):
            continue

        # Extract candidate name/number
        cand_match = re.match(r"###\s*(Candidate|Kandidat)\s+(\d+)", section, re.IGNORECASE)
        if cand_match:
            candidate_num = cand_match.group(2)
            candidate_name = f"Candidate {candidate_num}"
        else:
            # If no number found, try a less strict approach:
            cand_match_loose = re.match(r"###\s*(Candidate|Kandidat)\s*(.*)", section, re.IGNORECASE)
            if cand_match_loose and cand_match_loose.group(2).strip():
                candidate_name = "Candidate " + cand_match_loose.group(2).strip()
            else:
                candidate_name = "Unknown Candidate"

        # Look for compliance section in English or Norwegian
        # English: Compliance with requirements:
        # Norwegian: Samsvar med krav:
        comp_regex = r"(Compliance with requirements:|Samsvar med krav:)(.*?)(?:\n###|\nFinal Conclusion|\Z)"
        comp_section_match = re.search(comp_regex, section, flags=re.IGNORECASE | re.DOTALL)
        if not comp_section_match:
            print(f"No compliance section found for {candidate_name}")
            continue

        comp_section = comp_section_match.group(2).strip()
        lines = comp_section.split("\n")
        requirements = []
        
        # Each requirement line usually starts with '- '
        # Format: "- Requirement: ✓ justification"
        # or "- Krav: ✓ begrunnelse"
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                line = line[2:].strip()
                req_match = re.match(r"(.*?):\s*(✓|✗)", line)
                if req_match:
                    req_name = req_match.group(1).strip()
                    req_mark = req_match.group(2)
                    compliance = 1 if req_mark == "✓" else 0
                    requirements.append((req_name, compliance))

        print(f"Parsed for {candidate_name}: {requirements}")
        candidate_data[candidate_name] = requirements

    if not candidate_data:
        print("No candidates or requirements were parsed. Check the LLM output format.")
    return candidate_data

def generate_compliance_charts(candidate_data):
    chart_paths = []
    for candidate, reqs in candidate_data.items():
        if not reqs:
            continue

        # Minimalistic visualization:
        # A single column of points (green = met, red = not met) with "Met"/"Not Met"
        req_names = [r[0] for r in reqs]
        compliances = [r[1] for r in reqs]

        fig, ax = plt.subplots(figsize=(6, max(len(reqs)*0.4, 2)))
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
        file_path = os.path.join(os.getcwd(), f'{candidate}_compliance_chart.png')
        plt.savefig(file_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        chart_paths.append(file_path)

    return chart_paths

def generate_summary_and_charts(cv_files, project_file):
    # This function assumes you have a function process_and_generate_summary 
    # that returns the LLM summary text after analyzing the CVs and project.
    summary = process_and_generate_summary(cv_files, project_file)

    candidate_requirements = parse_summary_for_requirements(summary)
    if not candidate_requirements:
        # If no data parsed, return just the summary
        return summary, []
    charts = generate_compliance_charts(candidate_requirements)
    return summary, charts





# Set up Gradio interface to handle file uploads, interact with the model, and display results
with gr.Blocks() as interface:
    gr.Markdown(
        """
        <div style="text-align: center; font-size: 2.5em; font-weight: bold; color: #1842a8; padding: 20px; font-family: Arial, sans-serif;">
            POC Demo
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            cv_uploads = gr.File(label = "Upload your CVs", file_types =[".docx",".pdf", ".txt" ], file_count="multiple" ) 
        with gr.Column():
            project_upload = gr.File(label="Upload Project Description", file_types=[".txt", ".docx", ".pdf"])
    
    match_button = gr.Button("Generate Similarity Charts")
    chart_outputs = gr.Gallery(label="Similarity Charts", columns=[3], rows=[1])  # Gallery for similarity charts
          
    summary_button = gr.Button("Generate Summary and Dashboard")
    summary_output = gr.Markdown(label = "Summary of Documents")
    compliance_charts_output = gr.Gallery(label="Compliance Dashboard", columns=2, rows=2)
   
    match_button.click(
        match_cv_to_project_and_display_charts,
        inputs=[cv_uploads, project_upload],
        outputs=chart_outputs
    )
    
    summary_button.click(generate_summary_and_charts,
        inputs=[cv_uploads, project_upload],
        outputs=[summary_output, compliance_charts_output], 
        show_progress="full")

if __name__ == "__main__":
    interface.launch()
