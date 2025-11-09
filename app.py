# app.py
import os
import json
import re
import tempfile
import ssl
import io
import shutil
import requests

import streamlit as st
import fitz
import pytesseract
from pdf2image import convert_from_path
import nltk
import boto3

# Optional graph visualization libs
import networkx as nx
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, ClientError

# ---------------------------
# Load local .env only if present
# ---------------------------
if os.path.exists(".env"):
    load_dotenv()

# ---------------------------
# Tesseract (OCR) availability check
# ---------------------------
TESSERACT_AVAILABLE = bool(shutil.which("tesseract"))
if TESSERACT_AVAILABLE:
    pass
else:
    st.warning("Tesseract binary not found — OCR is disabled. To enable OCR install Tesseract and Poppler.")

# ---------------------------
# Ensure NLTK data (only download if not present)
# ---------------------------
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab")
except Exception:
    nltk.download("punkt_tab")
try:
    nltk.data.find("corpora/wordnet")
except Exception:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# ---------------------------
# AWS Bedrock client setup
# ---------------------------
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

bedrock = None
if aws_access_key_id and aws_secret_access_key:
    try:
        bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
    except Exception as e:
        st.error(f"Error creating Bedrock client: {e}")
else:
    st.info("AWS credentials not found in environment. Bedrock calls will be disabled until credentials are provided.")

# ---------------------------
# Gemini (Generative) LLM config (fallback)
# ---------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # put your Gemimi/Generative API key in .env
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "text-bison-001")  # adjust model name as needed

def call_gemini_api(prompt_text, max_output_tokens=300, temperature=0.7):
    """
    Call Google Generative Language REST API (best-effort).
    Requirements:
      - Set GEMINI_API_KEY in .env or environment variables.
      - Adjust GEMINI_MODEL if you use a different model name.
    Note: The exact response fields may differ depending on the GenAI API version.
    Adjust parsing if needed.
    """
    if not GEMINI_API_KEY:
        return None
    try:
        # Example REST endpoint pattern for Google Generative API with API key
        url = f"https://generativelanguage.googleapis.com/v1beta2/models/{GEMINI_MODEL}:generateText"
        params = {"key": GEMINI_API_KEY}
        payload = {
            "prompt": {"text": prompt_text},
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
        resp = requests.post(url, params=params, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Typical field: 'candidates'[0]['content'] or 'output' depending on API version
        if "candidates" in data and isinstance(data["candidates"], list) and len(data["candidates"]) > 0:
            return data["candidates"][0].get("content") or data["candidates"][0].get("text") or None
        # fallback: some endpoints return 'output' or 'text'
        if "output" in data:
            return data["output"]
        if "text" in data:
            return data["text"]
        # last resort: stringify entire response
        return json.dumps(data)
    except Exception as e:
        st.warning(f"Gemini API call failed: {e}")
        return None

# ---------------------------
# Utility: convert PDF to text
# ---------------------------
def convert_pdf_to_txt(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error converting PDF to text: {e}")
        return ""

# ---------------------------
# Utility: OCR (images -> text)
# ---------------------------
def extract_text_from_images(images):
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.warning(f"Error in OCR: {e}")
        return ""

# ---------------------------
# Preprocess text
# ---------------------------
def preprocess_text(text):
    try:
        def clean_text(txt):
            txt = re.sub(r"[^\x00-\x7F]+", " ", txt)
            txt = re.sub(r"\s+", " ", txt).strip()
            return txt
        cleaned_text = clean_text(text)
        sentences = nltk.tokenize.sent_tokenize(cleaned_text)
        return sentences
    except Exception as e:
        st.error(f"Error in preprocessing text: {e}")
        return []

# ---------------------------
# Chunking
# ---------------------------
def chunk_text(text, max_chunk_size=4000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    for word in words:
        if current_size + len(word) + 1 > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ---------------------------
# Bedrock wrapper
# ---------------------------
def generate_response(prompt, model_id="mistral.mistral-7b-instruct-v0:2", max_tokens=512):
    if bedrock is None:
        return None
    try:
        chunks = chunk_text(prompt)
        responses = []
        for chunk in chunks:
            body = json.dumps(
                {
                    "prompt": chunk,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            response = bedrock.invoke_model(body=body, modelId=model_id)
            response_body = json.loads(response.get("body").read())
            outputs = response_body.get("outputs") or []
            out_text = outputs[0].get("text", "") if outputs else response_body.get("text") or ""
            responses.append(out_text)
        return " ".join(responses)
    except NoCredentialsError:
        st.error("AWS credentials not found for Bedrock.")
        return None
    except ClientError as e:
        st.error(f"AWS ClientError: {e}")
        return None
    except Exception as e:
        st.error(f"Error generating response from Bedrock: {e}")
        return None

# ---------------------------
# Simple detection if model refused/doesn't have answer in document
# ---------------------------
_no_answer_patterns = [
    r"not mentioned",
    r"not present",
    r"unable to provide",
    r"cannot provide",
    r"i do not see",
    r"i'm unable",
    r"can't find",
    r"no information",
    r"not available",
    r"cannot find",
    r"couldn't find",
    r"does not provide",       
    r"doesn't provide",
    r"does not contain",
    r"no.*information about",
    r"no details about",
    r"no mention",
    r"not included",
    r"not available in the document",
]
_no_answer_re = re.compile("|".join(_no_answer_patterns), re.IGNORECASE)

def model_says_no_answer(text):
    """
    Return True if the model's response is clearly refusing / saying the doc lacks info.
    Also treat very short generic responses (<= 6 words) or responses with many negation tokens as no-answer.
    """
    if not text:
        return True

    # direct pattern match
    if _no_answer_re.search(text):
        return True

    # short/low-information responses -> trigger fallback
    tokens = text.strip().split()
    if len(tokens) <= 6:
        # if it is a question-like or 'I don't know' style, treat as no answer
        low = text.strip().lower()
        if any(phr in low for phr in ("don't", "do not", "no", "cannot", "can't", "unable", "not sure", "unknown")):
            return True
        # also if extremely short, trigger fallback
        if len(tokens) <= 3:
            return True

    # heuristic: many negation words
    neg_count = sum(1 for w in re.findall(r"\b(not|no|cannot|can't|unable|don't|doesn't|didn't|none)\b", text.lower()))
    if neg_count >= 2:
        return True

    return False

# ---------------------------
# 1) KG: Ask LLM for structured entities & relations (JSON)
#    (kept simple; consider using robust version if you have parsing issues)
# ---------------------------
def ask_for_entities_and_relations(text, model_id="mistral.mistral-7b-instruct-v0:2"):
    if bedrock is None:
        st.warning("Bedrock client is not configured — cannot extract structured entities.")
        return None

    prompt = f"""
Return ONLY valid JSON. Analyze the following text (invoice/transaction) and output:
{{
    "entities": [{{"id":"E1","text":"...","type":"...","attrs":{{}}}}]],
    "relations": [{{"source":"E1","target":"E2","type":"...","attrs":{{}}}}]
}}
Do not include any commentary or text outside JSON.

Text:
\"\"\"{text}\"\"\"
"""
    raw = generate_response(prompt, model_id=model_id, max_tokens=512)
    if not raw:
        return None

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == -1:
        st.warning("No JSON braces found in LLM output.")
        st.code(raw[:1000])
        return None
    json_text = raw[start:end].strip()
    json_text = json_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    json_text = re.sub(r"\s{2,}", " ", json_text)
    json_text = re.sub(r",\s*([\]}])", r"\1", json_text)
    json_text = re.sub(r"([{,]\s*)([A-Za-z0-9_]+)(\s*:)", r'\1"\2"\3', json_text)

    try:
        parsed = json.loads(json_text)
        return parsed
    except Exception as e:
        st.error(f"JSON parse error: {e}")
        return None

# ---------------------------
# 2) Build Knowledge Graph
# ---------------------------
def build_knowledge_graph(parsed):
    G = nx.DiGraph()
    if not parsed:
        return G

    for ent in parsed.get("entities", []):
        ent_id = ent.get("id")
        if not ent_id:
            continue
        attrs = ent.get("attrs", {}) or {}
        attrs.update({"text": ent.get("text", ""), "type": ent.get("type", "")})
        G.add_node(ent_id, **attrs)

    for rel in parsed.get("relations", []):
        src = rel.get("source")
        tgt = rel.get("target")
        if not src or not tgt:
            continue
        rattrs = rel.get("attrs", {}) or {}
        rattrs.update({"type": rel.get("type", "")})
        G.add_edge(src, tgt, **rattrs)

    return G

# ---------------------------
# 3) Draw graph image
# ---------------------------
def draw_graph_image_bytes(G, figsize=(8, 6)):
    if G.number_of_nodes() == 0:
        return None
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, k=0.7, seed=42)
    labels = {n: f"{data.get('text','')}\n({data.get('type','')})" for n, data in G.nodes(data=True)}
    nx.draw_networkx_nodes(G, pos, node_size=1500)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=12)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    edge_labels = {(u, v): data.get("type", "") for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.axis("off")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return buf

# ---------------------------
# Chat fallback flow: Bedrock first, then Gemini general knowledge
# ---------------------------
def answer_with_optional_gemini(question, document_text):
    """
    1) Ask Bedrock to answer using the document primarily.
    2) If Bedrock's reply looks like a refusal or very low info, call Gemini to produce a direct answer.
    Returns the final answer text (direct).
    """
    # If Bedrock isn't configured, use Gemini directly
    if bedrock is None:
        gprompt = f"You are a knowledgeable assistant. Answer the question directly and concisely:\nQuestion: {question}"
        gem = call_gemini_api(gprompt, max_output_tokens=300)
        return gem or "No LLM available to answer."

    # 1) Ask Bedrock (document-first)
    prompt_doc = f"""
You are a helpful assistant. Use the document below to answer the question if possible.
Document:
\"\"\"{document_text}\"\"\"

Question: {question}

Answer concisely and directly. If the document lacks the information, you may respond briefly (do not invent facts).
"""
    resp_doc = generate_response(prompt_doc, max_tokens=256)
    if not resp_doc:
        # no reply from Bedrock -> try Gemini
        gprompt = f"You are a knowledgeable assistant. Answer the question directly and concisely:\nQuestion: {question}"
        gem = call_gemini_api(gprompt, max_output_tokens=300)
        return gem or "No response from LLMs."

    # If Bedrock indicates no-answer (by our detector) -> fallback to Gemini
    if model_says_no_answer(resp_doc):
        # debug note (you can remove later)
        st.info("Document didn't contain the answer — using Gemini to answer.")
        # Use Gemini to answer directly
        gprompt = f"You are a knowledgeable assistant. Use your general knowledge to answer the question directly and concisely:\nQuestion: {question}"
        gem_resp = call_gemini_api(gprompt, max_output_tokens=300)
        if gem_resp:
            return gem_resp
        # if Gemini not available, try asking Bedrock again to use general knowledge
        fallback_prompt = f"You are a knowledgeable assistant. The document did not provide the answer. Use your general knowledge to answer the question directly and concisely:\nQuestion: {question}"
        fallback_resp = generate_response(fallback_prompt, max_tokens=256)
        return fallback_resp or resp_doc or "No suitable response available."

    # Otherwise return the Bedrock answer (document-based)
    return resp_doc

# ---------------------------
# Main Streamlit UI
# ---------------------------
st.title("Intelligent Document Processing + Knowledge Graph (Bedrock + Gemini fallback)")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    text = convert_pdf_to_txt(temp_file_path)
    full_text = text

    try:
        if TESSERACT_AVAILABLE:
            with tempfile.TemporaryDirectory() as tempdir:
                pages = convert_from_path(temp_file_path, dpi=300)
                image_text = extract_text_from_images(pages)
                full_text += "\n" + image_text
        else:
            st.info("OCR skipped because Tesseract is not available on this system.")
    except Exception as e:
        st.warning(f"OCR skipped due to error: {e}")

    preprocessed_text = preprocess_text(full_text)

    # Basic LLM operations
    entities_text = relationships_text = summary_text = classification_text = None
    if bedrock is not None:
        chunk_for_prompt = full_text[:2000]
        entities_text = generate_response(f"Extract entities from the following text: {chunk_for_prompt}", max_tokens=256)
        relationships_text = generate_response(f"Extract relationships between entities from the following text: {chunk_for_prompt}", max_tokens=256)
        summary_text = generate_response(f"Summarize the following text: {chunk_for_prompt}", max_tokens=256)
        categories = ["Personal", "Work", "Legal", "Medical"]
        classification_text = generate_response(f"Classify the following text into one of these categories: {categories}\n\nText: {chunk_for_prompt}", max_tokens=128)
    else:
        entities_text = relationships_text = summary_text = classification_text = "Bedrock not configured."

    # ---------------------------
    # Chatbot (Bedrock first; Gemini fallback)
    # ---------------------------
    st.header("Chat with the Document")
    user_input = st.text_input("Ask a question about the document:")
    if user_input:
        with st.spinner("Generating answer..."):
            answer = answer_with_optional_gemini(user_input, full_text[:8000])
        st.subheader("Answer")
        if answer:
            st.write(answer.strip())
        else:
            st.warning("No response generated.")

    # ---------------------------
    # Results & KG Section
    # ---------------------------
    st.header("Document Insights")
    with st.expander("Extracted Text (preview)"):
        st.write(full_text[:2000] + "..." if len(full_text) > 2000 else full_text)

    with st.expander("Preprocessed Text (first 10 sentences)"):
        st.write(preprocessed_text[:10])

    with st.expander("LLM Results"):
        st.subheader("Entities (raw)")
        st.text(entities_text or "N/A")
        st.subheader("Relationships (raw)")
        st.text(relationships_text or "N/A")
        st.subheader("Summary")
        st.text(summary_text or "N/A")
        st.subheader("Classification")
        st.text(classification_text or "N/A")

    st.header("Knowledge Graph (Entities & Relations)")
    with st.expander("Generate Knowledge Graph from Document"):
        st.write("This will call the LLM to extract structured entities and relations (JSON) and build a small KG.")
        if st.button("Generate Knowledge Graph"):
            with st.spinner("Calling LLM to extract entities & relations..."):
                parsed = ask_for_entities_and_relations(full_text[:4000])
                if not parsed:
                    st.error("Could not extract structured entities/relations from the LLM output.")
                else:
                    st.subheader("Parsed JSON")
                    st.json(parsed)
                    G = build_knowledge_graph(parsed)
                    img_buf = draw_graph_image_bytes(G)
                    if img_buf:
                        st.image(img_buf)
                    else:
                        st.info("No entities found to build a graph.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download KG as JSON",
                            data=json.dumps(parsed, indent=2),
                            file_name="knowledge_graph.json",
                            mime="application/json",
                        )
                    with col2:
                        try:
                            gml_bytes = "\n".join(list(nx.generate_graphml(G)))
                            st.download_button(
                                "Download KG as GraphML",
                                data=gml_bytes,
                                file_name="knowledge_graph.graphml",
                                mime="application/xml",
                            )
                        except Exception as e:
                            st.warning(f"Could not export GraphML: {e}")

    try:
        os.unlink(temp_file_path)
    except Exception:
        pass
