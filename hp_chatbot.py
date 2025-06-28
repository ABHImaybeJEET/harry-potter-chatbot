import os
import pickle
import random
import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from rapidfuzz import fuzz
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

BOOKS_FOLDER = '/content/drive/My Drive/hp_books/'

# Import knowledge base, example Q&A pairs, and random facts from a separate file in your repo
from hp_data import knowledge_base, example_qa_pairs, random_facts

example_q_set = set(q['question'].strip().lower() for q in example_qa_pairs)

def search_kb_fuzzy(query):
    q = query.strip().lower()
    if q in example_q_set:
        for ex in example_qa_pairs:
            if ex['question'].strip().lower() == q:
                kb_name = ex['kb_name']
                for entry in knowledge_base:
                    if entry['name'].lower() == kb_name.lower():
                        return entry
    best_entry, best_score = None, 0
    for entry in knowledge_base:
        for name in [entry["name"].lower()] + [s.lower() for s in entry.get("synonyms",[])]:
            score = fuzz.token_sort_ratio(q, name)
            if score > best_score:
                best_score = score
                best_entry = entry
    if best_score >= 80:
        return best_entry
    for entry in knowledge_base:
        if q in entry["description"].lower():
            return entry
    return None

def extract_corpus():
    files = [os.path.join(BOOKS_FOLDER, fname) for fname in os.listdir(BOOKS_FOLDER) if fname.endswith('.txt')]
    all_text = ""
    for fname in files:
        with open(fname, encoding="utf-8", errors="ignore") as f:
            all_text += f.read() + "\n"
    return all_text

def chunk_text(text, chunk_size=3200, overlap=500):
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk: chunks.append(chunk.strip())
    result = []
    for i in range(len(chunks)):
        combined = chunks[i]
        if i > 0 and overlap > 0:
            combined = chunks[i-1][-overlap:] + combined
        result.append(combined)
    return result

EMBED_PATH = os.path.join(BOOKS_FOLDER, "hp_chunks.pkl")
FAISS_PATH = os.path.join(BOOKS_FOLDER, "hp_faiss.idx")

def prepare_embeddings_and_index(all_text):
    if os.path.exists(EMBED_PATH) and os.path.exists(FAISS_PATH):
        with open(EMBED_PATH, "rb") as f:
            chunks, chunk_embeddings = pickle.load(f)
        index = faiss.read_index(FAISS_PATH)
    else:
        chunks = chunk_text(all_text)
        embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        chunk_embeddings = embedder.encode(chunks, show_progress_bar=True, batch_size=32)
        index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        index.add(np.array(chunk_embeddings).astype('float32'))
        with open(EMBED_PATH, "wb") as f:
            pickle.dump((chunks, chunk_embeddings), f)
        faiss.write_index(index, FAISS_PATH)
    embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return chunks, embedder, index

def get_llm_pipeline():
    return pipeline(
        "text-generation",
        model="google/gemma-2b-it",
        max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95,
        device=0 if hasattr(gr, "is_colab") and gr.is_colab() else -1
    )

all_text = extract_corpus()
chunks, embedder, index = prepare_embeddings_and_index(all_text)
qa_pipeline = get_llm_pipeline()

def answer_question(question):
    kb_entry = search_kb_fuzzy(question)
    if kb_entry:
        cat = kb_entry.get('category','Fact')
        return f"**{cat}: {kb_entry['name']}**\n\n{kb_entry['description']}"
    q_emb = embedder.encode([question])
    D, I = index.search(np.array(q_emb).astype('float32'), 3)
    context_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    context = "\n---\n".join(context_chunks)
    if context.strip():
        prompt = (
            "You are a Harry Potter expert. Use the following book context to answer the user's question as accurately as possible. "
            "If the answer is not in the context, answer using your best knowledge of the Harry Potter universe.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
    else:
        prompt = (
            "You are a helpful and creative Harry Potter assistant. "
            "Answer in detail and stay true to canon. If you don't know, say so honestly.\n\n"
            f"Question: {question}\nAnswer:"
        )
    try:
        result = qa_pipeline(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )[0]["generated_text"]
        answer = result.split("Answer:")[-1].strip()
        if answer:
            return answer
    except Exception:
        pass
    return "Sorry, I couldn't generate an answer. Please try again."

def show_random_fact():
    return f"✨ {random.choice(random_facts)}"

def respond(message, chat_state):
    answer = answer_question(message)
    chat_state = chat_state + [(message, answer)]
    return chat_state, chat_state

def clear_history():
    return [], []

custom_theme = gr.themes.Base(
    primary_hue="purple", secondary_hue="purple", neutral_hue="gray",
    font=[gr.themes.GoogleFont("Quicksand"), "Arial", "sans-serif"],
    font_mono=["JetBrains Mono", "monospace"],
)
css = """
body { background: #f7f3fa; }
h1, h2, h3, .gr-markdown { color: #6e39a9 !important; }
.gr-button { background: #6e39a9 !important; color: #fff !important; border-radius: 8px; }
.gr-dropdown, .gr-textbox { border-radius: 8px !important; }
"""

with gr.Blocks(theme=custom_theme, css=css, title="🧙 Harry Potter Ultimate Chatbot") as demo:
    gr.Markdown("## 🧙 Harry Potter Ultimate Chatbot")

    with gr.Row():
        random_fact_btn = gr.Button("🔮 Get Random Fact")
        random_fact_box = gr.Markdown("")

    gr.Markdown("### ✨ Example Questions (click to paste below and chat!):")
    with gr.Column():
        ex_q_dropdown = gr.Dropdown(
            choices=[pair['question'] for pair in example_qa_pairs],
            label="Example Questions"
        )
    with gr.Row():
        msg = gr.Textbox(label="Your Question (or any HP topic!)", lines=1)
        btn = gr.Button("Ask")
        clear_btn = gr.Button("🧹 Clear Chat")
    chatbot = gr.Chatbot()
    state_chat = gr.State([])

    def fill_input_from_example(qtext):
        return qtext or ""

    random_fact_btn.click(show_random_fact, None, random_fact_box)
    btn.click(respond, [msg, state_chat], [chatbot, state_chat])
    ex_q_dropdown.change(fill_input_from_example, ex_q_dropdown, msg)
    clear_btn.click(clear_history, None, [chatbot, state_chat])

demo.launch(share=True)
