import os
import zipfile
import shutil
import glob
import pickle
import random
import gradio as gr
import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from rapidfuzz import fuzz

# --- Knowledge Base (expand as needed) ---
knowledge_base = [
    # --- CHARACTERS ---
    {"name": "Harry Potter", "synonyms": ["The Boy Who Lived"], "description": "The main protagonist, known for his lightning bolt scar. Member of Gryffindor House."},
    {"name": "Hermione Granger", "synonyms": ["Brightest witch", "Hermione"], "description": "Brilliant Muggle-born witch, best friend to Harry and Ron. Gryffindor."},
    {"name": "Ron Weasley", "synonyms": ["Harry's best friend", "Ron"], "description": "Youngest son of the Weasley family, loyal friend, Gryffindor."},
    {"name": "Albus Dumbledore", "synonyms": ["Headmaster"], "description": "Headmaster of Hogwarts, founder of the Order of the Phoenix, one of the greatest wizards."},
    {"name": "Severus Snape", "synonyms": ["Half-Blood Prince", "Professor Snape"], "description": "Potions Master and later Headmaster at Hogwarts, double agent for Dumbledore."},
    {"name": "Sirius Black", "synonyms": ["Padfoot"], "description": "Harry's godfather, Animagus, member of the Marauders and Order of the Phoenix."},
    {"name": "Draco Malfoy", "synonyms": ["Draco"], "description": "Slytherin student, Harry's rival at Hogwarts."},
    {"name": "Lord Voldemort", "synonyms": ["Tom Riddle", "He Who Must Not Be Named"], "description": "Dark wizard, leader of the Death Eaters, main antagonist."},
    {"name": "Minerva McGonagall", "synonyms": ["Professor McGonagall"], "description": "Deputy Headmistress, Transfiguration teacher, Head of Gryffindor House."},
    {"name": "Rubeus Hagrid", "synonyms": ["Hagrid"], "description": "Keeper of Keys and Grounds at Hogwarts, Care of Magical Creatures professor."},
    {"name": "Neville Longbottom", "synonyms": ["Neville"], "description": "Gryffindor student, member of Dumbledore's Army, known for his bravery."},
    {"name": "Ginny Weasley", "synonyms": ["Ginny"], "description": "Youngest Weasley child, Gryffindor, excellent Quidditch player."},
    {"name": "Luna Lovegood", "synonyms": ["Luna"], "description": "Ravenclaw student known for her eccentricity and loyalty."},
    {"name": "Fred Weasley", "synonyms": ["Fred"], "description": "One of the Weasley twins, known for mischief and inventions."},
    {"name": "George Weasley", "synonyms": ["George"], "description": "One of the Weasley twins, co-founder of Weasleys' Wizard Wheezes."},
    {"name": "Molly Weasley", "synonyms": ["Mrs Weasley"], "description": "Matriarch of the Weasley family, known for her caring nature."},
    {"name": "Arthur Weasley", "synonyms": ["Mr Weasley"], "description": "Patriarch of the Weasley family, works at the Ministry of Magic."},
    {"name": "Bellatrix Lestrange", "synonyms": ["Bellatrix"], "description": "Fiercely loyal Death Eater, cousin to Sirius Black."},
    {"name": "Remus Lupin", "synonyms": ["Moony"], "description": "Werewolf, Defense Against the Dark Arts teacher, member of the Marauders."},
    {"name": "Peter Pettigrew", "synonyms": ["Wormtail"], "description": "Animagus, Marauder who betrayed Harry's parents to Voldemort."},

    # --- SPELLS ---
    {"name": "Expelliarmus", "synonyms": ["Disarm"], "description": "Disarming Charm. Incantation: Expelliarmus. Disarms the opponent."},
    {"name": "Avada Kedavra", "synonyms": ["Killing Curse"], "description": "The Killing Curse. Causes instant death. One of the Unforgivable Curses."},
    {"name": "Stupefy", "synonyms": ["Stunning spell"], "description": "Stunning Spell. Incantation: Stupefy. Renders the target unconscious."},
    {"name": "Lumos", "synonyms": ["Light spell"], "description": "Creates light at the tip of the caster's wand."},
    {"name": "Alohomora", "synonyms": ["Unlocking Charm"], "description": "Unlocks doors and windows."},
    {"name": "Wingardium Leviosa", "synonyms": ["Levitation Charm"], "description": "Levitates objects."},
    {"name": "Expecto Patronum", "synonyms": ["Patronus Charm"], "description": "Summons a Patronus to drive away Dementors."},
    {"name": "Protego", "synonyms": ["Shield Charm"], "description": "Creates a magical shield to block spells."},
    {"name": "Crucio", "synonyms": ["Cruciatus Curse"], "description": "Causes unbearable pain. One of the Unforgivable Curses."},
    {"name": "Imperio", "synonyms": ["Imperius Curse"], "description": "Controls the victim's actions. One of the Unforgivable Curses."},
    {"name": "Accio", "synonyms": ["Summoning Charm"], "description": "Summons objects to the caster."},
    {"name": "Obliviate", "synonyms": ["Memory Charm"], "description": "Erases specific memories."},
    {"name": "Sectumsempra", "synonyms": ["Slashing Curse"], "description": "Causes deep gashes on the target."},
    {"name": "Riddikulus", "synonyms": ["Boggart banisher"], "description": "Transforms a Boggart into something humorous."},
    {"name": "Petrificus Totalus", "synonyms": ["Full Body-Bind Curse"], "description": "Temporarily paralyzes the victim."},
    {"name": "Obscuro", "synonyms": ["Blindfolding Charm"], "description": "Conjures a blindfold over the victim's eyes."},
    {"name": "Incendio", "synonyms": ["Fire-Making Spell"], "description": "Produces fire."},
    {"name": "Aguamenti", "synonyms": ["Water-Making Spell"], "description": "Produces water from the caster's wand."},

    # --- POTIONS ---
    {"name": "Polyjuice Potion", "synonyms": ["Shape-shifting potion"], "description": "Allows the drinker to assume the form of another person."},
    {"name": "Felix Felicis", "synonyms": ["Liquid Luck"], "description": "A potion that makes the drinker lucky for a period of time."},
    {"name": "Amortentia", "synonyms": ["Love Potion"], "description": "The most powerful love potion in existence."},
    {"name": "Veritaserum", "synonyms": ["Truth Serum"], "description": "Forces the drinker to tell the truth."},
    {"name": "Draught of Living Death", "synonyms": ["Sleeping Draught"], "description": "Causes the drinker to fall into a deep, almost irreversible sleep."},
    {"name": "Wolfsbane Potion", "synonyms": ["Werewolf potion"], "description": "Allows a werewolf to retain their mind after transformation."},
    {"name": "Skele-Gro", "synonyms": ["Bone-Growing Potion"], "description": "Regrows bones."},
    {"name": "Pepperup Potion", "synonyms": ["Pepperup"], "description": "Cures the common cold and relieves minor illnesses."},

    # --- MAGICAL OBJECTS ---
    {"name": "Horcrux", "synonyms": ["soul fragment"], "description": "A dark magical object containing part of a wizard's soul to attain immortality."},
    {"name": "Elder Wand", "synonyms": ["Deathly Hallows wand"], "description": "The most powerful wand ever made, one of the Deathly Hallows."},
    {"name": "Invisibility Cloak", "synonyms": ["Deathly Hallows cloak"], "description": "Makes the wearer invisible. One of the Deathly Hallows."},
    {"name": "Resurrection Stone", "synonyms": ["Deathly Hallows stone"], "description": "Supposedly brings back the dead. One of the Deathly Hallows."},
    {"name": "Marauder's Map", "synonyms": ["magical map"], "description": "A magical map of Hogwarts showing everyone's location."},
    {"name": "Time-Turner", "synonyms": ["time travel device"], "description": "A device used to travel back in time."},
    {"name": "Sorcerer's Stone", "synonyms": ["Philosopher's Stone"], "description": "Legendary alchemical stone that grants immortality and turns metal into gold."},
    {"name": "Deluminator", "synonyms": ["Put-Outer"], "description": "Device invented by Dumbledore to collect and release light."},
    {"name": "Remembrall", "synonyms": ["Memory ball"], "description": "A glass ball that glows red when the owner has forgotten something."},

    # --- CREATURES ---
    {"name": "Dementor", "synonyms": ["Azkaban guard"], "description": "Dark, soul-sucking creatures that guard Azkaban prison."},
    {"name": "Basilisk", "synonyms": ["giant serpent"], "description": "A giant serpent whose gaze is instantly fatal."},
    {"name": "Hippogriff", "synonyms": ["Buckbeak"], "description": "A magical creature with the front half of an eagle and the hind half of a horse."},
    {"name": "Thestral", "synonyms": ["invisible horse"], "description": "Winged horses that can only be seen by those who have witnessed death."},
    {"name": "Phoenix", "synonyms": ["Fawkes"], "description": "Magical bird that can regenerate, known for its healing tears and loyalty."},
    {"name": "Acromantula", "synonyms": ["giant spider"], "description": "Giant talking spider, such as Aragog."},
    {"name": "House-elf", "synonyms": ["Dobby", "Kreacher"], "description": "Magical creatures bound to serve wizarding families."},

    # --- LOCATIONS ---
    {"name": "Hogwarts", "synonyms": ["school"], "description": "School of Witchcraft and Wizardry in Scotland."},
    {"name": "Hogsmeade", "synonyms": ["wizarding village"], "description": "The only all-wizarding village in Britain, near Hogwarts."},
    {"name": "Diagon Alley", "synonyms": ["wizard market"], "description": "A hidden street in London where witches and wizards shop."},
    {"name": "The Burrow", "synonyms": ["Weasley home"], "description": "Family home of the Weasley family."},
    {"name": "Azkaban", "synonyms": ["wizard prison"], "description": "Wizarding prison guarded by Dementors."},
    {"name": "Godric's Hollow", "synonyms": ["village"], "description": "Birthplace of Godric Gryffindor and where Harry's parents died."},
    {"name": "Forbidden Forest", "synonyms": ["Dark Forest"], "description": "Dangerous forest on the Hogwarts grounds, home to many magical creatures."},

    # --- ORGANIZATIONS / GROUPS ---
    {"name": "Order of the Phoenix", "synonyms": ["Dumbledore's order"], "description": "Secret society founded by Dumbledore to fight Voldemort."},
    {"name": "Dumbledore's Army", "synonyms": ["DA"], "description": "Student group led by Harry to teach Defense Against the Dark Arts."},
    {"name": "The Marauders", "synonyms": ["Moony, Wormtail, Padfoot, and Prongs"], "description": "Group of four friends at Hogwarts: Lupin, Pettigrew, Sirius, and James Potter."},
    {"name": "Death Eaters", "synonyms": ["Voldemort's followers"], "description": "Followers of Lord Voldemort."},

    # --- GAMES / SPORTS ---
    {"name": "Quidditch", "synonyms": ["broomstick sport"], "description": "The most popular wizarding sport, played on broomsticks."},
    {"name": "Wizard Chess", "synonyms": ["magical chess"], "description": "A chess game in which the pieces move on their own and can destroy each other."},
    {"name": "Gobstones", "synonyms": ["wizard marble game"], "description": "A game similar to marbles, but the stones squirt a foul-smelling liquid at the loser."},
    {"name": "Exploding Snap", "synonyms": ["card game"], "description": "A wizarding card game known for its unpredictably exploding cards."},

    # --- BOOK SUMMARIES ---
    {"name": "Philosopher's Stone", "synonyms": ["Book 1", "Sorcerer's Stone", "first book"], "description": "Harry discovers he's a wizard, attends Hogwarts, and stops Voldemort from stealing the Philosopher's Stone."},
    {"name": "Chamber of Secrets", "synonyms": ["Book 2", "second book"], "description": "Harry returns to Hogwarts, defeats the Basilisk, and saves Ginny Weasley."},
    {"name": "Prisoner of Azkaban", "synonyms": ["Book 3", "third book"], "description": "Harry learns the truth about Sirius Black and uses a Time-Turner to save him."},
    {"name": "Goblet of Fire", "synonyms": ["Book 4", "fourth book"], "description": "Harry is forced into the Triwizard Tournament and witnesses Voldemort's return."},
    {"name": "Order of the Phoenix", "synonyms": ["Book 5", "fifth book"], "description": "Harry forms Dumbledore's Army, battles the Ministry, and loses Sirius Black."},
    {"name": "Half-Blood Prince", "synonyms": ["Book 6", "sixth book"], "description": "Harry learns about Horcruxes and witnesses Dumbledore's death."},
    {"name": "Deathly Hallows", "synonyms": ["Book 7", "seventh book"], "description": "Harry, Ron, and Hermione hunt Horcruxes and defeat Voldemort."},
]


# --- Example Questions (all must be in KB for instant answer) ---
example_qa_pairs = [
    {"question": "Who is Harry Potter?", "kb_name": "Harry Potter"},
    {"question": "Who is Hermione Granger?", "kb_name": "Hermione Granger"},
    {"question": "What is Expelliarmus?", "kb_name": "Expelliarmus"},
    {"question": "What is Avada Kedavra?", "kb_name": "Avada Kedavra"},
    {"question": "What is Polyjuice Potion?", "kb_name": "Polyjuice Potion"},
    {"question": "What is a Horcrux?", "kb_name": "Horcrux"},
    {"question": "What is Felix Felicis?", "kb_name": "Felix Felicis"},
    {"question": "What is the Elder Wand?", "kb_name": "Elder Wand"},
    {"question": "Summarize Book 1", "kb_name": "Philosopher's Stone"},
    {"question": "What are the Deathly Hallows?", "kb_name": "Deathly Hallows"},
]

# --- Random facts ---
random_facts = [
    "Hogwarts has 142 staircases.",
    "Dumbledore's full name is Albus Percival Wulfric Brian Dumbledore.",
    "Hermione's Patronus is an otter.",
    "The Sorting Hat originally belonged to Godric Gryffindor.",
    "Fawkes the phoenix saved Harry twice.",
    "The Weasley twins' birthday is April Fool's Day.",
    "The Hogwarts motto means 'Never tickle a sleeping dragon.'",
    "Harry's wand and Voldemort's wand share the same phoenix feather core.",
    "Platform 9¾ is invisible to Muggles at King's Cross Station.",
    "Azkaban is guarded by Dementors who feed on despair.",
]

# --- Fuzzy KB search (priority 1) ---
def search_kb_fuzzy(query):
    q = query.strip().lower()
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

# --- File upload/Book extraction (Colab) ---
def extract_corpus():
    from google.colab import files
    print("⬆️ Please upload your .zip or .txt Harry Potter books:")
    uploaded = files.upload()
    path = "hp_books"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    all_text = ""
    for fn in uploaded:
        if fn.endswith(".zip"):
            with zipfile.ZipFile(fn, "r") as zip_ref:
                zip_ref.extractall(path)
        elif fn.endswith(".txt"):
            shutil.copy(fn, os.path.join(path, fn))
    for fname in glob.glob(os.path.join(path, "*.txt")):
        with open(fname, encoding="utf-8", errors="ignore") as f:
            all_text += f.read() + "\n"
    return all_text

# --- Chunking ---
def chunk_text(text, chunk_size=3200, overlap=500):
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sentence in tqdm(sentences, desc="Chunking text"):
        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk: chunks.append(chunk.strip())
    return [" ".join(chunks[max(0, i-1):i+1]) for i in range(len(chunks))]

# --- Embedding & Index ---
EMBED_PATH, FAISS_PATH = "hp_chunks.pkl", "hp_faiss.idx"
def prepare_embeddings_and_index(all_text):
    # Try/catch: if files are corrupt, force re-processing
    try:
        if os.path.exists(EMBED_PATH) and os.path.exists(FAISS_PATH):
            with open(EMBED_PATH, "rb") as f:
                chunks, chunk_embeddings = pickle.load(f)
            index = faiss.read_index(FAISS_PATH)
        else:
            raise Exception("No index!")
    except Exception:
        chunks = chunk_text(all_text)
        embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        chunk_embeddings = []
        for i in tqdm(range(0, len(chunks), 128), desc="Embedding chunks"):
            chunk_embeddings.extend(embedder.encode(chunks[i:i+128], show_progress_bar=False))
        chunk_embeddings = torch.stack([torch.tensor(x) for x in chunk_embeddings]).cpu().numpy()
        index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        index.add(chunk_embeddings)
        with open(EMBED_PATH, "wb") as f:
            pickle.dump((chunks, chunk_embeddings), f)
        faiss.write_index(index, FAISS_PATH)
    embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return chunks, embedder, index

# --- LLM pipeline ---
def get_llm_pipeline():
    return pipeline(
        "text-generation",
        model="google/gemma-2b-it",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95,
    )

# --- Chatbot answer logic: KB first, then embeddings, then LLM ---
def answer_question(question):
    kb_entry = search_kb_fuzzy(question)
    if kb_entry:
        return kb_entry["description"]
    # 2. Embedding/vector search
    try:
        q_emb = embedder.encode([question])
        D, I = index.search(q_emb, 3)
        context = "\n---\n".join([chunks[i] for i in I[0] if i < len(chunks)])
        if context.strip():
            prompt = (
                "You are a Harry Potter expert. Use the following book context to answer the user's question as accurately as possible. "
                "If the answer is not in the context, answer using your best knowledge of the Harry Potter universe.\n\n"
                f"Context:\n{context}\n\n"
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
                if answer and "not found" not in answer.lower():
                    return answer
            except Exception:
                pass
    except Exception:
        pass
    # 3. LLM fallback
    try:
        prompt = (
            "You are a helpful and creative Harry Potter assistant. "
            "Answer in detail and stay true to canon. If you don't know, say so honestly.\n\n"
            f"Question: {question}\nAnswer:"
        )
        result = qa_pipeline(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )[0]["generated_text"]
        return result.split('Answer:')[-1].strip()
    except Exception:
        return "Sorry, I couldn't generate an answer. Please try again."

# --- Gradio UI ---
def show_random_fact():
    return f"✨ {random.choice(random_facts)}"

def respond(message, chat_state):
    answer = answer_question(message)
    chat_state = chat_state + [(message, answer)]
    return chat_state, chat_state

# --- MAIN SETUP (run once per session) ---
all_text = extract_corpus()
chunks, embedder, index = prepare_embeddings_and_index(all_text)
qa_pipeline = get_llm_pipeline()

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
    chatbot = gr.Chatbot()
    state_chat = gr.State([])

    def fill_input_from_example(qtext):
        return qtext or ""

    random_fact_btn.click(show_random_fact, None, random_fact_box)
    btn.click(respond, [msg, state_chat], [chatbot, state_chat])
    ex_q_dropdown.change(fill_input_from_example, ex_q_dropdown, msg)

demo.launch(share=True)
