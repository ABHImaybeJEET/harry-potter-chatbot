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

# ---- DATA: Knowledge Base, Example QA, Random Facts ----
knowledge_base = [
    # CHARACTERS
    {"category": "Character", "name": "Harry Potter", "synonyms": ["The Boy Who Lived"], "description": "The main protagonist, known for his lightning bolt scar. Member of Gryffindor House. Raised by the Dursleys. Defeated Lord Voldemort."},
    {"category": "Character", "name": "Hermione Granger", "synonyms": ["Brightest witch", "Hermione"], "description": "Brilliant Muggle-born witch, best friend to Harry and Ron. Known for her intelligence and quick thinking. Member of Gryffindor."},
    {"category": "Character", "name": "Ron Weasley", "synonyms": ["Harry's best friend", "Ron"], "description": "Youngest son of the Weasley family, loyal friend, Gryffindor. Known for his humor and loyalty."},
    {"category": "Character", "name": "Albus Dumbledore", "synonyms": ["Headmaster"], "description": "Headmaster of Hogwarts, founder of the Order of the Phoenix, one of the greatest wizards. Defeated Grindelwald. Wise and compassionate."},
    {"category": "Character", "name": "Lord Voldemort", "synonyms": ["Tom Riddle", "He Who Must Not Be Named"], "description": "Dark wizard, leader of the Death Eaters, main antagonist. Obsessed with immortality and pure-blood supremacy."},
    {"category": "Character", "name": "Severus Snape", "synonyms": ["Half-Blood Prince", "Professor Snape"], "description": "Potions Master and later Headmaster at Hogwarts, double agent for Dumbledore. Complex and brave character."},
    {"category": "Character", "name": "Sirius Black", "synonyms": ["Padfoot"], "description": "Harry's godfather, Animagus, member of the Marauders and Order of the Phoenix. Wrongly imprisoned in Azkaban."},
    {"category": "Character", "name": "Draco Malfoy", "synonyms": ["Draco"], "description": "Slytherin student, Harry's rival, son of Lucius Malfoy. Eventually redeems himself."},
    {"category": "Character", "name": "Minerva McGonagall", "synonyms": ["Professor McGonagall"], "description": "Deputy Headmistress, Transfiguration teacher, Head of Gryffindor House, Animagus (cat)."},
    {"category": "Character", "name": "Rubeus Hagrid", "synonyms": ["Hagrid"], "description": "Keeper of Keys and Grounds at Hogwarts, Care of Magical Creatures professor. Half-giant, friend to Harry."},
    {"category": "Character", "name": "Ginny Weasley", "synonyms": ["Ginevra Weasley"], "description": "Youngest Weasley sibling, talented witch, member of Dumbledore's Army. Marries Harry Potter."},
    {"category": "Character", "name": "Neville Longbottom", "synonyms": ["Neville"], "description": "Gryffindor student, skilled in Herbology, plays a key role in the final battle."},
    {"category": "Character", "name": "Luna Lovegood", "synonyms": ["Luna"], "description": "Ravenclaw student, known for her eccentricity and kindness. Member of Dumbledore's Army."},
    {"category": "Character", "name": "Fred Weasley", "synonyms": ["Fred"], "description": "One of the Weasley twins, known for their jokes and inventions. Co-founder of Weasleys' Wizard Wheezes."},
    {"category": "Character", "name": "George Weasley", "synonyms": ["George"], "description": "One of the Weasley twins, co-founder of Weasleys' Wizard Wheezes. Loyal and creative."},
    {"category": "Character", "name": "Molly Weasley", "synonyms": ["Mrs. Weasley"], "description": "Matriarch of the Weasley family. Defeated Bellatrix Lestrange in the Battle of Hogwarts."},
    {"category": "Character", "name": "Arthur Weasley", "synonyms": ["Mr. Weasley"], "description": "Patriarch of the Weasley family. Works at the Ministry of Magic, fascinated by Muggle artifacts."},
    {"category": "Character", "name": "Remus Lupin", "synonyms": ["Professor Lupin", "Moony"], "description": "Werewolf, Defense Against the Dark Arts teacher, member of the Marauders and Order of the Phoenix."},
    {"category": "Character", "name": "Nymphadora Tonks", "synonyms": ["Tonks"], "description": "Auror, Metamorphmagus, member of the Order of the Phoenix. Marries Remus Lupin."},
    {"category": "Character", "name": "Bellatrix Lestrange", "synonyms": ["Bellatrix"], "description": "Fiercely loyal Death Eater, cousin to Sirius Black. Known for her cruelty."},
    {"category": "Character", "name": "Lucius Malfoy", "synonyms": ["Mr. Malfoy"], "description": "Father of Draco Malfoy, Death Eater, wealthy and influential."},
    {"category": "Character", "name": "Peter Pettigrew", "synonyms": ["Wormtail"], "description": "Animagus (rat), betrays the Potters to Voldemort, member of the Marauders."},
    {"category": "Character", "name": "Cho Chang", "synonyms": ["Cho"], "description": "Ravenclaw student, Harry's first crush, member of Dumbledore's Army."},
    {"category": "Character", "name": "Cedric Diggory", "synonyms": ["Cedric"], "description": "Hufflepuff student, Triwizard Tournament champion, killed by Peter Pettigrew on Voldemort's order."},

    # SPELLS
    {"category": "Spell", "name": "Expelliarmus", "synonyms": ["Disarm"], "description": "Disarming Charm. Incantation: Expelliarmus. Disarms the opponent."},
    {"category": "Spell", "name": "Avada Kedavra", "synonyms": ["Killing Curse"], "description": "The Killing Curse. Causes instant death. One of the Unforgivable Curses."},
    {"category": "Spell", "name": "Stupefy", "synonyms": ["Stunning spell"], "description": "Stunning Spell. Incantation: Stupefy. Renders the target unconscious."},
    {"category": "Spell", "name": "Lumos", "synonyms": ["Light spell"], "description": "Creates light at the tip of the caster's wand."},
    {"category": "Spell", "name": "Alohomora", "synonyms": ["Unlocking Charm"], "description": "Unlocks doors and windows."},
    {"category": "Spell", "name": "Wingardium Leviosa", "synonyms": ["Levitation Charm"], "description": "Makes objects fly or levitate."},
    {"category": "Spell", "name": "Expecto Patronum", "synonyms": ["Patronus Charm"], "description": "Summons a Patronus to drive away Dementors."},
    {"category": "Spell", "name": "Protego", "synonyms": ["Shield Charm"], "description": "Creates a magical shield to block spells."},
    {"category": "Spell", "name": "Crucio", "synonyms": ["Cruciatus Curse"], "description": "Causes unbearable pain. One of the Unforgivable Curses."},
    {"category": "Spell", "name": "Imperio", "synonyms": ["Imperius Curse"], "description": "Controls the victim's actions. One of the Unforgivable Curses."},
    {"category": "Spell", "name": "Accio", "synonyms": ["Summoning Charm"], "description": "Summons objects to the caster."},
    {"category": "Spell", "name": "Obliviate", "synonyms": ["Memory Charm"], "description": "Erases specific memories."},
    {"category": "Spell", "name": "Sectumsempra", "synonyms": ["Slashing Curse"], "description": "Causes deep gashes on the target."},
    {"category": "Spell", "name": "Riddikulus", "synonyms": ["Boggart banisher"], "description": "Transforms a Boggart into something humorous."},
    {"category": "Spell", "name": "Petrificus Totalus", "synonyms": ["Full Body-Bind Curse"], "description": "Temporarily paralyzes the victim."},
    {"category": "Spell", "name": "Incendio", "synonyms": ["Fire-Making Spell"], "description": "Produces fire."},
    {"category": "Spell", "name": "Aguamenti", "synonyms": ["Water-Making Spell"], "description": "Produces water from the caster's wand."},
    {"category": "Spell", "name": "Nox", "synonyms": ["Light Extinguishing Spell"], "description": "Extinguishes light produced by Lumos."},
    {"category": "Spell", "name": "Obscuro", "synonyms": ["Blindfolding Charm"], "description": "Conjures a blindfold over the victim's eyes."},

    # POTIONS
    {"category": "Potion", "name": "Polyjuice Potion", "synonyms": ["Shape-shifting potion"], "description": "Allows the drinker to assume the form of another person."},
    {"category": "Potion", "name": "Felix Felicis", "synonyms": ["Liquid Luck"], "description": "A potion that makes the drinker lucky for a period of time."},
    {"category": "Potion", "name": "Amortentia", "synonyms": ["Love Potion"], "description": "The most powerful love potion in existence."},
    {"category": "Potion", "name": "Veritaserum", "synonyms": ["Truth Serum"], "description": "Forces the drinker to tell the truth."},
    {"category": "Potion", "name": "Draught of Living Death", "synonyms": ["Sleeping Draught"], "description": "Causes the drinker to fall into a deep, almost irreversible sleep."},
    {"category": "Potion", "name": "Wolfsbane Potion", "synonyms": ["Werewolf potion"], "description": "Allows a werewolf to retain their mind after transformation."},
    {"category": "Potion", "name": "Skele-Gro", "synonyms": ["Bone-Growing Potion"], "description": "Regrows bones."},
    {"category": "Potion", "name": "Pepperup Potion", "synonyms": ["Pepperup"], "description": "Cures the common cold and relieves minor illnesses."},

    # OBJECTS
    {"category": "Object", "name": "Horcrux", "synonyms": ["soul fragment"], "description": "A dark magical object containing part of a wizard's soul to attain immortality."},
    {"category": "Object", "name": "Elder Wand", "synonyms": ["Deathly Hallows wand"], "description": "The most powerful wand ever made, one of the Deathly Hallows."},
    {"category": "Object", "name": "Invisibility Cloak", "synonyms": ["Deathly Hallows cloak"], "description": "Makes the wearer invisible. One of the Deathly Hallows."},
    {"category": "Object", "name": "Resurrection Stone", "synonyms": ["Deathly Hallows stone"], "description": "Supposedly brings back the dead. One of the Deathly Hallows."},
    {"category": "Object", "name": "Marauder's Map", "synonyms": ["magical map"], "description": "A magical map of Hogwarts showing everyone's location."},
    {"category": "Object", "name": "Time-Turner", "synonyms": ["time travel device"], "description": "A device used to travel back in time."},
    {"category": "Object", "name": "Sorcerer's Stone", "synonyms": ["Philosopher's Stone"], "description": "Legendary alchemical stone that grants immortality and turns metal into gold."},
    {"category": "Object", "name": "Deluminator", "synonyms": ["Put-Outer"], "description": "Device invented by Dumbledore to collect and release light."},
    {"category": "Object", "name": "Remembrall", "synonyms": ["Memory ball"], "description": "A glass ball that glows red when the owner has forgotten something."},

    # CREATURES
    {"category": "Creature", "name": "Dementor", "synonyms": ["Azkaban guard"], "description": "Dark, soul-sucking creatures that guard Azkaban prison. Cause despair and can perform the Dementor's Kiss."},
    {"category": "Creature", "name": "Basilisk", "synonyms": ["giant serpent"], "description": "A giant serpent whose gaze is instantly fatal. Hidden in the Chamber of Secrets."},
    {"category": "Creature", "name": "Hippogriff", "synonyms": ["Buckbeak"], "description": "A magical creature with the front half of an eagle and the hind half of a horse."},
    {"category": "Creature", "name": "Thestral", "synonyms": ["invisible horse"], "description": "Winged horses that can only be seen by those who have witnessed death."},
    {"category": "Creature", "name": "Phoenix", "synonyms": ["Fawkes"], "description": "Magical bird that can regenerate, known for its healing tears and loyalty."},
    {"category": "Creature", "name": "Acromantula", "synonyms": ["giant spider"], "description": "Giant talking spider, such as Aragog."},
    {"category": "Creature", "name": "House-elf", "synonyms": ["Dobby", "Kreacher"], "description": "Magical creatures bound to serve wizarding families. Can perform powerful magic."},

    # LOCATIONS
    {"category": "Location", "name": "Hogwarts", "synonyms": ["school"], "description": "School of Witchcraft and Wizardry in Scotland."},
    {"category": "Location", "name": "Hogsmeade", "synonyms": ["wizarding village"], "description": "The only all-wizarding village in Britain, near Hogwarts."},
    {"category": "Location", "name": "Diagon Alley", "synonyms": ["wizard market"], "description": "A hidden street in London where witches and wizards shop."},
    {"category": "Location", "name": "The Burrow", "synonyms": ["Weasley home"], "description": "Family home of the Weasley family."},
    {"category": "Location", "name": "Azkaban", "synonyms": ["wizard prison"], "description": "Wizarding prison guarded by Dementors."},
    {"category": "Location", "name": "Godric's Hollow", "synonyms": ["village"], "description": "Birthplace of Godric Gryffindor and where Harry's parents died."},
    {"category": "Location", "name": "Forbidden Forest", "synonyms": ["Dark Forest"], "description": "Dangerous forest on the Hogwarts grounds, home to many magical creatures."},
    {"category": "Location", "name": "Number 12 Grimmauld Place", "synonyms": ["Grimmauld Place"], "description": "The headquarters of the Order of the Phoenix, former Black family home."},
    {"category": "Location", "name": "Ministry of Magic", "synonyms": ["Ministry"], "description": "The governing body for the magical community in Britain."},

    # ORGANIZATIONS
    {"category": "Organization", "name": "Order of the Phoenix", "synonyms": ["Dumbledore's order"], "description": "Secret society founded by Dumbledore to fight Voldemort."},
    {"category": "Organization", "name": "Dumbledore's Army", "synonyms": ["DA"], "description": "Student group led by Harry to teach Defense Against the Dark Arts."},
    {"category": "Organization", "name": "The Marauders", "synonyms": ["Moony, Wormtail, Padfoot, and Prongs"], "description": "Group of four friends at Hogwarts: Lupin, Pettigrew, Sirius, and James Potter."},
    {"category": "Organization", "name": "Death Eaters", "synonyms": ["Voldemort's followers"], "description": "Followers of Lord Voldemort."},
    {"category": "Organization", "name": "SPEW", "synonyms": ["Society for the Promotion of Elfish Welfare"], "description": "Organization founded by Hermione Granger to promote house-elf rights."},

    # GAMES
    {"category": "Game", "name": "Quidditch", "synonyms": ["broomstick sport"], "description": "The most popular wizarding sport, played on broomsticks."},
    {"category": "Game", "name": "Wizard Chess", "synonyms": ["magical chess"], "description": "A chess game in which the pieces move on their own and can destroy each other."},
    {"category": "Game", "name": "Gobstones", "synonyms": ["wizard marble game"], "description": "A game similar to marbles, but the stones squirt a foul-smelling liquid at the loser."},
    {"category": "Game", "name": "Exploding Snap", "synonyms": ["card game"], "description": "A wizarding card game known for its unpredictably exploding cards."},

    # BOOK SUMMARIES
    {"category": "Book Summary", "name": "Philosopher's Stone", "synonyms": ["Book 1", "Sorcerer's Stone", "first book"], "description": "Harry discovers he's a wizard, attends Hogwarts, and stops Voldemort from stealing the Philosopher's Stone."},
    {"category": "Book Summary", "name": "Chamber of Secrets", "synonyms": ["Book 2", "second book"], "description": "Harry returns to Hogwarts, defeats the Basilisk, and saves Ginny Weasley."},
    {"category": "Book Summary", "name": "Prisoner of Azkaban", "synonyms": ["Book 3", "third book"], "description": "Harry learns the truth about Sirius Black and uses a Time-Turner to save him."},
    {"category": "Book Summary", "name": "Goblet of Fire", "synonyms": ["Book 4", "fourth book"], "description": "Harry is forced into the Triwizard Tournament and witnesses Voldemort's return."},
    {"category": "Book Summary", "name": "Order of the Phoenix", "synonyms": ["Book 5", "fifth book"], "description": "Harry forms Dumbledore's Army, battles the Ministry, and loses Sirius Black."},
    {"category": "Book Summary", "name": "Half-Blood Prince", "synonyms": ["Book 6", "sixth book"], "description": "Harry learns about Horcruxes and witnesses Dumbledore's death."},
    {"category": "Book Summary", "name": "Deathly Hallows", "synonyms": ["Book 7", "seventh book"], "description": "Harry, Ron, and Hermione hunt Horcruxes and defeat Voldemort."}
]


example_qa_pairs = [
    {"question": "Who is Harry Potter?", "kb_name": "Harry Potter"},
    {"question": "Who is Hermione Granger?", "kb_name": "Hermione Granger"},
    {"question": "Who is Ron Weasley?", "kb_name": "Ron Weasley"},
    {"question": "Who is Lord Voldemort?", "kb_name": "Lord Voldemort"},
    {"question": "What is Expelliarmus?", "kb_name": "Expelliarmus"},
    {"question": "What is Avada Kedavra?", "kb_name": "Avada Kedavra"},
    {"question": "What is Polyjuice Potion?", "kb_name": "Polyjuice Potion"},
    {"question": "What is a Horcrux?", "kb_name": "Horcrux"},
    {"question": "What is Felix Felicis?", "kb_name": "Felix Felicis"},
    {"question": "What is the Elder Wand?", "kb_name": "Elder Wand"},
    {"question": "Summarize Book 1", "kb_name": "Philosopher's Stone"},
    {"question": "Summarize Book 2", "kb_name": "Chamber of Secrets"},
    {"question": "Summarize Book 3", "kb_name": "Prisoner of Azkaban"},
    {"question": "Summarize Book 4", "kb_name": "Goblet of Fire"},
    {"question": "Summarize Book 5", "kb_name": "Order of the Phoenix"},
    {"question": "Summarize Book 6", "kb_name": "Half-Blood Prince"},
    {"question": "Summarize Book 7", "kb_name": "Deathly Hallows"},
    {"question": "What is Hogwarts?", "kb_name": "Hogwarts"},
    {"question": "What is Azkaban?", "kb_name": "Azkaban"},
   
]

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
    "The Marauder's Map reveals everyone's location at Hogwarts.",
    "Hermione founded S.P.E.W., the Society for the Promotion of Elfish Welfare.",
    "Minerva McGonagall is an Animagus and can turn into a cat.",
    "The Time-Turner used in Book 3 is a Ministry-regulated time travel device.",
    "Fred and George Weasley invented Skiving Snackboxes.",
    "Hedwig is Harry Potter's snowy owl.",
    "Butterbeer is a popular wizarding beverage.",
    "Voldemort's real name is Tom Marvolo Riddle.",
    "A Boggart takes the form of your worst fear.",
    "Harry inherited the Invisibility Cloak from his father, James Potter.",
    "The Room of Requirement only appears when a person is in great need of it.",
    "House-elves can apparate in and out of Hogwarts where wizards cannot.",
    "Neville Longbottom is excellent at Herbology.",
    "The Triwizard Tournament is held between three wizarding schools.",
    "The Leaky Cauldron is the gateway between Muggle London and Diagon Alley.",
 
]    

example_q_set = set(q['question'].strip().lower() for q in example_qa_pairs)

# ---- Functions ----

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
