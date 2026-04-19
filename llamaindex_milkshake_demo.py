"""
LlamaIndex demo of the modern RAG stack on a household-facts corpus:

    HyDE  →  BM25 + dense (RRF fusion)  →  cross-encoder rerank  →  answer

Install:
    pip install -r requirements.txt
Set:
    export OPENAI_API_KEY=...
"""

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# ---------- 1. A tiny household-facts "vector DB" ----------
# Three groups of facts, deliberately chosen so HyDE's generic grocery
# output will MATCH many of the distractors but NOT "My kids like milkshake."
# The milkshake fact is a personal-preference statement phrased
# conversationally — it has no surface-level grocery vocabulary.

PERSONAL_FACTS = [
    "My kids like milkshake.",
    "My wife is lactose intolerant and avoids dairy.",
    "My son has a peanut allergy.",
    "My daughter refuses to eat anything spicy.",
    "I'm trying to cut down on processed sugar.",
    "My father-in-law is diabetic.",
    "My mother is visiting next week, she loves tea.",
    "My youngest won't eat any kind of vegetable.",
    "I follow a mostly vegetarian diet but eat fish on weekends.",
    "My teenager started going to the gym and wants more protein.",
]

# Grocery / food distractors — generic shopping-list vocabulary.
# HyDE's hypothetical passage will overlap heavily with these.
GROCERY_FACTS = [
    "Bananas are usually $0.59 per pound at the supermarket.",
    "Whole-grain bread is in aisle 4 next to the bakery.",
    "The supermarket runs weekly promotions on eggs and cereal.",
    "Organic produce section was expanded last quarter.",
    "Frozen vegetables cost less than fresh on average.",
    "Oat milk and almond milk are shelved with dairy alternatives.",
    "Peanut butter, jam, and honey share a single aisle.",
    "The deli counter sells sliced turkey, ham, and cheese.",
    "Greek yogurt comes in plain, vanilla, and strawberry.",
    "Pasta sauces range from marinara to alfredo to pesto.",
    "Breakfast cereals are stocked in aisle 6 near the coffee.",
    "Snack aisle includes chips, crackers, cookies, and granola bars.",
    "Frozen pizza brands include DiGiorno, Tombstone, and Red Baron.",
    "Canned soups stock tomato, chicken noodle, and minestrone.",
    "Produce includes apples, pears, oranges, grapes, and berries.",
    "Rice section has basmati, jasmine, brown, and sushi rice.",
    "Cheese aisle stocks cheddar, mozzarella, parmesan, and brie.",
    "Meat counter sells beef, chicken, pork, lamb, and fish.",
    "Ice cream comes in vanilla, chocolate, strawberry, and mint.",
    "Smoothie ingredients: frozen berries, bananas, yogurt, juice.",
    "Milkshakes can be made with milk, ice cream, and syrup.",
    "Baking aisle has flour, sugar, baking soda, and vanilla extract.",
    "Condiment section stocks ketchup, mustard, mayo, and hot sauce.",
    "Shopping list templates usually include produce, dairy, grains, proteins.",
    "Popular shopping-list apps include AnyList and Google Keep.",
    "Bulk items at the supermarket include rice, beans, pasta, and flour.",
    "Children's lunchbox favorites are sandwiches, juice boxes, and fruit.",
    "Dairy-free alternatives include soy, oat, almond, and coconut milk.",
    "Party snacks for kids often include chips, cookies, and soda.",
    "Common breakfast foods: cereal, toast, eggs, pancakes, yogurt.",
    "Healthy grocery habits: buy from the perimeter of the store first.",
    "Dinner staples: chicken breast, ground beef, pasta, rice, vegetables.",
    "Lunch staples: sandwich bread, deli meat, cheese, lettuce, tomato.",
    "Pantry basics: olive oil, salt, pepper, garlic, onions.",
    "Fruits good for kids: apples, bananas, grapes, berries, oranges.",
    "Kid-friendly drinks: juice, chocolate milk, lemonade, smoothies.",
    "Frozen food aisle has pizza, waffles, vegetables, and ice cream.",
    "Grocery list should cover breakfast, lunch, dinner, and snacks.",
    "Supermarket self-checkout accepts cards and mobile payments.",
    "Seasonal produce in spring: asparagus, strawberries, rhubarb, peas.",
]

# Household / admin distractors — unrelated to food entirely.
HOUSEHOLD_FACTS = [
    "We ran out of bread last weekend.",
    "The dog's food bag is almost empty.",
    "We usually buy groceries from the supermarket on Sundays.",
    "The car needs an oil change next month.",
    "Mortgage payment is due on the 5th.",
    "My daughter has soccer practice every Tuesday.",
    "Electricity bill was higher than usual this month.",
    "The coffee machine needs descaling.",
    "We have a family reunion planned in June.",
    "I booked a dentist appointment for Thursday.",
    "The washing machine is making a strange noise.",
    "The lawn needs mowing this weekend.",
    "Our internet provider raised prices by 10%.",
    "The kitchen sink has been dripping for three days.",
    "We planned a road trip for spring break.",
    "The smoke detector battery is low.",
    "My home office chair needs replacing.",
    "The guest bedroom needs fresh linens.",
    "I should renew the car registration before it expires.",
    "The garage door opener remote is lost.",
]

FACTS = PERSONAL_FACTS + GROCERY_FACTS + HOUSEHOLD_FACTS
docs = [Document(text=f) for f in FACTS]

# ---------- 2. Models ----------
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm         = OpenAI(model="gpt-4o-mini")

index = VectorStoreIndex.from_documents(docs)
vector_retriever = index.as_retriever(similarity_top_k=10)
bm25_retriever   = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=10
)

# ---------- 3. Hybrid retrieval with RRF fusion ----------
fusion = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=1,          # set >1 to also auto-generate query variants
    mode=FUSION_MODES.RECIPROCAL_RANK,
    use_async=False,
)

# ---------- 4. Cross-encoder reranker ----------
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=5,
)

base_engine = RetrieverQueryEngine.from_args(
    retriever=fusion, node_postprocessors=[reranker]
)

# ---------- 5. HyDE wrapper ----------
hyde   = HyDEQueryTransform(include_original=True)
engine = TransformQueryEngine(base_engine, query_transform=hyde)

# ---------- 6. Run the milkshake query ----------
query = "What should I buy from the supermarket? Add things to the shopping list."

print("=" * 72)
print("QUERY:", query)
print("=" * 72)

# Raw fusion (no HyDE, no rerank) — shows what basic RAG sees
print("\n--- Stage A: BM25 + dense fusion only ---")
for i, node in enumerate(fusion.retrieve(query)[:5], 1):
    print(f"  [{i}] {node.text}   (score={node.score:.3f})")

# Full stack — HyDE + fusion + cross-encoder rerank
print("\n--- Stage B: HyDE + fusion + cross-encoder rerank ---")
response = engine.query(query)
for i, node in enumerate(response.source_nodes, 1):
    print(f"  [{i}] {node.text}   (score={node.score:.3f})")
print("\nAnswer:", str(response))
