import os
import re
import time
import networkx as nx
import requests
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

# Load local env vars (fallback for local dev)
load_dotenv()

# ----------------------------
# 1. Configuration & Constants
# ----------------------------
PAGE_TITLE = "🛍️ Sigmoix GraphRAG"
DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/ragapplications/refs/heads/main/refined_dataset/combined_corpus_fixed.md"
DEFAULT_MODEL = "gpt-4o-mini"
TOP_K_RETRIEVAL = 20

STOP_WORDS = {
    "new", "sale", "best", "exclusive", "offer", "discount", "hot", "top", 
    "original", "premium", "smart", "super", "mega", "combo", "buy", "get"
}

KNOWN_BRANDS = {
    "apple", "samsung", "xiaomi", "realme", "oneplus", "oppo", "vivo", "google",
    "hp", "dell", "lenovo", "asus", "acer", "msi", "gigabyte", "intel", "amd",
    "nvidia", "sony", "canon", "nikon", "fujifilm", "dji", "gopro", "amazon",
    "logitech", "razer", "corsair", "tp-link", "d-link", "netgear", "tenda",
    "mikrotik", "cisco", "huion", "wacom", "hoco", "baseus", "anker", "remax",
    "joyroom", "haylou", "qcy", "soundpeats", "jbl", "bose", "sony", "edifier"
}

# ----------------------------
# 2. Security & Auth Module
# ----------------------------
def check_password():
    """Returns `True` if the user had the correct password."""
    
    # Check if password is set in secrets, if not, allow open access (Warning: Risky)
    if "APP_PASSWORD" not in st.secrets:
        st.warning("⚠️ No password set in secrets. App is open.")
        return True

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Enter Business Access Code", type="password", on_change=password_entered, key="password"
        )
        st.caption("Protected for Internal Use Only")
        return False
        
    elif not st.session_state["password_correct"]:
        # Password was incorrect, show input again.
        st.text_input(
            "Enter Business Access Code", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
        
    else:
        # Password was correct.
        return True

# ----------------------------
# 3. Data Structures
# ----------------------------

@dataclass
class ProductDoc:
    doc_id: str
    title: str
    source: str
    category: str
    brand: str
    price_val: float
    url: str
    raw_text: str

    @property
    def clean_text(self) -> str:
        return f"{self.title} {self.brand} {self.category} {self.source}"

# ----------------------------
# 4. Utilities (FIXED PRICE PARSING)
# ----------------------------

class SmartTokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text)
        return tokens

def infer_brand_robust(title: str, explicitly_tagged_brand: str = None) -> str:
    if explicitly_tagged_brand and explicitly_tagged_brand.lower() != "generic":
        return explicitly_tagged_brand.lower()
    
    if not title:
        return "generic"
    
    title_lower = title.lower()
    tokens = SmartTokenizer.tokenize(title_lower)
    
    for token in tokens:
        if token in KNOWN_BRANDS:
            return token
            
    if tokens:
        candidate = tokens[0]
        if candidate not in STOP_WORDS and len(candidate) > 2:
            return candidate
            
    return "generic"

def parse_price(price_str: str) -> float:
    """
    FIXED: Handles cases like '23,850৳ 26,500৳' by taking ONLY the first number.
    """
    if not price_str: return 0.0
    
    # 1. Remove commas and currency symbols
    clean_str = price_str.replace(',', '').replace('৳', '')
    
    # 2. Find separate number groups (e.g., '23850' and '26500')
    nums = re.findall(r'\d+(?:\.\d+)?', clean_str)
    
    # 3. Return the FIRST number found (the current price)
    if nums:
        return float(nums[0])
        
    return 0.0

# ----------------------------
# 5. Data Ingestion
# ----------------------------

@st.cache_data(ttl=3600)
def fetch_raw_data(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        return ""

def parse_corpus(text: str) -> List[ProductDoc]:
    if not text: return []
    product_blocks = re.split(r'\n---\n', text)
    docs = []
    
    re_docid = re.compile(r"\*\*DocID:\*\*\s*`([^`]+)`")
    re_source = re.compile(r"\*\*Source:\*\*\s*(.+)")
    re_cat = re.compile(r"\*\*Category:\*\*\s*(.+)")
    re_brand = re.compile(r"\*\*Brand:\*\*\s*(.+)")
    re_url = re.compile(r"\*\*URL:\*\*\s*(\S+)")
    re_price = re.compile(r"\*\*Price:\*\*\s*(.+)")
    re_title = re.compile(r"^##\s*(.+)", re.MULTILINE)

    for block in product_blocks:
        block = block.strip()
        if not block or "**DocID:**" not in block:
            continue
            
        doc_id_m = re_docid.search(block)
        doc_id = doc_id_m.group(1).strip() if doc_id_m else f"unknown-{hash(block)}"
        
        title_m = re_title.search(block)
        title = title_m.group(1).strip() if title_m else "Unknown Product"
        
        brand_m = re_brand.search(block)
        raw_brand = brand_m.group(1).strip() if brand_m else None
        final_brand = infer_brand_robust(title, raw_brand)
        
        source = re_source.search(block).group(1).strip() if re_source.search(block) else "Unknown"
        category = re_cat.search(block).group(1).strip().lower() if re_cat.search(block) else "general"
        url = re_url.search(block).group(1).strip() if re_url.search(block) else ""
        price_val = parse_price(re_price.search(block).group(1)) if re_price.search(block) else 0.0

        docs.append(ProductDoc(
            doc_id=doc_id, title=title, source=source, category=category,
            brand=final_brand, price_val=price_val, url=url, raw_text=block
        ))
        
    return docs

# ----------------------------
# 6. GraphRAG Engine
# ----------------------------

class GraphRAGIndex:
    def __init__(self, docs: List[ProductDoc]):
        self.doc_map = {d.doc_id: d for d in docs}
        self.graph = nx.Graph()
        self.docs_daraz = [d for d in docs if 'daraz' in d.source.lower()]
        self.docs_startech = [d for d in docs if 'startech' in d.source.lower()]
        self.bm25_daraz = self._build_bm25(self.docs_daraz)
        self.bm25_startech = self._build_bm25(self.docs_startech)
        self._build_knowledge_graph(docs)

    def _build_bm25(self, doc_list: List[ProductDoc]):
        if not doc_list: return None
        tokenized_corpus = [SmartTokenizer.tokenize(d.clean_text) for d in doc_list]
        return BM25Okapi(tokenized_corpus)

    def _build_knowledge_graph(self, docs: List[ProductDoc]):
        for doc in docs:
            self.graph.add_node(doc.doc_id, type='product', source=doc.source)
            if doc.brand and doc.brand not in ["generic", "unknown", "other"]:
                brand_node = f"BRAND:{doc.brand}"
                self.graph.add_node(brand_node, type='brand')
                self.graph.add_edge(doc.doc_id, brand_node)
            if doc.category and doc.category not in ["general", "unknown"]:
                cat_node = f"CAT:{doc.category}"
                self.graph.add_node(cat_node, type='category')
                self.graph.add_edge(doc.doc_id, cat_node)

    def search(self, query: str, total_k: int = 20) -> List[ProductDoc]:
        half_k = total_k // 2
        tokenized_query = SmartTokenizer.tokenize(query)
        daraz_hits = self._query_bm25(self.bm25_daraz, self.docs_daraz, tokenized_query, half_k)
        startech_hits = self._query_bm25(self.bm25_startech, self.docs_startech, tokenized_query, half_k)
        
        combined = []
        max_len = max(len(daraz_hits), len(startech_hits))
        for i in range(max_len):
            if i < len(daraz_hits): combined.append(daraz_hits[i])
            if i < len(startech_hits): combined.append(startech_hits[i])
            
        expanded_results = []
        seen_ids = {d.doc_id for d in combined}
        expanded_results.extend(combined)
        
        if combined:
            seeds = combined[:2]
            graph_hits = []
            for seed in seeds:
                try:
                    neighbors = list(self.graph.neighbors(seed.doc_id))
                    for node in neighbors:
                        siblings = list(self.graph.neighbors(node))
                        for sib_id in siblings:
                            if sib_id != seed.doc_id and sib_id not in seen_ids and sib_id in self.doc_map:
                                graph_hits.append(self.doc_map[sib_id])
                                seen_ids.add(sib_id)
                except:
                    pass
            expanded_results.extend(graph_hits[:5])
        return expanded_results

    def _query_bm25(self, bm25_idx, doc_source, tokenized_query, k):
        if not bm25_idx or not doc_source: return []
        scores = bm25_idx.get_scores(tokenized_query)
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [doc_source[i] for i in top_n if scores[i] > 0.0]

@st.cache_resource(show_spinner=False)
def load_search_engine():
    raw_text = fetch_raw_data(DEFAULT_CORPUS_URL)
    if not raw_text: return None
    docs = parse_corpus(raw_text)
    return GraphRAGIndex(docs)

# ----------------------------
# 7. UI / Main App
# ----------------------------

def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon="🛍️")

    # --- SECURITY CHECKPOINT ---
    if not check_password():
        st.stop()  # Stop execution if not authenticated
    # ---------------------------
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(PAGE_TITLE)
        st.caption("Engineered for Fairness: Dual-Index Retrieval + Graph Expansion")
    with col2:
        st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)

    # Sidebar
    with st.sidebar:
        st.info("System Status")
        
        with st.spinner("Initializing Knowledge Graph..."):
            index = load_search_engine()
            
        if index:
            st.success(f"✅ System Online")
            st.metric("Total Products", len(index.doc_map))
        else:
            st.error("❌ System Offline (Data Fetch Failed)")
            st.stop()

        st.divider()
        if st.button("🗑️ Reset Conversation"):
            st.session_state.messages = []
            st.rerun()

        if st.button("🧹 Clear Cache (Admin)"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you compare prices today?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ex: 'Best gaming laptop under 100k'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        start_time = time.time()
        results = index.search(prompt, total_k=TOP_K_RETRIEVAL)
        latency = time.time() - start_time

        if not results:
            with st.chat_message("assistant"):
                st.warning("No matching products found in the database.")
            return

        context_str = ""
        context_display = []
        for i, doc in enumerate(results, 1):
            context_str += f"Item {i}: {{ Title: '{doc.title}', Brand: '{doc.brand}', Price: {doc.price_val}, Store: '{doc.source}', URL: '{doc.url}' }}\n"
            context_display.append(doc)

        with st.chat_message("assistant"):
            with st.expander(f"🔍 Retrieved {len(results)} items in {latency:.3f}s", expanded=False):
                for doc in context_display:
                    color = "blue" if "daraz" in doc.source.lower() else "red"
                    st.markdown(f":{color}[**{doc.source}**] [{doc.title}]({doc.url}) - **{doc.price_val:,.0f}৳**")

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
            
            if not client:
                st.info("💡 API Key missing. Showing raw results above.")
            else:
                stream_box = st.empty()
                full_resp = ""
                
                system_prompt = (
                    "You are Sigmoix-AI, a precision Procurement Analyst. "
                    "Your task is to compare product prices based STRICTLY on the provided Context. "
                    
                    "### GUIDELINES:\n"
                    "1. **ZERO HALLUCINATION**: If the answer is not in the Context, state: 'Data not available in current index.' Do NOT make up prices or specs.\n"
                    "2. **SOURCE TRUTH**: Trust the Context prices over your internal knowledge. Prices change; the Context is the only truth.\n"
                    "3. **CITATION**: When mentioning a product, you MUST format it as a markdown link: [Product Title](URL).\n"
                    "4. **COMPARISON**: If a product exists on both Daraz and StarTech, create a Markdown Table comparing them.\n"
                    "5. **CURRENCY**: Use Bangladesh Taka (৳).\n"
                    "6. **RECOMMENDATION**: Boldly highlight the best value option.\n\n"
                    
                    "Output Format:\n"
                    "- **Analysis**: Brief summary of findings.\n"
                    "- **Comparison Table**: (If applicable)\n"
                    "- **Best Deal**: Clear verdict."
                )
                
                user_message_content = (
                    f"### Context Data:\n{context_str}\n\n"
                    f"### User Question:\n{prompt}\n\n"
                    "### Instruction:\nAnswer the question using ONLY the Context Data above."
                )
                
                try:
                    stream = client.chat.completions.create(
                        model=DEFAULT_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message_content}
                        ],
                        stream=True
                    )
                    for chunk in stream:
                        content = chunk.choices[0].delta.content or ""
                        full_resp += content
                        stream_box.markdown(full_resp + "▌")
                    stream_box.markdown(full_resp)
                    st.session_state.messages.append({"role": "assistant", "content": full_resp})
                except Exception as e:
                    st.error(f"LLM Error: {e}")

if __name__ == "__main__":
    main()