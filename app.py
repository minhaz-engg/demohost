import os
import subprocess
import sys

# ----------------------------------------------------------------
# 🔧 AUTO-SETUP: High-IQ Deployment Fix for Streamlit Cloud
# This replicates the 'crawl4ai-setup' command programmatically.
# ----------------------------------------------------------------
def ensure_browsers_installed():
    """
    Checks if the specific browser binary exists. If not, it runs the 
    installation command. This is critical for Streamlit Cloud.
    """
    # We use a marker file to prevent re-running this on every app reload
    marker_file = "crawl4ai_setup_complete.flag"
    
    if not os.path.exists(marker_file):
        print("⚙️ Initiating First-Time Setup: Installing Browsers...")
        try:
            # 1. Install Playwright Browsers (The Core Engine)
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
            
            # 2. Run crawl4ai specific setup if strictly needed (optional but safer)
            # Note: The crawl4ai-setup command is a console script, so we call it via shell
            subprocess.run(["crawl4ai-setup"], shell=True)
            
            # 3. Create the marker file so we don't do this again
            with open(marker_file, "w") as f:
                f.write("Setup Complete")
                
            print("✅ Browser Installation Complete.")
        except Exception as e:
            print(f"⚠️ Setup Warning: {e}")

# EXECUTE SETUP BEFORE APP LOADS
ensure_browsers_installed()
# ----------------------------------------------------------------




import re
import json
import asyncio
import hashlib
import pickle
import numpy as np
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from urllib.parse import urljoin, urlparse

import streamlit as st
import nest_asyncio
from dotenv import load_dotenv

# --- AI & Search Imports ---
from openai import OpenAI
from rank_bm25 import BM25Okapi

# --- Scraping Imports ---
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, JsonCssExtractionStrategy

# Setup Environment
nest_asyncio.apply()
load_dotenv()

# ----------------------------
# 1. System Configuration
# ----------------------------
APP_TITLE = "🛍️ MarketIntel AI"
APP_ICON = "📊"
CACHE_DIR = "intel_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Default Configs
DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/scrape-scheduler/refs/heads/main/out/combined_corpus.md"
DEFAULT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_RESULTS = 70  # High retrieval count for better filtering

# ----------------------------
# 2. Data Structures
# ----------------------------
@dataclass
class ProductDoc:
    doc_id: str
    title: str
    source: str
    category: str
    price_value: float
    url: Optional[str]
    raw_md: str
    embedding: Optional[List[float]] = field(default=None)

@dataclass
class SearchResult:
    doc: ProductDoc
    score: float
    reason: str = ""

# ----------------------------
# 3. Robust Logic Utilities
# ----------------------------

def simple_tokenize(text: str) -> List[str]:
    """
    Business Logic: Normalizes product names for keyword search.
    Handles plurals (e.g., 'Laptops' -> 'Laptop') to ensure matches.
    """
    if not text: return []
    words = re.findall(r'\w+', text.lower())
    # Remove 's' from end if word > 3 chars
    return [w[:-1] if (w.endswith('s') and len(w) > 3) else w for w in words]

def parse_price(price_str: str) -> float:
    """
    Price Intelligence: Extracts accurate pricing from messy formats.
    Fixes merged numbers (e.g., StarTech '1300015000' bug).
    """
    if not price_str: return 0.0
    
    # Replace symbols with space to ensure separation
    clean_str = re.sub(r'(৳|Tk\.?|BDT)', ' ', str(price_str), flags=re.IGNORECASE)
    
    # Extract all number groups (handling commas)
    matches = re.findall(r'[\d,]+(?:\.\d+)?', clean_str)
    
    for match in matches:
        clean_num = match.replace(',', '')
        try:
            val = float(clean_num)
            # Filter out unrealistic low numbers (e.g., '1' year warranty)
            if val > 100: 
                return val
        except: continue
    return 0.0

def parse_corpus_text(raw_text: str, filter_source: str = "Both") -> List[ProductDoc]:
    """
    Parses the internal dataset with source filtering capabilities.
    """
    docs = []
    # Regex to capture individual product blocks
    pattern = re.compile(
        r"(##\s*(?P<title>.+?)\n"          
        r"\*\*DocID:\*\*\s*`(?P<id>[^`]+)`" 
        r"(?P<content>[\s\S]+?))"           
        r"(?=\n##\s|\Z)", re.MULTILINE
    )

    for match in pattern.finditer(raw_text):
        try:
            full_block = match.group(1).strip()
            content = match.group("content")
            
            # Extract Source
            src_m = re.search(r"\*\*Source:\*\*\s*(.+)", content)
            src = src_m.group(1).strip() if src_m else "Unknown"

            # Apply Filter
            if filter_source != "Both":
                if filter_source.lower() not in src.lower():
                    continue

            # Extract Details
            title = match.group("title").strip()
            doc_id = match.group("id").strip()
            
            cat_m = re.search(r"\*\*Category:\*\*\s*(.+)", content)
            cat = cat_m.group(1).strip() if cat_m else "General"
            
            url_m = re.search(r"\*\*URL:\*\*\s*(.+)", content)
            url = url_m.group(1).strip() if url_m else "#"
            
            price_m = re.search(r"\*\*Price:\*\*\s*(.+)", content)
            price = parse_price(price_m.group(1)) if price_m else 0.0

            docs.append(ProductDoc(doc_id, title, src, cat, price, url, full_block))
        except: continue
    return docs

# ----------------------------
# 4. Deep Live Scraper (Engine)
# ----------------------------

async def crawl_category(url: str, source: str) -> List[ProductDoc]:
    """
    Deep Scraper: Scrolls aggressively to fetch maximum product inventory.
    """
    # 1. Define Selectors based on Source
    if source == "StarTech":
        schema = {
            "baseSelector": ".p-item",
            "fields": [
                {"name": "name", "selector": "h4.p-item-name a", "type": "text"},
                {"name": "url", "selector": "h4.p-item-name a", "type": "attribute", "attribute": "href"},
                {"name": "price", "selector": ".p-item-price", "type": "text"}
            ]
        }
        wait_for = "css:.p-item"
    else: # Daraz
        schema = {
            "baseSelector": "div[data-qa-locator='product-item']",
            "fields": [
                {"name": "name", "selector": "a[title]", "type": "attribute", "attribute": "title"},
                {"name": "url", "selector": "a[href]", "type": "attribute", "attribute": "href"},
                {"name": "price", "selector": "span:not([style])", "type": "text"}
            ]
        }
        wait_for = "css:body"

    # 2. Configure Crawler
    config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(schema),
        cache_mode=CacheMode.BYPASS,
        wait_for_images=False,
        verbose=True
    )

    # 3. Aggressive Scrolling Script (Fetches ~40-80 items per page)
    js_scroll = """
        let lastH = 0;
        for(let i=0; i<20; i++) {
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(r => setTimeout(r, 800));
            // Slight scroll up to trigger viewport-based lazy loaders
            window.scrollBy(0, -200); 
            await new Promise(r => setTimeout(r, 400));
            window.scrollTo(0, document.body.scrollHeight);
            
            if(document.body.scrollHeight === lastH && i > 5) break;
            lastH = document.body.scrollHeight;
        }
    """

    # 4. Execute Scrape
    raw_items = []
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config, js_code=js_scroll, wait_for=wait_for)
        
        if result.success:
            # Try Auto-Extract
            try:
                data = json.loads(result.extracted_content)
                if isinstance(data, list): raw_items.extend(data)
                elif isinstance(data, dict): raw_items.append(data)
            except: pass
            
            # Robust Fallback (HTML Parsing)
            if not raw_items and result.html:
                soup = BeautifulSoup(result.html, 'html.parser')
                if source == "Daraz":
                    for card in soup.select("div[data-qa-locator='product-item']"):
                        try:
                            raw_items.append({
                                "name": card.select_one("a[title]")['title'],
                                "url": card.select_one("a[href]")['href'],
                                "price": card.select_one("span:not([style])").get_text(strip=True)
                            })
                        except: continue
                elif source == "StarTech":
                    for card in soup.select(".p-item"):
                        try:
                            # SAFE TEXT EXTRACTION (Separates "13,000" and "15,000" with space)
                            price_txt = card.select_one(".p-item-price").get_text(separator=" ", strip=True)
                            raw_items.append({
                                "name": card.select_one("h4 a").get_text(strip=True),
                                "url": card.select_one("h4 a")['href'],
                                "price": price_txt
                            })
                        except: continue

    # 5. Normalize Data
    docs = []
    cat_name = urlparse(url).path.split('/')[-1] or "Live-Session"
    
    for item in raw_items:
        title = item.get('name', 'Unknown')
        raw_url = item.get('url', '')
        
        # URL Correction
        if raw_url.startswith("//"): raw_url = "https:" + raw_url
        elif raw_url.startswith("/"):
             base = "https://www.startech.com.bd" if source == "StarTech" else "https://www.daraz.com.bd"
             raw_url = urljoin(base, raw_url)
        
        price = parse_price(item.get('price', ''))
        doc_id = f"{source}_{hashlib.md5(title.encode()).hexdigest()[:8]}"
        
        raw_md = f"## {title}\n**DocID:** `{doc_id}`\n**Category:** {cat_name}\n**Price:** {price}\n**Source:** {source}\n**URL:** {raw_url}\n---"
        docs.append(ProductDoc(doc_id, title, source, cat_name, price, raw_url, raw_md))
        
    return docs

# ----------------------------
# 5. Hybrid Search Core
# ----------------------------

class HybridSearchEngine:
    """
    The Brain of the Application.
    Combines Vector Search (Semantics) with BM25 (Keywords).
    """
    def __init__(self, products: List[ProductDoc]):
        self.products = products
        self.client = OpenAI()
        self.bm25 = None
        self.corpus_embeddings = None
        self.categories: Set[str] = set()
        
        # Initialization
        self.update_categories()
        self.rebuild_bm25()
        
        # Intelligent Embedding Loading
        # If dataset is small (Live Scrape), generate fresh. If large (Historical), check cache.
        if len(products) < 500:
             self.generate_embeddings_fresh(use_cache=False)
        else:
             self.load_or_generate_embeddings()

    def update_categories(self):
        self.categories = sorted(list(set(p.category for p in self.products)))

    def rebuild_bm25(self):
        tokens = [simple_tokenize(p.title + " " + p.category) for p in self.products]
        self.bm25 = BM25Okapi(tokens)

    def load_or_generate_embeddings(self):
        if not self.products: return
        # Create signature of the dataset
        content_hash = hashlib.md5("".join([p.doc_id for p in self.products]).encode()).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"emb_{content_hash}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.corpus_embeddings = pickle.load(f)
            # Map embeddings back to objects
            for i, p in enumerate(self.products):
                if i < len(self.corpus_embeddings): p.embedding = self.corpus_embeddings[i]
        else:
            self.generate_embeddings_fresh(use_cache=True)

    def generate_embeddings_fresh(self, use_cache=True):
        if not self.products: return
        
        msg = "Processing Live Data..." if len(self.products) < 500 else "Indexing Knowledge Base..."
        if len(self.products) > 100:
            progress = st.progress(0, text=msg)
        
        texts = [f"{p.title} {p.category} Price: {p.price_value}" for p in self.products]
        all_embs = []
        batch_size = 200
        
        # Batch Processing
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                resp = self.client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
                all_embs.extend([d.embedding for d in resp.data])
            except Exception as e: 
                # Fail gracefully by adding zero vectors so indexes match
                all_embs.extend([[0.0]*1536] * len(batch))
            
            if len(self.products) > 100:
                progress.progress(min((i+batch_size)/len(texts), 1.0))
        
        if len(self.products) > 100: progress.empty()
        
        self.corpus_embeddings = np.array(all_embs)
        
        for i, p in enumerate(self.products):
            p.embedding = all_embs[i]

        if use_cache:
            content_hash = hashlib.md5("".join([p.doc_id for p in self.products]).encode()).hexdigest()
            with open(os.path.join(CACHE_DIR, f"emb_{content_hash}.pkl"), "wb") as f:
                pickle.dump(self.corpus_embeddings, f)

    # def search(self, query: str, filters: Dict, top_k: int = TOP_K_RESULTS) -> List[SearchResult]:
    #     # 1. Hard Filtering (Metadata)
    #     valid_indices = []
    #     for i, p in enumerate(self.products):
    #         # Price Logic
    #         if filters.get('max_price') and p.price_value > filters['max_price']: continue
    #         if filters.get('min_price') and p.price_value < filters['min_price']: continue
    #         # Category Logic (Smart Fuzzy Match)
    #         if filters.get('category'):
    #             q_cat = filters['category'].lower()
    #             p_cat = p.category.lower()
    #             if q_cat not in p_cat and p_cat not in q_cat:
    #                 # Exception for common synonyms
    #                 if "laptop" in q_cat and ("macbook" in p_cat or "notebook" in p_cat): pass
    #                 else: continue
    #         valid_indices.append(i)

    #     if not valid_indices: return []

    #     # 2. Vector Search (Semantic)
    #     q_emb = self.client.embeddings.create(input=query, model=EMBEDDING_MODEL).data[0].embedding
    #     valid_embs = self.corpus_embeddings[valid_indices]
    #     sem_scores = np.dot(valid_embs, np.array(q_emb))

    #     # 3. Keyword Search (Exact Match)
    #     q_tok = simple_tokenize(query)
    #     bm25_full = self.bm25.get_scores(q_tok)
    #     kw_scores = np.array([bm25_full[i] for i in valid_indices])

    #     # 4. Score Fusion (70% Semantic, 30% Keyword)
    #     def norm(arr):
    #         if len(arr) == 0 or np.max(arr) == np.min(arr): return np.zeros_like(arr)
    #         return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    #     final_scores = (0.7 * norm(sem_scores)) + (0.3 * norm(kw_scores))

    #     results = []
    #     for idx_in_valid, score in enumerate(final_scores):
    #         results.append(SearchResult(self.products[valid_indices[idx_in_valid]], score))

    #     return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

    def search(self, query: str, filters: Dict, top_k: int = TOP_K_RESULTS) -> List[SearchResult]:
        # 1. Hard Filtering (Metadata)
        valid_indices = []
        
        # Pre-process query category if it exists for O(1) matching inside loop
        q_cat_tokens = set()
        if filters.get('category'):
            # Logic: Lowercase -> Replace separators -> Split -> Remove trailing 's'/'es'
            raw_q = filters['category'].lower().replace('-', ' ').replace('_', ' ')
            q_cat_tokens = {w.rstrip('s') for w in raw_q.split() if len(w) > 2}

        for i, p in enumerate(self.products):
            # Price Logic
            if filters.get('max_price') and p.price_value > filters['max_price']: continue
            if filters.get('min_price') and p.price_value < filters['min_price']: continue
            
            # --- HIGH-IQ CATEGORY MATCHING (The Fix) ---
            if filters.get('category') and q_cat_tokens:
                # 1. Normalize the Product Category same way as Query
                p_raw = p.category.lower().replace('-', ' ').replace('_', ' ')
                p_cat_tokens = {w.rstrip('s') for w in p_raw.split()}
                
                # 2. Set Intersection Logic
                # If the Query is "Smart Watches" -> {smart, watche}
                # If the Product is "smart-watch" -> {smart, watche}
                # Intersection is non-empty = MATCH.
                
                # We require at least one significant token match to allow the pass
                if not q_cat_tokens.intersection(p_cat_tokens):
                    # Fallback: keep the laptop exception just in case
                    q_str = filters['category'].lower()
                    p_str = p.category.lower()
                    if "laptop" in q_str and ("macbook" in p_str or "notebook" in p_str): 
                        pass
                    else: 
                        continue
            # -------------------------------------------

            valid_indices.append(i)

        if not valid_indices: return []

        # 2. Vector Search (Semantic)
        q_emb = self.client.embeddings.create(input=query, model=EMBEDDING_MODEL).data[0].embedding
        valid_embs = self.corpus_embeddings[valid_indices]
        sem_scores = np.dot(valid_embs, np.array(q_emb))

        # 3. Keyword Search (Exact Match)
        q_tok = simple_tokenize(query)
        bm25_full = self.bm25.get_scores(q_tok)
        kw_scores = np.array([bm25_full[i] for i in valid_indices])

        # 4. Score Fusion (70% Semantic, 30% Keyword)
        def norm(arr):
            if len(arr) == 0 or np.max(arr) == np.min(arr): return np.zeros_like(arr)
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        final_scores = (0.7 * norm(sem_scores)) + (0.3 * norm(kw_scores))

        results = []
        for idx_in_valid, score in enumerate(final_scores):
            results.append(SearchResult(self.products[valid_indices[idx_in_valid]], score))

        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

# ----------------------------
# 6. User Interface (Streamlit)
# ----------------------------

st.set_page_config(
    page_title=APP_TITLE, 
    page_icon=APP_ICON, 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom Styling for Business Look
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .sidebar .sidebar-content { background: #ffffff; }
    h1 { color: #1f2937; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
    .stChatInput { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("AI-Powered Market Research & Price Intelligence Tool")

# --- Session Management ---
if "engine" not in st.session_state: st.session_state.engine = None
if "mode" not in st.session_state: st.session_state.mode = "Not Initialized"
if "messages" not in st.session_state: 
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 **Welcome!** Please select a data source from the sidebar to begin analyzing products."}
    ]

# --- SIDEBAR: Control Panel ---
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    tab1, tab2 = st.tabs(["📚 Knowledge Base", "🕸️ Live Scanner"])
    
    # TAB 1: Historical Data
    with tab1:
        st.write("Load pre-collected market data.")
        url_in = st.text_input("Dataset URL", value=DEFAULT_CORPUS_URL, help="Link to the raw Markdown corpus.")
        
        st.write("**Source Preference:**")
        source_pref = st.radio(
            "Select Source",
            ["Both", "Daraz", "StarTech"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if st.button("Load Knowledge Base", type="primary"):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("Missing OpenAI API Key in .env file.")
            else:
                with st.spinner(f"Indexing {source_pref} Data..."):
                    try:
                        resp = requests.get(url_in)
                        if resp.status_code == 200:
                            docs = parse_corpus_text(resp.text, filter_source=source_pref)
                            if not docs:
                                st.warning(f"No data found for {source_pref}.")
                            else:
                                st.session_state.engine = HybridSearchEngine(docs)
                                st.session_state.mode = f"Knowledge Base ({source_pref})"
                                st.success(f"✅ Ready! Indexed {len(docs)} items.")
                                # Reset chat for new context
                                st.session_state.messages = [{"role": "assistant", "content": f"✅ Loaded **{len(docs)}** products from {source_pref}. Ask me about prices, specs, or comparisons."}]
                        else: st.error("Failed to fetch dataset URL.")
                    except Exception as e: st.error(f"System Error: {e}")

    # TAB 2: Live Scraping
    with tab2:
        st.info("Paste a category link to analyze real-time pricing.")
        scrape_url = st.text_input("Category URL (Daraz/StarTech)")
        
        if st.button("Scan & Analyze URL"):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("Missing OpenAI API Key.")
            else:
                source = "StarTech" if "startech" in scrape_url.lower() else "Daraz"
                with st.spinner(f"🕷️ Scanning {source} (Deep Scroll Mode)..."):
                    try:
                        # Clear old engine to create a focused session
                        st.session_state.engine = None 
                        new_docs = asyncio.run(crawl_category(scrape_url, source))
                        
                        if new_docs:
                            st.session_state.engine = HybridSearchEngine(new_docs)
                            st.session_state.mode = f"Live Scan ({len(new_docs)} items)"
                            st.success(f"✅ Scanned {len(new_docs)} items successfully.")
                            st.session_state.messages = [{"role": "assistant", "content": f"✅ **Live Scan Complete.** I found {len(new_docs)} items from the link provided. You can now ask specific questions about these products."}]
                        else: st.warning("No items found. Please check the URL.")
                    except Exception as e: st.error(f"Scanning Error: {e}")

    # --- System Status ---
    if st.session_state.engine:
        st.divider()
        st.markdown(f"**🟢 System Status:** Active")
        st.markdown(f"**📁 Mode:** `{st.session_state.mode}`")
        st.markdown(f"**📦 Products:** {len(st.session_state.engine.products)}")
        with st.expander("Show Active Categories"):
            st.write(st.session_state.engine.categories)
    else:
        st.divider()
        st.markdown("**🔴 System Status:** Offline (Load Data first)")

# --- MAIN: Chat Interface ---

# Display History
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Chat Input
if prompt := st.chat_input("Ex: 'Best gaming laptop under 100k' or 'Compare Redmi and Realme prices'"):
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # Logic
    if not st.session_state.engine:
        err_msg = "⚠️ **System Offline.** Please load the Knowledge Base or Scan a URL from the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": err_msg})
        with st.chat_message("assistant"): st.error(err_msg)
    else:
        with st.chat_message("assistant"):
            
            client = OpenAI()
            intent_prompt = (
                "You are a Data Analyst Assistant. Extract filters from the user query.\n"
                "Return JSON: {\"query\": string, \"filters\": {\"max_price\": int, \"min_price\": int, \"category\": string}}\n"
                f"User Query: {prompt}"
            )
            
            with st.spinner("🧠 Analyzing Request..."):
                try:
                    raw = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"system", "content":"JSON only."}, {"role":"user", "content":intent_prompt}],
                        response_format={"type": "json_object"}
                    )
                    intent = json.loads(raw.choices[0].message.content)
                    q_text = intent.get("query", prompt)
                    filters = intent.get("filters", {})
                except: q_text, filters = prompt, {}
            
            # 2. Retrieval
            results = st.session_state.engine.search(q_text, filters, top_k=TOP_K_RESULTS)

            if not results:
                fail_msg = f"❌ No products matched your criteria in the current {st.session_state.mode}."
                st.warning(fail_msg)
                st.session_state.messages.append({"role": "assistant", "content": fail_msg})
            else:
                # 3. Report Generation
                # Format context for the LLM
                context_str = "\n".join([
                    f"- {r.doc.title} | Price: ৳{r.doc.price_value:,.0f} | Source: {r.doc.source} | [Link]({r.doc.url})" 
                    for r in results
                ])
                
                system_instruction = (
                    "You are a Senior Market Analyst. Provide a professional business response.\n"
                    "1. **Executive Summary**: Recommend the top 3-5 options based on value and specs.\n"
                    "2. **Pricing**: Always state price in Taka (৳).\n"
                    "3. **Sourcing**: Mention if it's from StarTech or Daraz.\n"
                    "4. **Formatting**: Use Markdown tables for comparisons if multiple items are discussed.\n"
                    "5. **Links**: Ensure product names are clickable links using the provided [Link](url) format.\n"
                    "6. Be concise, objective, and data-driven."
                )
                
                user_prompt = f"User Question: {prompt}\n\nMarket Data:\n{context_str}"

                stream = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=True
                )
                
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})