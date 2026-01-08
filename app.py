import os
import subprocess
import sys

# ----------------------------------------------------------------
# 🔧 AUTO-SETUP: Fix for Streamlit Cloud
# ----------------------------------------------------------------
def ensure_browsers_installed():
    """
    Checks if the specific browser binary exists. If not, it runs the 
    installation command. This is critical for Streamlit Cloud.
    """
    marker_file = "crawl4ai_setup_complete.flag"
    
    if not os.path.exists(marker_file):
        print("⚙️ Initiating First-Time Setup: Installing Browsers...")
        try:
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
            subprocess.run(["crawl4ai-setup"], shell=True)
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
APP_TITLE = "🛍️ MarketIntel AI (Visual Mode)"
APP_ICON = "🦅"
CACHE_DIR = "intel_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/ragapplications/refs/heads/main/refined_dataset/combined_corpus_fixed.md"
DEFAULT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_RESULTS = 70 

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
    image_url: Optional[str]  # <--- ADDED THIS FIELD
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
    if not text: return []
    words = re.findall(r'\w+', text.lower())
    return [w[:-1] if (w.endswith('s') and len(w) > 3) else w for w in words]

def parse_price(price_str: str) -> float:
    if not price_str: return 0.0
    # Normalize: Remove currency symbols but keep separators
    clean_str = re.sub(r'(৳|Tk\.?|BDT|Price:|Regular|Sale)', ' ', str(price_str), flags=re.IGNORECASE)
    matches = re.findall(r'[\d,]+(?:\.\d+)?', clean_str)
    
    valid_prices = []
    for match in matches:
        clean_num = match.replace(',', '')
        try:
            val = float(clean_num)
            if 100 < val < 2000000: 
                valid_prices.append(val)
        except: continue
    
    if not valid_prices: return 0.0
    return min(valid_prices)

def parse_corpus_text(raw_text: str, filter_source: str = "Both") -> List[ProductDoc]:
    docs = []
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
            
            src_m = re.search(r"\*\*Source:\*\*\s*(.+)", content)
            src = src_m.group(1).strip() if src_m else "Unknown"

            if filter_source != "Both" and filter_source.lower() not in src.lower():
                continue

            title = match.group("title").strip()
            doc_id = match.group("id").strip()
            
            cat_m = re.search(r"\*\*Category:\*\*\s*(.+)", content)
            cat = cat_m.group(1).strip() if cat_m else "General"
            
            url_m = re.search(r"\*\*URL:\*\*\s*(.+)", content)
            url = url_m.group(1).strip() if url_m else "#"

            # Try to find image in corpus if it exists (Optional fallback)
            img_m = re.search(r"\*\*Image:\*\*\s*(.+)", content)
            image_url = img_m.group(1).strip() if img_m else ""
            
            price_m = re.search(r"\*\*Price:\*\*\s*(.+)", content)
            price = parse_price(price_m.group(1)) if price_m else 0.0

            docs.append(ProductDoc(doc_id, title, src, cat, price, url, image_url, full_block))
        except: continue
    return docs

# ----------------------------
# 4. Deep Live Scraper (Engine)
# ----------------------------

async def crawl_category(url: str, source: str) -> List[ProductDoc]:
    """
    Deep Scraper V2.1: Now extracts IMAGES.
    """
    # 1. Define Selectors based on Source
    if source == "StarTech":
        schema = {
            "baseSelector": ".p-item",
            "fields": [
                {"name": "name", "selector": "h4.p-item-name a", "type": "text"},
                {"name": "url", "selector": "h4.p-item-name a", "type": "attribute", "attribute": "href"},
                {"name": "price_text", "selector": ".p-item-price", "type": "text"},
                # Capture Image Src
                {"name": "image", "selector": ".p-item-img img", "type": "attribute", "attribute": "src"}
            ]
        }
        wait_for = "css:.p-item"
    else: # Daraz
        schema = {
            "baseSelector": "div[data-qa-locator='product-item']",
            "fields": [
                {"name": "name", "selector": "a[title]", "type": "attribute", "attribute": "title"},
                {"name": "url", "selector": "a[href]", "type": "attribute", "attribute": "href"},
                {"name": "price_text", "selector": "span", "type": "text"},
                # Capture Image Src
                {"name": "image", "selector": "img", "type": "attribute", "attribute": "src"} 
            ]
        }
        wait_for = "css:body"

    config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(schema),
        cache_mode=CacheMode.BYPASS,
        wait_for_images=True, # Force wait for images to load
        verbose=True
    )

    # Scroll Script
    js_scroll = """
        let lastH = 0;
        for(let i=0; i<15; i++) {
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(r => setTimeout(r, 1000));
            if(document.body.scrollHeight === lastH && i > 3) break;
            lastH = document.body.scrollHeight;
        }
    """

    raw_items = []
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config, js_code=js_scroll, wait_for=wait_for)
        
        if result.success:
            # 1. Extraction Strategy Results
            try:
                data = json.loads(result.extracted_content)
                if isinstance(data, list): raw_items.extend(data)
                elif isinstance(data, dict): raw_items.append(data)
            except: pass
            
            # 2. BeautifulSoup Fallback
            if not raw_items and result.html:
                soup = BeautifulSoup(result.html, 'html.parser')
                
                if source == "Daraz":
                    for card in soup.select("div[data-qa-locator='product-item']"):
                        try:
                            img_tag = card.select_one("img")
                            img_src = img_tag['src'] if img_tag else ""
                            raw_items.append({
                                "name": card.select_one("a[title]")['title'],
                                "url": card.select_one("a[href]")['href'],
                                "price_text": card.get_text(separator=" ", strip=True),
                                "image": img_src
                            })
                        except: continue
                        
                elif source == "StarTech":
                    for card in soup.select(".p-item"):
                        try:
                            img_tag = card.select_one(".p-item-img img")
                            img_src = img_tag['src'] if img_tag else ""
                            raw_items.append({
                                "name": card.select_one("h4 a").get_text(strip=True),
                                "url": card.select_one("h4 a")['href'],
                                "price_text": card.select_one(".p-item-price").get_text(separator=" ", strip=True),
                                "image": img_src
                            })
                        except: continue

    # 5. Normalize Data & Calculate Prices
    docs = []
    cat_name = urlparse(url).path.split('/')[-1] or "Live-Session"
    
    for item in raw_items:
        title = item.get('name', 'Unknown')
        raw_url = item.get('url', '')
        raw_img = item.get('image', '')

        # URL Correction for Link
        if raw_url.startswith("//"): raw_url = "https:" + raw_url
        elif raw_url.startswith("/"):
             base = "https://www.startech.com.bd" if source == "StarTech" else "https://www.daraz.com.bd"
             raw_url = urljoin(base, raw_url)
        
        # URL Correction for Image (Sometimes they are relative)
        if raw_img and raw_img.startswith("//"): raw_img = "https:" + raw_img
        elif raw_img and raw_img.startswith("/"):
             base = "https://www.startech.com.bd" if source == "StarTech" else "https://www.daraz.com.bd"
             raw_img = urljoin(base, raw_img)

        price_txt = item.get('price_text', '')
        if not price_txt: price_txt = item.get('price', '')
        
        real_price = parse_price(price_txt)
        
        if real_price < 100: continue

        doc_id = f"{source}_{hashlib.md5(title.encode()).hexdigest()[:8]}"
        
        # Construct MD including Image info for LLM
        raw_md = f"## {title}\n**DocID:** `{doc_id}`\n**Category:** {cat_name}\n**Price:** {real_price}\n**Image:** {raw_img}\n**Source:** {source}\n**URL:** {raw_url}\n---"
        docs.append(ProductDoc(doc_id, title, source, cat_name, real_price, raw_url, raw_img, raw_md))
        
    return docs

# ----------------------------
# 5. Hybrid Search Core
# ----------------------------

class HybridSearchEngine:
    def __init__(self, products: List[ProductDoc]):
        self.products = products
        self.client = OpenAI()
        self.bm25 = None
        self.corpus_embeddings = None
        self.categories: Set[str] = set()
        
        self.update_categories()
        self.rebuild_bm25()
        
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
        content_hash = hashlib.md5("".join([p.doc_id for p in self.products]).encode()).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"emb_{content_hash}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.corpus_embeddings = pickle.load(f)
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
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                resp = self.client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
                all_embs.extend([d.embedding for d in resp.data])
            except Exception as e: 
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

    def search(self, query: str, filters: Dict, top_k: int = TOP_K_RESULTS) -> List[SearchResult]:
        valid_indices = []
        q_cat_tokens = set()
        if filters.get('category'):
            raw_q = filters['category'].lower().replace('-', ' ').replace('_', ' ')
            q_cat_tokens = {w.rstrip('s') for w in raw_q.split() if len(w) > 2}

        for i, p in enumerate(self.products):
            if filters.get('max_price') and p.price_value > filters['max_price']: continue
            if filters.get('min_price') and p.price_value < filters['min_price']: continue
            
            # Category Matching
            if filters.get('category') and q_cat_tokens:
                p_raw = p.category.lower().replace('-', ' ').replace('_', ' ')
                p_cat_tokens = {w.rstrip('s') for w in p_raw.split()}
                
                if not q_cat_tokens.intersection(p_cat_tokens):
                    q_str = filters['category'].lower()
                    p_str = p.category.lower()
                    if "laptop" in q_str and ("macbook" in p_str or "notebook" in p_str): 
                        pass
                    else: 
                        continue
            
            valid_indices.append(i)

        if not valid_indices: return []

        q_emb = self.client.embeddings.create(input=query, model=EMBEDDING_MODEL).data[0].embedding
        valid_embs = self.corpus_embeddings[valid_indices]
        sem_scores = np.dot(valid_embs, np.array(q_emb))

        q_tok = simple_tokenize(query)
        bm25_full = self.bm25.get_scores(q_tok)
        kw_scores = np.array([bm25_full[i] for i in valid_indices])

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

st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .sidebar .sidebar-content { background: #ffffff; }
    h1 { color: #1f2937; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
    .stChatInput { border-radius: 10px; }
    /* Fix for small images in tables */
    td img { max-width: 80px !important; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("AI-Powered Market Research & Price Intelligence Tool")

if "engine" not in st.session_state: st.session_state.engine = None
if "mode" not in st.session_state: st.session_state.mode = "Not Initialized"
if "messages" not in st.session_state: 
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 **Welcome!** Please select a data source from the sidebar to begin analyzing products."}
    ]

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    tab1, tab2 = st.tabs(["📚 Knowledge Base", "🕸️ Live Scanner"])
    
    # TAB 1: Historical Data
    with tab1:
        st.write("Load pre-collected market data.")
        url_in = st.text_input("Dataset URL", value=DEFAULT_CORPUS_URL)
        source_pref = st.radio("Select Source", ["Both", "Daraz", "StarTech"], horizontal=True)
        
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
                                st.session_state.messages = [{"role": "assistant", "content": f"✅ Loaded **{len(docs)}** products from {source_pref}. Data cleaning applied: Prices extracted carefully."}]
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
                        st.session_state.engine = None 
                        new_docs = asyncio.run(crawl_category(scrape_url, source))
                        
                        if new_docs:
                            st.session_state.engine = HybridSearchEngine(new_docs)
                            st.session_state.mode = f"Live Scan ({len(new_docs)} items)"
                            st.success(f"✅ Scanned {len(new_docs)} items.")
                            st.session_state.messages = [{"role": "assistant", "content": f"✅ **Live Scan Complete.** I found {len(new_docs)} items with images."}]
                        else: st.warning("No items found. Please check the URL.")
                    except Exception as e: st.error(f"Scanning Error: {e}")

    if st.session_state.engine:
        st.divider()
        st.markdown(f"**🟢 System Status:** Active")
        st.markdown(f"**📁 Mode:** `{st.session_state.mode}`")
        st.markdown(f"**📦 Products:** {len(st.session_state.engine.products)}")
        with st.expander("Show Active Categories"):
            st.write(st.session_state.engine.categories)
    else:
        st.divider()
        st.markdown("**🔴 System Status:** Offline")

# --- MAIN CHAT ---
for m in st.session_state.messages:
    # Use unsafe_allow_html to render the small images
    with st.chat_message(m["role"]): st.markdown(m["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Ask about prices, specs, or deals..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if not st.session_state.engine:
        err_msg = "⚠️ **System Offline.** Please load the Knowledge Base or Scan a URL from the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": err_msg})
        with st.chat_message("assistant"): st.error(err_msg)
    else:
        with st.chat_message("assistant"):
            client = OpenAI()
            
            # 1. Intent Analysis
            intent_prompt = (
                "You are a Data Analyst. Extract filters from user query.\n"
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
                fail_msg = f"❌ No products matched criteria in {st.session_state.mode}."
                st.warning(fail_msg)
                st.session_state.messages.append({"role": "assistant", "content": fail_msg})
            else:
                # 3. Report Generation
                # We feed the IMAGE URL into the context
                context_str = "\n".join([
                    f"- {r.doc.title} | Price: {r.doc.price_value} BDT | ImageURL: {r.doc.image_url} | Link: {r.doc.url}" 
                    for r in results
                ])
                
                system_instruction = (
                    "You are a Product Guide. \n"
                    "RULES:\n"
                    "1. Present products in a **Table** or List.\n"
                    "2. **IMAGES:** For every product, you MUST display its image in a small size.\n"
                    "   Use this HTML format exactly: <img src='IMAGE_URL' width='100' style='border-radius:5px;' />\n"
                    "3. If ImageURL is empty or 'None', do not show an image tag.\n"
                    "4. Columns: Image, Title, Price, Link.\n"
                    "5. Keep it concise."
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
                    response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                
                response_placeholder.markdown(full_response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full_response})