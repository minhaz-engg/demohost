# import os
# import re
# import time
# import networkx as nx
# import requests
# import streamlit as st
# from dataclasses import dataclass
# from typing import List, Dict, Set, Optional
# from rank_bm25 import BM25Okapi
# from openai import OpenAI
# from dotenv import load_dotenv

# # Load local env vars (fallback for local dev)
# load_dotenv()

# # ----------------------------
# # 1. Configuration & Constants
# # ----------------------------
# PAGE_TITLE = "🛍️ Sigmoix GraphRAG"
# DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/ragapplications/refs/heads/main/refined_dataset/combined_corpus_fixed.md"
# DEFAULT_MODEL = "gpt-4o-mini"
# TOP_K_RETRIEVAL = 20

# STOP_WORDS = {
#     "new", "sale", "best", "exclusive", "offer", "discount", "hot", "top", 
#     "original", "premium", "smart", "super", "mega", "combo", "buy", "get"
# }

# KNOWN_BRANDS = {
#     "apple", "samsung", "xiaomi", "realme", "oneplus", "oppo", "vivo", "google",
#     "hp", "dell", "lenovo", "asus", "acer", "msi", "gigabyte", "intel", "amd",
#     "nvidia", "sony", "canon", "nikon", "fujifilm", "dji", "gopro", "amazon",
#     "logitech", "razer", "corsair", "tp-link", "d-link", "netgear", "tenda",
#     "mikrotik", "cisco", "huion", "wacom", "hoco", "baseus", "anker", "remax",
#     "joyroom", "haylou", "qcy", "soundpeats", "jbl", "bose", "sony", "edifier"
# }

# # ----------------------------
# # 2. Security & Auth Module
# # ----------------------------
# def check_password():
#     """Returns `True` if the user had the correct password."""
    
#     # Check if password is set in secrets, if not, allow open access (Warning: Risky)
#     if "APP_PASSWORD" not in st.secrets:
#         st.warning("⚠️ No password set in secrets. App is open.")
#         return True

#     def password_entered():
#         """Checks whether a password entered by the user is correct."""
#         if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # don't store password
#         else:
#             st.session_state["password_correct"] = False

#     if "password_correct" not in st.session_state:
#         # First run, show input for password.
#         st.text_input(
#             "Enter Business Access Code", type="password", on_change=password_entered, key="password"
#         )
#         st.caption("Protected for Internal Use Only")
#         return False
        
#     elif not st.session_state["password_correct"]:
#         # Password was incorrect, show input again.
#         st.text_input(
#             "Enter Business Access Code", type="password", on_change=password_entered, key="password"
#         )
#         st.error("😕 Password incorrect")
#         return False
        
#     else:
#         # Password was correct.
#         return True

# # ----------------------------
# # 3. Data Structures
# # ----------------------------

# @dataclass
# class ProductDoc:
#     doc_id: str
#     title: str
#     source: str
#     category: str
#     brand: str
#     price_val: float
#     url: str
#     raw_text: str

#     @property
#     def clean_text(self) -> str:
#         return f"{self.title} {self.brand} {self.category} {self.source}"

# # ----------------------------
# # 4. Utilities (FIXED PRICE PARSING)
# # ----------------------------

# class SmartTokenizer:
#     @staticmethod
#     def tokenize(text: str) -> List[str]:
#         text = text.lower()
#         tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text)
#         return tokens

# def infer_brand_robust(title: str, explicitly_tagged_brand: str = None) -> str:
#     if explicitly_tagged_brand and explicitly_tagged_brand.lower() != "generic":
#         return explicitly_tagged_brand.lower()
    
#     if not title:
#         return "generic"
    
#     title_lower = title.lower()
#     tokens = SmartTokenizer.tokenize(title_lower)
    
#     for token in tokens:
#         if token in KNOWN_BRANDS:
#             return token
            
#     if tokens:
#         candidate = tokens[0]
#         if candidate not in STOP_WORDS and len(candidate) > 2:
#             return candidate
            
#     return "generic"

# def parse_price(price_str: str) -> float:
#     """
#     FIXED: Handles cases like '23,850৳ 26,500৳' by taking ONLY the first number.
#     """
#     if not price_str: return 0.0
    
#     # 1. Remove commas and currency symbols
#     clean_str = price_str.replace(',', '').replace('৳', '')
    
#     # 2. Find separate number groups (e.g., '23850' and '26500')
#     nums = re.findall(r'\d+(?:\.\d+)?', clean_str)
    
#     # 3. Return the FIRST number found (the current price)
#     if nums:
#         return float(nums[0])
        
#     return 0.0

# # ----------------------------
# # 5. Data Ingestion
# # ----------------------------

# @st.cache_data(ttl=3600)
# def fetch_raw_data(url: str) -> str:
#     try:
#         resp = requests.get(url, timeout=10)
#         resp.raise_for_status()
#         return resp.text
#     except Exception as e:
#         return ""

# def parse_corpus(text: str) -> List[ProductDoc]:
#     if not text: return []
#     product_blocks = re.split(r'\n---\n', text)
#     docs = []
    
#     re_docid = re.compile(r"\*\*DocID:\*\*\s*`([^`]+)`")
#     re_source = re.compile(r"\*\*Source:\*\*\s*(.+)")
#     re_cat = re.compile(r"\*\*Category:\*\*\s*(.+)")
#     re_brand = re.compile(r"\*\*Brand:\*\*\s*(.+)")
#     re_url = re.compile(r"\*\*URL:\*\*\s*(\S+)")
#     re_price = re.compile(r"\*\*Price:\*\*\s*(.+)")
#     re_title = re.compile(r"^##\s*(.+)", re.MULTILINE)

#     for block in product_blocks:
#         block = block.strip()
#         if not block or "**DocID:**" not in block:
#             continue
            
#         doc_id_m = re_docid.search(block)
#         doc_id = doc_id_m.group(1).strip() if doc_id_m else f"unknown-{hash(block)}"
        
#         title_m = re_title.search(block)
#         title = title_m.group(1).strip() if title_m else "Unknown Product"
        
#         brand_m = re_brand.search(block)
#         raw_brand = brand_m.group(1).strip() if brand_m else None
#         final_brand = infer_brand_robust(title, raw_brand)
        
#         source = re_source.search(block).group(1).strip() if re_source.search(block) else "Unknown"
#         category = re_cat.search(block).group(1).strip().lower() if re_cat.search(block) else "general"
#         url = re_url.search(block).group(1).strip() if re_url.search(block) else ""
#         price_val = parse_price(re_price.search(block).group(1)) if re_price.search(block) else 0.0

#         docs.append(ProductDoc(
#             doc_id=doc_id, title=title, source=source, category=category,
#             brand=final_brand, price_val=price_val, url=url, raw_text=block
#         ))
        
#     return docs

# # ----------------------------
# # 6. GraphRAG Engine
# # ----------------------------

# class GraphRAGIndex:
#     def __init__(self, docs: List[ProductDoc]):
#         self.doc_map = {d.doc_id: d for d in docs}
#         self.graph = nx.Graph()
#         self.docs_daraz = [d for d in docs if 'daraz' in d.source.lower()]
#         self.docs_startech = [d for d in docs if 'startech' in d.source.lower()]
#         self.bm25_daraz = self._build_bm25(self.docs_daraz)
#         self.bm25_startech = self._build_bm25(self.docs_startech)
#         self._build_knowledge_graph(docs)

#     def _build_bm25(self, doc_list: List[ProductDoc]):
#         if not doc_list: return None
#         tokenized_corpus = [SmartTokenizer.tokenize(d.clean_text) for d in doc_list]
#         return BM25Okapi(tokenized_corpus)

#     def _build_knowledge_graph(self, docs: List[ProductDoc]):
#         for doc in docs:
#             self.graph.add_node(doc.doc_id, type='product', source=doc.source)
#             if doc.brand and doc.brand not in ["generic", "unknown", "other"]:
#                 brand_node = f"BRAND:{doc.brand}"
#                 self.graph.add_node(brand_node, type='brand')
#                 self.graph.add_edge(doc.doc_id, brand_node)
#             if doc.category and doc.category not in ["general", "unknown"]:
#                 cat_node = f"CAT:{doc.category}"
#                 self.graph.add_node(cat_node, type='category')
#                 self.graph.add_edge(doc.doc_id, cat_node)

#     def search(self, query: str, total_k: int = 20) -> List[ProductDoc]:
#         half_k = total_k // 2
#         tokenized_query = SmartTokenizer.tokenize(query)
#         daraz_hits = self._query_bm25(self.bm25_daraz, self.docs_daraz, tokenized_query, half_k)
#         startech_hits = self._query_bm25(self.bm25_startech, self.docs_startech, tokenized_query, half_k)
        
#         combined = []
#         max_len = max(len(daraz_hits), len(startech_hits))
#         for i in range(max_len):
#             if i < len(daraz_hits): combined.append(daraz_hits[i])
#             if i < len(startech_hits): combined.append(startech_hits[i])
            
#         expanded_results = []
#         seen_ids = {d.doc_id for d in combined}
#         expanded_results.extend(combined)
        
#         if combined:
#             seeds = combined[:2]
#             graph_hits = []
#             for seed in seeds:
#                 try:
#                     neighbors = list(self.graph.neighbors(seed.doc_id))
#                     for node in neighbors:
#                         siblings = list(self.graph.neighbors(node))
#                         for sib_id in siblings:
#                             if sib_id != seed.doc_id and sib_id not in seen_ids and sib_id in self.doc_map:
#                                 graph_hits.append(self.doc_map[sib_id])
#                                 seen_ids.add(sib_id)
#                 except:
#                     pass
#             expanded_results.extend(graph_hits[:5])
#         return expanded_results

#     def _query_bm25(self, bm25_idx, doc_source, tokenized_query, k):
#         if not bm25_idx or not doc_source: return []
#         scores = bm25_idx.get_scores(tokenized_query)
#         top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
#         return [doc_source[i] for i in top_n if scores[i] > 0.0]

# @st.cache_resource(show_spinner=False)
# def load_search_engine():
#     raw_text = fetch_raw_data(DEFAULT_CORPUS_URL)
#     if not raw_text: return None
#     docs = parse_corpus(raw_text)
#     return GraphRAGIndex(docs)

# # ----------------------------
# # 7. UI / Main App
# # ----------------------------

# def main():
#     st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon="🛍️")

#     # --- SECURITY CHECKPOINT ---
#     if not check_password():
#         st.stop()  # Stop execution if not authenticated
#     # ---------------------------
    
#     # Header
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.title(PAGE_TITLE)
#         st.caption("Engineered for Fairness: Dual-Index Retrieval + Graph Expansion")
#     with col2:
#         st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)

#     # Sidebar
#     with st.sidebar:
#         st.info("System Status")
        
#         with st.spinner("Initializing Knowledge Graph..."):
#             index = load_search_engine()
            
#         if index:
#             st.success(f"✅ System Online")
#             st.metric("Total Products", len(index.doc_map))
#         else:
#             st.error("❌ System Offline (Data Fetch Failed)")
#             st.stop()

#         st.divider()
#         if st.button("🗑️ Reset Conversation"):
#             st.session_state.messages = []
#             st.rerun()

#         if st.button("🧹 Clear Cache (Admin)"):
#             st.cache_resource.clear()
#             st.cache_data.clear()
#             st.rerun()

#     # Chat Interface
#     if "messages" not in st.session_state:
#         st.session_state.messages = [{"role": "assistant", "content": "How can I help you compare prices today?"}]

#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     if prompt := st.chat_input("Ex: 'Best gaming laptop under 100k'"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         start_time = time.time()
#         results = index.search(prompt, total_k=TOP_K_RETRIEVAL)
#         latency = time.time() - start_time

#         if not results:
#             with st.chat_message("assistant"):
#                 st.warning("No matching products found in the database.")
#             return

#         context_str = ""
#         context_display = []
#         for i, doc in enumerate(results, 1):
#             context_str += f"Item {i}: {{ Title: '{doc.title}', Brand: '{doc.brand}', Price: {doc.price_val}, Store: '{doc.source}', URL: '{doc.url}' }}\n"
#             context_display.append(doc)

#         with st.chat_message("assistant"):
#             with st.expander(f"🔍 Retrieved {len(results)} items in {latency:.3f}s", expanded=False):
#                 for doc in context_display:
#                     color = "blue" if "daraz" in doc.source.lower() else "red"
#                     st.markdown(f":{color}[**{doc.source}**] [{doc.title}]({doc.url}) - **{doc.price_val:,.0f}৳**")

#             client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
            
#             if not client:
#                 st.info("💡 API Key missing. Showing raw results above.")
#             else:
#                 stream_box = st.empty()
#                 full_resp = ""
                
#                 system_prompt = (
#                     "You are Sigmoix-AI, a precision Procurement Analyst. "
#                     "Your task is to compare product prices based STRICTLY on the provided Context. "
                    
#                     "### GUIDELINES:\n"
#                     "1. **ZERO HALLUCINATION**: If the answer is not in the Context, state: 'Data not available in current index.' Do NOT make up prices or specs.\n"
#                     "2. **SOURCE TRUTH**: Trust the Context prices over your internal knowledge. Prices change; the Context is the only truth.\n"
#                     "3. **CITATION**: When mentioning a product, you MUST format it as a markdown link: [Product Title](URL).\n"
#                     "4. **COMPARISON**: If a product exists on both Daraz and StarTech, create a Markdown Table comparing them.\n"
#                     "5. **CURRENCY**: Use Bangladesh Taka (৳).\n"
#                     "6. **RECOMMENDATION**: Boldly highlight the best value option.\n\n"
                    
#                     "Output Format:\n"
#                     "- **Analysis**: Brief summary of findings.\n"
#                     "- **Comparison Table**: (If applicable)\n"
#                     "- **Best Deal**: Clear verdict."
#                 )
                
#                 user_message_content = (
#                     f"### Context Data:\n{context_str}\n\n"
#                     f"### User Question:\n{prompt}\n\n"
#                     "### Instruction:\nAnswer the question using ONLY the Context Data above."
#                 )
                
#                 try:
#                     stream = client.chat.completions.create(
#                         model=DEFAULT_MODEL,
#                         messages=[
#                             {"role": "system", "content": system_prompt},
#                             {"role": "user", "content": user_message_content}
#                         ],
#                         stream=True
#                     )
#                     for chunk in stream:
#                         content = chunk.choices[0].delta.content or ""
#                         full_resp += content
#                         stream_box.markdown(full_resp + "▌")
#                     stream_box.markdown(full_resp)
#                     st.session_state.messages.append({"role": "assistant", "content": full_resp})
#                 except Exception as e:
#                     st.error(f"LLM Error: {e}")

# if __name__ == "__main__":
#     main()


# -*- coding: utf-8 -*-
"""
🛍️ MarketIntel: Enterprise E-Commerce Intelligence
==================================================
A Hybrid RAG solution for analyzing product data from Daraz & StarTech.
Designed for Business Teams to retrieve pricing, specs, and market insights.
"""

import os
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
TOP_K_RESULTS = 30  # High retrieval count for better filtering

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

    def search(self, query: str, filters: Dict, top_k: int = TOP_K_RESULTS) -> List[SearchResult]:
        # 1. Hard Filtering (Metadata)
        valid_indices = []
        for i, p in enumerate(self.products):
            # Price Logic
            if filters.get('max_price') and p.price_value > filters['max_price']: continue
            if filters.get('min_price') and p.price_value < filters['min_price']: continue
            # Category Logic (Smart Fuzzy Match)
            if filters.get('category'):
                q_cat = filters['category'].lower()
                p_cat = p.category.lower()
                if q_cat not in p_cat and p_cat not in q_cat:
                    # Exception for common synonyms
                    if "laptop" in q_cat and ("macbook" in p_cat or "notebook" in p_cat): pass
                    else: continue
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
            # 1. Intent Recognition (Hidden from Business Users for cleanliness)
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