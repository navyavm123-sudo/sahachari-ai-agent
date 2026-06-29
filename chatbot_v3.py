import os
import re
import asyncio
import logging
import json
import hashlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from collections import defaultdict
from typing import Optional

import httpx
import torch
import uvicorn
from cachetools import TTLCache
from chromadb import PersistentClient
from dotenv import load_dotenv
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline

try:
    from FlagEmbedding import FlagReranker
except Exception:
    FlagReranker = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sahachari")

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH         = os.path.join(BASE_DIR, "chroma_db")
NESTJS_BACKEND_URL  = os.getenv("NESTJS_BACKEND_URL", "http://127.0.0.1:3000")
MONGO_URI           = os.getenv("MONGO_URI")
MONGO_DB            = os.getenv("MONGO_DB", "sahachari_db")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 24 * 7)))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "-2.0"))
QWEN_MODEL_NAME     = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
FRONTEND_ORIGIN     = os.getenv("FRONTEND_ORIGIN", "http://localhost:3001")

if not MONGO_URI:
    raise RuntimeError(
        "MONGO_URI is not set. Add it to your .env file:\n"
        "MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/dbname"
    )

ALLOWED_ORIGINS = [o.strip() for o in FRONTEND_ORIGIN.split(",") if o.strip()]
OK_STATUSES     = (200, 201, 204)

# ─────────────────────────────────────────────────────────────────────────────
# Caches
# ─────────────────────────────────────────────────────────────────────────────

_product_cache:        TTLCache = TTLCache(maxsize=500, ttl=300)
_session_memory_cache: TTLCache = TTLCache(maxsize=300, ttl=600)
_faq_answer_cache:     TTLCache = TTLCache(maxsize=200, ttl=1800)
_catalog_cache:        TTLCache = TTLCache(maxsize=50,  ttl=120)

# ─────────────────────────────────────────────────────────────────────────────
# Service name normalisation
# ─────────────────────────────────────────────────────────────────────────────

SERVICE_NAME_NORMALISE = {
    "dishwashing":       "Dishwash",
    "dish washing":      "Dishwash",
    "dish wash":         "Dishwash",
    "dishwash service":  "Dishwash",
    "dishwash":          "Dishwash",
    "washing":           "Dishwash",
    "cleaning service":  "Cleaning",
    "cleaning":          "Cleaning",
    "clean":             "Cleaning",
    "house clean":       "Cleaning",
    "home clean":        "Cleaning",
    "room clean":        "Cleaning",
    "kitchen clean":     "Cleaning",
    "sweeping":          "Sweeping",
    "sweep":             "Sweeping",
    "mopping":           "Mopping",
    "mopping service":   "Mopping",
    "mop":               "Mopping",
}

SERVICE_ITEM_NAMES = {
    "dishwash", "cleaning", "sweeping", "mopping", "mop", "clean", "sweep",
}


def normalise_service_name(name: str) -> str:
    lower = name.lower().strip()
    for variant, canonical in SERVICE_NAME_NORMALISE.items():
        if lower == variant or lower.startswith(variant):
            return canonical
    return name.title()


def is_service_item(item_name: str) -> bool:
    lower = item_name.lower().strip()
    return (
        any(lower == s or lower.startswith(s) for s in SERVICE_ITEM_NAMES)
        or bool(SERVICE_QUERY_PATTERN.search(lower))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Auth helper
# ─────────────────────────────────────────────────────────────────────────────

def clean_auth_token(auth_token: str) -> str:
    if not auth_token:
        return auth_token
    token = auth_token.strip()
    lower = token.lower()
    if lower.count("bearer") > 1:
        return "Bearer " + token.split(" ")[-1]
    if not lower.startswith("bearer "):
        return "Bearer " + token
    return "Bearer " + token[len("bearer "):]


# ─────────────────────────────────────────────────────────────────────────────
# Default entity structure
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ENTITIES = {
    "household_size":     None,
    "budget_preference":  None,
    "delivery_time_pref": None,
    "allergies":          [],
    "complaint_items":    [],
    "known_address":      None,
    "last_seen_products": [],
}

ENTITY_PATTERNS = {
    "household_size": re.compile(
        r"\b(family of (\d+)|(\d+)\s*(?:people|persons?|members?)|"
        r"just (?:me|myself)|living alone|two of us|three of us|four of us)\b",
        re.IGNORECASE,
    ),
    "budget": re.compile(
        r"\b(budget|cheap|affordable|economical|premium|organic|best quality)\b",
        re.IGNORECASE,
    ),
    "allergy": re.compile(
        r"\b(?:allergic to|allergy to|cannot eat|can'?t eat|intolerant to)\s+([a-z ]+)",
        re.IGNORECASE,
    ),
    "delivery_time": re.compile(
        r"\b(?:deliver (?:in the |during the )?|prefer (?:delivery in )?)"
        r"(morning|afternoon|evening|night)\b",
        re.IGNORECASE,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Session store
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SESSION = {
    "wishlist":             [],
    "pending_item":         None,
    "pending_intent":       None,
    "last_browse_category": None,
    "recipe_items":         None,
    "session_budget":       None,
    "last_mentioned_items": [],
    "slot_items":           [],
    "updated_at":           None,
    "analytics": {
        "intents_this_session": [],
        "items_viewed":         [],
        "cart_abandonment":     0,
        "session_start":        None,
    },
    "_greeted_with_name": False,
}


class SessionStore:
    def __init__(self, mongo_db):
        self._collection = mongo_db["chat_sessions"]
        self._cache: TTLCache = TTLCache(maxsize=500, ttl=3600)

    async def ensure_indexes(self):
        try:
            await self._collection.create_index("session_id", unique=True)
            await self._collection.create_index(
                "updated_at", expireAfterSeconds=SESSION_TTL_SECONDS,
            )
        except Exception as e:
            log.warning(f"Could not create session indexes: {e}")

    async def get(self, session_id: str) -> dict:
        if session_id in self._cache:
            return self._cache[session_id]
        doc = await self._collection.find_one({"session_id": session_id})
        if doc is None:
            session = {**DEFAULT_SESSION, "wishlist": [], "session_id": session_id,
                       "analytics": {**DEFAULT_SESSION["analytics"],
                                     "session_start": datetime.now(timezone.utc).isoformat()}}
        else:
            session = {
                "wishlist":             doc.get("wishlist", []),
                "pending_item":         doc.get("pending_item"),
                "pending_intent":       doc.get("pending_intent"),
                "last_browse_category": doc.get("last_browse_category"),
                "recipe_items":         doc.get("recipe_items"),
                "session_budget":       doc.get("session_budget"),
                "last_mentioned_items": doc.get("last_mentioned_items", []),
                "slot_items":           doc.get("slot_items", []),
                "session_id":           session_id,
                "analytics":            doc.get("analytics", {**DEFAULT_SESSION["analytics"]}),
                "_greeted_with_name":   doc.get("_greeted_with_name", False),
            }
        self._cache[session_id] = session
        return session

    async def save(self, session_id: str, session: dict):
        self._cache[session_id] = session
        try:
            await self._collection.update_one(
                {"session_id": session_id},
                {"$set": {
                    "wishlist":             session.get("wishlist", []),
                    "pending_item":         session.get("pending_item"),
                    "pending_intent":       session.get("pending_intent"),
                    "last_browse_category": session.get("last_browse_category"),
                    "recipe_items":         session.get("recipe_items"),
                    "session_budget":       session.get("session_budget"),
                    "last_mentioned_items": session.get("last_mentioned_items", []),
                    "slot_items":           session.get("slot_items", []),
                    "analytics":            session.get("analytics", {}),
                    "_greeted_with_name":   session.get("_greeted_with_name", False),
                    "updated_at":           datetime.now(timezone.utc),
                }},
                upsert=True,
            )
        except Exception as e:
            log.warning(f"Failed to persist session {session_id}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Memory store
# ─────────────────────────────────────────────────────────────────────────────

class MemoryStore:
    MAX_HISTORY       = 20
    SUMMARY_THRESHOLD = 14

    def __init__(self, mongo_db):
        self._collection = mongo_db["user_memory"]
        self._cache: TTLCache = TTLCache(maxsize=500, ttl=3600)

    async def ensure_indexes(self):
        try:
            await self._collection.create_index("session_id", unique=True)
        except Exception as e:
            log.warning(f"Could not create memory indexes: {e}")

    async def get(self, session_id: str) -> dict:
        if session_id in self._cache:
            return self._cache[session_id]
        doc = await self._collection.find_one({"session_id": session_id})
        if doc is None:
            memory = self._blank_memory(session_id)
        else:
            memory = {
                "session_id":           session_id,
                "conversation_history": doc.get("conversation_history", []),
                "conversation_summary": doc.get("conversation_summary", ""),
                "preferences":          doc.get("preferences", {}),
                "frequent_items":       doc.get("frequent_items", {}),
                "item_quantities":      doc.get("item_quantities", {}),
                "last_order_summary":   doc.get("last_order_summary"),
                "name":                 doc.get("name"),
                "location":             doc.get("location"),
                "preferred_language":   doc.get("preferred_language", "en"),
                "sentiment_score":      doc.get("sentiment_score", 0.0),
                "total_sessions":       doc.get("total_sessions", 1),
                "first_seen":           doc.get("first_seen"),
                "last_seen":            doc.get("last_seen"),
                "brand_preferences":    doc.get("brand_preferences", {}),
                "spice_preference":     doc.get("spice_preference", "medium"),
                "organic_preference":   doc.get("organic_preference", False),
                "budget_hint":          doc.get("budget_hint", None),
                "entities":             doc.get("entities", {**DEFAULT_ENTITIES}),
            }
        self._cache[session_id] = memory
        return memory

    def _blank_memory(self, session_id: str) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "session_id":           session_id,
            "conversation_history": [],
            "conversation_summary": "",
            "preferences":          {},
            "frequent_items":       {},
            "item_quantities":      {},
            "last_order_summary":   None,
            "name":                 None,
            "location":             None,
            "preferred_language":   "en",
            "sentiment_score":      0.0,
            "total_sessions":       1,
            "first_seen":           now,
            "last_seen":            now,
            "brand_preferences":    {},
            "spice_preference":     "medium",
            "organic_preference":   False,
            "budget_hint":          None,
            "entities":             {**DEFAULT_ENTITIES,
                                     "allergies": [], "complaint_items": [],
                                     "last_seen_products": []},
        }

    async def save(self, session_id: str, memory: dict):
        self._cache[session_id] = memory
        memory["last_seen"] = datetime.now(timezone.utc).isoformat()
        try:
            await self._collection.update_one(
                {"session_id": session_id},
                {"$set": {
                    "conversation_history": memory["conversation_history"],
                    "conversation_summary": memory.get("conversation_summary", ""),
                    "preferences":          memory["preferences"],
                    "frequent_items":       memory["frequent_items"],
                    "item_quantities":      memory.get("item_quantities", {}),
                    "last_order_summary":   memory["last_order_summary"],
                    "name":                 memory.get("name"),
                    "location":             memory.get("location"),
                    "preferred_language":   memory.get("preferred_language", "en"),
                    "sentiment_score":      memory.get("sentiment_score", 0.0),
                    "total_sessions":       memory.get("total_sessions", 1),
                    "first_seen":           memory.get("first_seen"),
                    "last_seen":            memory.get("last_seen"),
                    "brand_preferences":    memory.get("brand_preferences", {}),
                    "spice_preference":     memory.get("spice_preference", "medium"),
                    "organic_preference":   memory.get("organic_preference", False),
                    "budget_hint":          memory.get("budget_hint"),
                    "entities":             memory.get("entities", {}),
                    "updated_at":           datetime.now(timezone.utc),
                }},
                upsert=True,
            )
        except Exception as e:
            log.warning(f"Failed to persist memory {session_id}: {e}")

    async def maybe_summarise(self, memory: dict):
        history = memory["conversation_history"]
        if len(history) <= self.SUMMARY_THRESHOLD * 2:
            return

        to_summarise = history[:self.SUMMARY_THRESHOLD]
        keep         = history[self.SUMMARY_THRESHOLD:]

        if qwen_pipeline is not None:
            try:
                loop        = asyncio.get_running_loop()
                dialog_text = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in to_summarise
                )
                prompt = [{
                    "role": "user",
                    "content": (
                        "Summarise the following grocery chat in under 5 bullet points. "
                        "Focus on: items ordered, preferences stated, complaints, "
                        "user name, allergies, household size, and important decisions.\n\n"
                        + dialog_text
                    ),
                }]
                def _gen():
                    return qwen_pipeline(
                        prompt, max_new_tokens=200, return_full_text=False, do_sample=False,
                    )[0]["generated_text"].strip()
                summary_text = await loop.run_in_executor(None, _gen)
            except Exception as e:
                log.warning(f"Summarisation LLM failed: {e}")
                summary_text = self._structured_summary(to_summarise)
        else:
            summary_text = self._structured_summary(to_summarise)

        existing = memory.get("conversation_summary", "")
        memory["conversation_summary"] = (
            (existing + "\n\n" + summary_text).strip() if existing else summary_text
        )
        memory["conversation_history"] = keep
        log.info(f"Summarised {len(to_summarise)} turns → {len(keep)} remain in history.")

    @staticmethod
    def _structured_summary(turns: list) -> str:
        lines = ["[Auto-summary of earlier conversation]"]
        for m in turns:
            role    = "User" if m["role"] == "user" else "Bot"
            content = m["content"][:120].replace("\n", " ")
            lines.append(f"  {role}: {content}")
        return "\n".join(lines)

    def add_turn(self, memory: dict, user_msg: str, assistant_msg: str):
        memory["conversation_history"].append({"role": "user",      "content": user_msg})
        memory["conversation_history"].append({"role": "assistant", "content": assistant_msg})
        if len(memory["conversation_history"]) > self.MAX_HISTORY * 2:
            memory["conversation_history"] = memory["conversation_history"][-(self.MAX_HISTORY * 2):]

    def record_item_ordered(self, memory: dict, item_name: str, quantity: float = 1.0, unit: str = "piece"):
        key = item_name.strip().title()
        memory["frequent_items"][key] = memory["frequent_items"].get(key, 0) + 1
        if quantity and quantity > 0:
            memory.setdefault("item_quantities", {})[key] = {"qty": quantity, "unit": unit}

    def extract_entities(self, memory: dict, query: str):
        entities = memory.setdefault("entities", {**DEFAULT_ENTITIES,
                                                   "allergies": [], "complaint_items": [],
                                                   "last_seen_products": []})
        m = ENTITY_PATTERNS["household_size"].search(query)
        if m:
            entities["household_size"] = m.group(0).strip()

        m = ENTITY_PATTERNS["allergy"].search(query)
        if m:
            allergen = m.group(1).strip().title()
            if allergen not in entities.get("allergies", []):
                entities.setdefault("allergies", []).append(allergen)
                log.info(f"Allergy recorded for session: {allergen}")

        m = ENTITY_PATTERNS["delivery_time"].search(query)
        if m:
            entities["delivery_time_pref"] = m.group(1).lower()

        m = ENTITY_PATTERNS["budget"].search(query.lower())
        if m:
            entities["budget_preference"] = m.group(1).lower()

        if COMPLAINT_PATTERNS.search(query):
            item = extract_item_name(query)
            if item and item not in entities.get("complaint_items", []):
                entities.setdefault("complaint_items", []).append(item)

    def extract_preferences(self, memory: dict, query: str):
        self.extract_entities(memory, query)

        q        = query.lower()
        likes    = memory["preferences"].setdefault("likes", [])
        dislikes = memory["preferences"].setdefault("dislikes", [])
        diet     = memory["preferences"].get("diet", [])
        if not isinstance(diet, list):
            diet = [diet]

        like_match = re.search(r"\bi (?:love|like|enjoy|prefer|always buy)\s+([a-z ]+)", q)
        if like_match:
            item = like_match.group(1).strip().title()
            if item not in likes:
                likes.append(item)

        dislike_match = re.search(r"\bi (?:hate|dislike|don'?t like|avoid|never buy)\s+([a-z ]+)", q)
        if dislike_match:
            item = dislike_match.group(1).strip().title()
            if item not in dislikes:
                dislikes.append(item)

        for tag in ["vegan", "vegetarian", "gluten-free", "diabetic", "keto", "halal", "organic"]:
            if tag in q and tag not in diet:
                diet.append(tag)

        if re.search(r"\b(organic|natural|chemical[- ]free|pesticide[- ]free)\b", q):
            memory["organic_preference"] = True
        if re.search(r"\b(very spicy|extra spicy|hot)\b", q):
            memory["spice_preference"] = "high"
        elif re.search(r"\b(not spicy|mild|less spicy|no spice)\b", q):
            memory["spice_preference"] = "low"

        budget_match = re.search(
            r"(?:budget|within|under|below|at most|max)\s*(?:of\s*)?(?:₹|rs\.?)?\s*(\d+)", q
        )
        if budget_match:
            try:
                memory["budget_hint"] = float(budget_match.group(1))
            except Exception:
                pass

        name_match = re.search(r"\b(?:i am|i'm|my name is|call me)\s+([a-z][a-z]+)\b", q)
        if name_match:
            candidate = name_match.group(1).strip().title()
            if candidate not in {"Fine", "Good", "Okay", "Well", "Here", "Back", "Ready"}:
                memory["name"] = candidate

        loc_match = re.search(
            r"(?:i(?:'m| am) (?:in|at|from)|deliver(?:ing)? to|my area is|near)\s+([a-z][a-z\s,]+?)(?:\.|,|$)", q,
        )
        if loc_match:
            memory["location"] = loc_match.group(1).strip().title()

        pos   = len(re.findall(r"\b(great|thanks|perfect|love|good|awesome|excellent|happy)\b", q))
        neg   = len(re.findall(r"\b(bad|terrible|hate|wrong|broken|issue|problem|complaint)\b", q))
        delta = (pos - neg) * 0.1
        memory["sentiment_score"] = round(
            max(-1.0, min(1.0, memory.get("sentiment_score", 0.0) + delta)), 2
        )

        memory["preferences"]["likes"]    = likes
        memory["preferences"]["dislikes"] = dislikes
        memory["preferences"]["diet"]     = diet

    def get_safe_suggestions(self, memory: dict, candidates: list[str]) -> list[str]:
        entities        = memory.get("entities", {})
        blocked         = (entities.get("complaint_items", [])
                           + entities.get("allergies", []))
        blocked_lower   = {b.lower() for b in blocked}
        return [c for c in candidates if not any(b in c.lower() for b in blocked_lower)]

    def build_llm_context(self, memory: dict, session: dict, current_query: str) -> list[dict]:
        entities = memory.get("entities", {})
        parts    = [
            "You are Sahachari, a Kerala grocery delivery assistant.",
            "Sahachari sells: vegetables, fruits, dairy, grains, oils, groceries, "
            "snacks, beverages, fast food, homemade food, services, and rent items.",
        ]

        if memory.get("name"):
            parts.append(f"Customer name: {memory['name']}.")
        if entities.get("household_size"):
            parts.append(f"Household: {entities['household_size']}.")
        if entities.get("allergies"):
            parts.append(
                f"ALLERGIES — NEVER suggest these items: {', '.join(entities['allergies'])}."
            )
        if entities.get("complaint_items"):
            parts.append(
                f"Do not proactively suggest: {', '.join(entities['complaint_items'])}."
            )
        if memory.get("organic_preference"):
            parts.append("User prefers organic products.")
        sp = memory.get("spice_preference", "medium")
        if sp != "medium":
            parts.append(f"Spice preference: {sp}.")
        if memory.get("budget_hint"):
            parts.append(f"Typical budget: ₹{memory['budget_hint']:.0f}.")
        if memory.get("frequent_items"):
            top = sorted(memory["frequent_items"].items(), key=lambda x: x[1], reverse=True)[:3]
            parts.append(f"Frequently orders: {', '.join(k for k, _ in top)}.")
        if session.get("last_browse_category"):
            parts.append(f"Currently browsing: {session['last_browse_category']}.")

        parts += [
            "STRICT RULES:",
            "1. Answer ONLY from the provided Context.",
            "2. NEVER invent phone numbers, promo codes, or features not in Context.",
            "3. If Context is empty or irrelevant say: 'I cannot help with that through Sahachari.'",
            "4. Keep answer under 2 sentences.",
        ]

        messages = [{"role": "system", "content": " ".join(parts)}]

        if memory.get("conversation_summary"):
            messages.append({
                "role":    "system",
                "content": f"[Earlier session summary]:\n{memory['conversation_summary']}",
            })

        recent = memory["conversation_history"][-12:]
        messages.extend(recent)

        return messages

    def build_context_summary(self, memory: dict) -> str:
        parts = []
        if memory.get("conversation_summary"):
            parts.append("Past session summary:\n" + memory["conversation_summary"])
        freq = memory.get("frequent_items", {})
        if freq:
            top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
            parts.append("Frequently ordered: " + ", ".join(f"{k}({v}x)" for k, v in top))
        prefs = memory.get("preferences", {})
        if prefs.get("likes"):    parts.append("Likes: "    + ", ".join(prefs["likes"]))
        if prefs.get("dislikes"): parts.append("Dislikes: " + ", ".join(prefs["dislikes"]))
        if prefs.get("diet"):
            d = prefs["diet"]
            parts.append("Diet: " + (", ".join(d) if isinstance(d, list) else d))
        entities = memory.get("entities", {})
        if entities.get("allergies"):
            parts.append("Allergies: " + ", ".join(entities["allergies"]))
        if entities.get("household_size"):
            parts.append("Household: " + entities["household_size"])
        if memory.get("organic_preference"):
            parts.append("Prefers organic products.")
        sp = memory.get("spice_preference", "medium")
        if sp != "medium":
            parts.append(f"Spice preference: {sp}.")
        if memory.get("budget_hint"):
            parts.append(f"Typical budget: ₹{memory['budget_hint']:.0f}.")
        if memory.get("location"):
            parts.append(f"Location: {memory['location']}.")
        last = memory.get("last_order_summary")
        if last:
            parts.append(f"Last order: {last}")
        if memory.get("name"):
            parts.append(f"User name: {memory['name']}")
        return "\n".join(parts)

    def get_top_items(self, memory: dict, n: int = 3) -> list[str]:
        freq = memory.get("frequent_items", {})
        if not freq:
            return []
        safe = self.get_safe_suggestions(memory, list(freq.keys())) if memory_store else list(freq.keys())
        return [k for k, _ in sorted(
            [(k, freq[k]) for k in safe], key=lambda x: x[1], reverse=True
        )[:n]]

    def get_preferred_quantity(self, memory: dict, item_name: str) -> tuple:
        key = item_name.strip().title()
        rec = memory.get("item_quantities", {}).get(key)
        if rec:
            return rec.get("qty"), rec.get("unit", "piece")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Constants / lexicons
# ─────────────────────────────────────────────────────────────────────────────

INTENT_KEYWORDS = {
    "view_cart": [
        "view cart", "show cart", "show my cart", "my cart", "what's in my cart",
        "cart items", "see cart", "display cart", "check cart", "open cart",
    ],
    "wishlist_remove": [
        "remove from wishlist", "delete from wishlist", "remove wishlist",
        "clear wishlist", "empty wishlist",
    ],
    "wishlist": [
        "show wishlist", "view wishlist", "my wishlist",
        "save for later", "add to wishlist",
    ],
    "status": [
        "order status", "track my order", "where is my order", "track order",
        "track my current order", "current order status",
        "where is my current order", "track current order",
        "show my orders", "show orders", "list of orders", "list my orders",
        "my orders", "all my orders", "order history", "past orders",
        "previous orders", "show all orders",
        "show my last order", "show last order", "show my order",
        "show my recent order", "last order", "recent order", "my last order",
        "status of all orders", "status of orders", "show all orders",
        "show me all orders", "all orders", "list all orders",
    ],
    "cancel":     ["cancel my order", "cancel order", "stop my order", "cancel"],
    "browse": [
        "what do you have", "available items", "list all products",
        "show catalogue", "show me all",
        "show the available products", "show available products",
        "all products", "all items", "list products", "list items",
    ],
    "clear_cart": [
        "clear cart", "clear my cart", "empty cart", "empty my cart",
        "remove everything from cart", "remove everything",
        "delete everything from cart", "remove all from cart",
        "delete all items", "clear all items",
    ],
    "delete_cart": [
        "remove from cart", "delete from cart", "remove item", "delete item",
        "take out", "remove my", "delete my",
    ],
    "checkout": [
        "place my order", "confirm order", "checkout", "place order",
        "proceed to payment",
    ],
    "product_info": [
        "find", "search for", "look for", "is available", "available today",
        "price of", "how much is", "how much does", "how much do",
        "tell me about", "details of", "what is the price",
        "show me", "is there", "do you have", "in stock", "stock of", "cost of",
    ],
    "order": [
        "order my wishlist", "buy", "add to cart", "get me",
        "i want", "i need", "want to buy",
    ],
    "service_booking": [
        "book a service", "book service", "schedule service",
        "book dishwashing", "book cleaning", "book dish wash",
        "schedule cleaning", "schedule dishwashing",
        "book a cleaning", "book a dishwash", "book a dish wash",
        "schedule a service", "schedule a cleaning", "schedule a dishwash",
    ],
}

INTENT_LABELS = [
    "greeting", "courtesy", "view_cart", "wishlist", "wishlist_remove",
    "status", "cancel", "browse", "clear_cart", "delete_cart", "checkout",
    "product_info", "order", "service_booking", "preference", "rag",
    "complaint", "reorder", "recipe", "clarify", "update_cart", "repair",
]

CLOSING_WORDS  = {"thank you", "thanks", "thx", "thank u", "appreciate it"}
ORDER_ID_RE    = re.compile(r"#?\b([a-f0-9]{24})\b")
NON_ITEM_WORDS = {"last", "latest", "recent", "previous", "first", "current", "new"}

CATEGORY_SYNONYMS = {
    "fruit":     ["fruit", "fruits"],
    "vegetable": ["vegetable", "vegetables", "veg"],
    "leafy":     ["leafy", "spinach", "methi", "palak", "coriander", "greens",
                  "fenugreek", "amaranth", "mustard", "curry leaves"],
    "food":      ["food", "homemade", "home made", "home-made"],
    "snacks":    ["snack", "snacks"],
    "fastfood":  ["fast food", "fastfood"],
    "groceries": ["grocery", "groceries"],
    "service":   ["service", "services"],
    "rent":      ["rent", "rental"],
    "beverages": ["beverage", "beverages", "drink", "drinks", "juice"],
}

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "half": 0.5, "dozen": 12, "a": 1, "an": 1,
}

# FIX: Put multi-char units before single-char l/g to avoid partial matches
# Also use \b word boundaries on single-char units
UNIT_NORMALISE = {
    "kg": "kg", "kgs": "kg", "kilo": "kg", "kilogram": "kg", "kilograms": "kg",
    "kilos": "kg",
    "g": "g", "gm": "g", "gram": "g", "grams": "g", "gms": "g",
    "l": "litre", "ltr": "litre", "liter": "litre", "litre": "litre", "litres": "litre",
    "liters": "litre",
    "ml": "ml", "mls": "ml",
    "dozen": "dozen",
    "piece": "piece", "pc": "piece", "pieces": "piece", "pcs": "piece",
    "packet": "packet", "pack": "packet", "packets": "packet",
    "bunch": "bunch", "bunches": "bunch",
    "box": "box", "boxes": "box",
}

# FIX: Multi-char alternatives first, then single-char with word boundary
UNIT_WORDS_RE = (
    r"(kilograms?|kilos?|kg|litres?|liters?|ltr|grams?|gms?|mls?|ml|"
    r"dozen|pieces?|pcs?|packets?|bunches?|boxes?|box|"
    r"l(?=\s|$)|g(?=\s|$))"
)

STOP_WORDS = {
    "order", "food", "delivery", "item", "please", "want", "need", "get", "buy",
    "some", "and", "the", "of", "from", "in", "all", "available", "show", "list",
    "under", "below", "rs", "rupees", "now", "any", "for", "about",
    "price", "cost", "what", "is", "are", "do", "does", "you", "have", "there", "how", "much",
    "into", "my", "cart", "to", "onto", "me",
    "tell", "premium", "quality", "best", "good", "great", "nice", "fine", "top",
    "grade", "high", "pure", "natural", "local", "locally",
    "details", "info", "information",
    "it", "this", "that", "them", "those", "these",
    "someone", "person", "people", "anyone", "i",
}

# FIX: Updated NUM_WORDS_RE to support leading-dot decimals like .5
NUM_WORDS_RE = r"(\d*\.?\d+|\b(?:half|one|two|three|four|five|six|seven|eight|nine|ten|dozen|a|an)\b)"

TYPO_MAP = {
    "ttoday": "today", "todday": "today", "todat": "today",
    "avaliable": "available", "availble": "available", "availabel": "available",
    "prise": "price", "prce": "price",
    "hwo": "how", "whats": "what is",
    "onions": "onion", "tomatos": "tomato", "potatoe": "potato",
    "carrots": "carrot", "apples": "apple", "mangos": "mango",
}

PRODUCT_QUERY_PHRASES = [
    "can i buy", "can i get", "can i order",
    "what is the price of", "price of", "how much is", "tell me about",
    "details of", "info on", "information about",
    "do you have", "is there", "show me", "find", "search for", "look for",
    "how much does", "how much do",
]

PRODUCT_FILLER_WORDS = {
    "today", "available", "please", "the", "a", "an", "is", "are", "any", "some",
    "about", "for", "now", "details", "me", "show", "all", "under", "below", "less",
    "than", "at", "most", "maximum", "max", "within", "rs", "rupees", "what", "much",
    "price", "cost", "do", "does", "you", "have", "there", "how", "find", "search", "look",
    "sahachari", "buy", "can", "get", "from", "tell",
    "premium", "quality", "best", "good", "great", "nice", "fine", "top",
    "grade", "high", "pure", "natural", "local", "locally",
    "details", "of", "info", "information",
    "tomorrow", "morning", "evening", "afternoon", "tonight", "next", "week",
    "book", "schedule", "booking",
    "someone", "person", "people", "anyone", "i", "need", "want",
}

CART_REMOVE_PHRASES = [
    "remove from cart", "delete from cart", "remove item", "delete item",
    "take out", "remove my", "delete my", "don't want", "dont want",
    "remove", "delete", "drop", "from my cart", "from the cart",
]

SERVICE_DENIAL_PATTERNS = re.compile(
    r"\b(domestic help|domestic worker|domestic assistance|household help|"
    r"part[- ]?time help|babysitt\w*|nanny|caretaker|elder care|"
    r"plumbing|electrician|painting|pest control|"
    r"maid|cook|driver|technician|handyman)\b",
    re.IGNORECASE,
)

SERVICE_DENIAL_RESPONSE = (
    "🛒 That specific service isn't offered by Sahachari.\n\n"
    "Sahachari provides the following categories:\n"
    "  • 🥦 Vegetables & Fruits\n"
    "  • 🍱 Food & Home Made\n"
    "  • 🛒 Groceries & Snacks\n"
    "  • 🍔 Fast Food\n"
    "  • 🥤 Beverages\n"
    "  • 🔧 Service\n"
    "  • 📦 Rent\n\n"
    "Try: 'show me services' or 'I want 2 kg onions'."
)

BLOCKED_RAG_TOPICS = re.compile(
    r"\b(domestic help|domestic worker|domestic assistance|household help|"
    r"part[- ]?time help|babysitt\w*|nanny|caretaker|elder care|"
    r"maid|plumber|electrician|technician|handyman)\b",
    re.IGNORECASE,
)

META_QUESTION_PATTERNS = re.compile(
    r"\b(memory capacity|memory power|how much memory|what do you remember|"
    r"what can you remember|do you have memory|your memory|"
    r"what do you know about me|what have you learned|"
    r"forget me|clear my memory|reset memory)\b",
    re.IGNORECASE,
)

COMPLAINT_PATTERNS = re.compile(
    r"\b(rotten|spoiled|expired|wrong order|wrong item|someone else|refund|damaged|"
    r"missing item|not received|didn't arrive|bad quality|complaint|issue with|"
    r"problem with|didn't show up|no show|late delivery|overcharged|charged twice|"
    r"double charged|broken|leaking|dirty|unhygienic)\b",
    re.IGNORECASE,
)

REORDER_PATTERNS = re.compile(
    r"\b(reorder|order again|same as (last|before|previous)|order (the )?same|"
    r"repeat (my |last |previous )?order|get (the )?usual|my usual)\b",
    re.IGNORECASE,
)

RECIPE_PATTERNS = re.compile(
    r"\b(ingredients? for|stuff for (making|cooking)|items? (needed )?for|"
    r"what (do i need|should i buy) (to make|for)|recipe for|cooking)\b",
    re.IGNORECASE,
)

REPAIR_SIGNALS = re.compile(
    r"\b(that'?s? (wrong|not right|incorrect|not what i (meant|asked|wanted|said))|"
    r"no[,.]?\s+(i (said|meant|asked|wanted)|that'?s? not)|"
    r"you (misunderstood|got it wrong)|wrong (item|product|thing)|"
    r"not what i (want|wanted|need|needed|asked))\b",
    re.IGNORECASE,
)

META_QUESTION_RESPONSE = (
    "🧠 I remember the following about you across sessions:\n\n"
    "  • Your conversation history (last 20 messages)\n"
    "  • Items you order frequently\n"
    "  • Your food preferences and dietary needs\n"
    "  • Allergies and items to avoid\n"
    "  • Your last order summary\n"
    "  • Your name (if you've told me)\n\n"
    "This helps me personalise your grocery experience. "
    "Your data is stored securely and used only to assist you better!"
)

VAGUE_FOLLOWUPS = {
    "book it", "ok", "okay", "sure", "yes", "no", "maybe", "alright",
    "what you mean", "what do you mean", "huh", "what", "really",
    "and then", "then what", "go on", "continue", "tell me more",
}

PRONOUN_REFS = {
    "that", "it", "this", "the first one", "the second one",
    "the last one", "the cheapest one", "that one", "this one",
}

CHEAPEST_PATTERNS = re.compile(
    r"\b(cheapest|lowest price|most expensive|highest price|"
    r"which.*cheap|which.*expensive|which one|the cheapest)\b",
    re.IGNORECASE,
)

NON_GROCERY_KEYWORDS = {
    "shawarma", "burger", "biryani", "sandwich", "porota", "cake",
    "roll", "quarter", "bags",
}

_SHORT_QUERY_EXEMPT = {
    "cart", "order", "price", "buy", "add", "checkout", "cancel",
    "wishlist", "status", "browse", "track", "remove", "delete",
    "clear", "empty", "place", "confirm", "proceed", "fruit", "fruits",
    "vegetable", "vegetables", "veg", "leafy", "greens", "beverage",
    "drinks", "snack", "snacks", "service", "rent",
}

KNOWN_FRUITS = {
    "mango", "apple", "banana", "orange", "grape", "papaya",
    "pineapple", "watermelon", "guava", "pomegranate", "lemon",
    "lime", "melon", "kiwi", "chikoo", "sapota", "jackfruit",
}
KNOWN_VEGETABLES = {
    "onion", "carrot", "tomato", "potato", "cabbage", "cauliflower",
    "brinjal", "eggplant", "spinach", "methi", "palak", "coriander",
    "beans", "peas", "cucumber", "capsicum", "pepper", "ladies finger",
    "okra", "beetroot", "radish", "garlic", "ginger", "gourd", "drumstick",
    "leafy", "greens", "fenugreek", "amaranth", "mustard", "curry leaves",
}

# FIX: Added end-of-string "now" pattern
BUY_NOW_PATTERNS = re.compile(
    r"\b(buy now|order now|purchase now|get it now|want to buy now"
    r"|i want\b.{0,30}\bnow\b"
    r"|buy\b.{0,30}\bnow\b"
    r"|order\b.{0,30}\bnow\b"
    r"|\bnow\b.{0,10}\b(buy|order|get|purchase)\b"
    r"|\bnow\s*[.!?]?$)",
    re.IGNORECASE,
)

FAQ_PATTERNS = re.compile(
    r"\b(minimum order|minimum amount|delivery timing|delivery time|delivery hour|"
    r"delivery area|areas? (you )?serve|serviceable|which area|what area|"
    r"same.?day delivery|next.?day delivery|scheduled delivery|"
    r"how (long|soon|fast|quickly).{0,20}(deliver|arrive|reach|get)|"
    r"when (will|do) (you|it|my).{0,20}(deliver|arrive|reach)|"
    r"do you deliver|where do you deliver|"
    r"how much (does|do|is|are).{0,25}(delivery|charge)|"
    r"cost of delivery|what (is|are) the (charge|cost|fee|price).{0,20}delivery|"
    r"is .{0,20}delivery.{0,20}(free|available|charged)|"
    r"delivery charge|payment method|refund policy|cancel policy|"
    r"pincode|serviceable pincode|zip code|what pincode)\b",
    re.IGNORECASE,
)

SERVICE_QUERY_PATTERN = re.compile(
    r"\b(dishwash(?:ing)?|dish\s*wash(?:ing)?|clean(?:ing)?|wash(?:ing)?|sweep(?:ing)?|mop(?:ping)?|"
    r"someone to clean|person to clean|helper|kitchen clean|"
    r"house clean|home clean|room clean)\b",
    re.IGNORECASE,
)

PREFERENCE_STATEMENT_RE = re.compile(
    r"\bi\s+(?:love|like|enjoy|prefer|always buy|hate|dislike|don'?t like|avoid|never buy)\s+([a-z][a-z ]+)",
    re.IGNORECASE,
)

CAT_TO_BROWSE_KEYWORD = {
    "fruits":     "fruits",
    "vegetables": "vegetables",
    "leafy":      "leafy greens",
    "beverages":  "beverages",
    "snacks":     "snacks",
    "fastfood":   "fast food",
    "food":       "food",
    "groceries":  "groceries",
    "service":    "services",
    "rent":       "rent",
}

CROSS_SELL_MAP = {
    "onion":   ["Tomato", "Garlic", "Ginger"],
    "tomato":  ["Onion", "Garlic", "Coriander"],
    "rice":    ["Onion", "Tomato", "Garlic"],
    "pasta":   ["Tomato", "Capsicum", "Onion"],
    "egg":     ["Onion", "Tomato", "Capsicum"],
    "chicken": ["Onion", "Garlic", "Ginger", "Tomato"],
    "bread":   ["Tomato", "Cucumber"],
    "potato":  ["Onion", "Tomato"],
    "spinach": ["Garlic", "Onion"],
}

RECIPE_INGREDIENTS = {
    "biryani":      ["Rice", "Onion", "Tomato", "Ginger", "Garlic", "Carrot", "Capsicum"],
    "dal":          ["Onion", "Tomato", "Ginger", "Garlic"],
    "sabzi":        ["Onion", "Tomato", "Carrot"],
    "salad":        ["Tomato", "Carrot", "Cucumber"],
    "curry":        ["Onion", "Tomato", "Ginger", "Garlic"],
    "soup":         ["Tomato", "Carrot", "Onion", "Garlic"],
    "pulao":        ["Rice", "Onion", "Carrot", "Garlic"],
    "sambar":       ["Tomato", "Onion", "Drumstick", "Garlic", "Ginger"],
    "rasam":        ["Tomato", "Garlic", "Ginger"],
    "poha":         ["Onion", "Potato", "Carrot"],
    "upma":         ["Onion", "Carrot", "Beans"],
    "chapati":      ["Onion", "Potato", "Capsicum"],
    "paratha":      ["Potato", "Onion", "Coriander"],
    "pasta":        ["Tomato", "Onion", "Capsicum", "Carrot"],
    "fried rice":   ["Onion", "Carrot", "Capsicum", "Beans", "Garlic"],
    "avial":        ["Carrot", "Beans", "Cucumber", "Drumstick"],
    "palak paneer": ["Spinach", "Onion", "Tomato", "Garlic", "Ginger"],
    "aloo gobi":    ["Potato", "Cauliflower", "Onion", "Tomato", "Garlic"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Time-of-day helper
# ─────────────────────────────────────────────────────────────────────────────

def get_time_greeting() -> str:
    hour = datetime.now(timezone.utc).hour + 5
    if hour < 12:  return "Good morning"
    if hour < 17:  return "Good afternoon"
    return "Good evening"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sell hint builder
# ─────────────────────────────────────────────────────────────────────────────

def build_cross_sell_hint(cart_item_names: list[str], memory: dict | None = None) -> str | None:
    cart_lower  = {n.lower() for n in cart_item_names}
    suggestions = []
    for cart_name in cart_item_names:
        key = cart_name.lower().strip()
        for pattern, complements in CROSS_SELL_MAP.items():
            if pattern in key:
                for comp in complements:
                    if comp.lower() not in cart_lower and comp not in suggestions:
                        suggestions.append(comp)
    if not suggestions:
        return None
    if memory:
        entities    = memory.get("entities", {})
        blocked     = (entities.get("allergies", []) + entities.get("complaint_items", []))
        blocked_low = {b.lower() for b in blocked}
        suggestions = [s for s in suggestions if s.lower() not in blocked_low]
    if not suggestions:
        return None
    shown = suggestions[:3]
    return "💡 You might also need: " + ", ".join(shown) + ". Want me to add any?"


# ─────────────────────────────────────────────────────────────────────────────
# Simple stemmer for upsell dedup
# ─────────────────────────────────────────────────────────────────────────────

def simple_stem(name: str) -> str:
    """Remove trailing s for basic singular matching."""
    n = name.lower().strip()
    return n[:-1] if n.endswith("s") else n


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_non_grocery_mismatch(product_name: str, query_text: str) -> bool:
    name_l  = product_name.lower()
    query_l = query_text.lower()
    return any(kw in name_l and kw not in query_l for kw in NON_GROCERY_KEYWORDS)


def preprocess_fractions(q: str) -> str:
    # FIX: Handle leading-dot decimals like .5 → 0.5
    q = re.sub(r'(?<!\d)\.(\d+)', r'0.\1', q)
    # Handle fractions like 1/2 → 0.5
    return re.sub(
        r'\b(\d+)/(\d+)\b',
        lambda m: str(round(int(m.group(1)) / int(m.group(2)), 4)),
        q,
    )


def extract_item_name(query: str) -> str | None:
    q = query.lower()
    for phrase in CART_REMOVE_PHRASES:
        q = q.replace(phrase, "")
    q     = re.sub(r"[^\w\s]", "", q)
    words = [w for w in q.split() if w not in STOP_WORDS and len(w) >= 2]
    return " ".join(words).strip().title() if words else None


def normalize_query(q: str) -> str:
    for typo, fix in TYPO_MAP.items():
        q = q.replace(typo, fix)
    return q


def looks_like_price_filter(q: str) -> bool:
    return bool(re.search(r"\b(under|below|less than|at most|maximum|max|within)\b", q))


def looks_like_service_query(q: str) -> bool:
    return bool(re.search(
        r"\b(domestic help|household help|home help|plumbing|electrician|"
        r"painting|pest control|babysitt\w*|nanny|maid|handyman|technician)\b",
        q,
    ))


def looks_like_category_browse(q: str) -> bool:
    if re.search(
        r"\b(book|schedule|tomorrow|sunday|monday|tuesday|wednesday|"
        r"thursday|friday|saturday|next week|morning|evening|afternoon|tonight)\b", q,
    ):
        return False
    if re.search(
        r"\b(are|is|do|does|why|how|what|when|where|which|expiry|expire|"
        r"shelf|life|locally|organic|sourced|grown|healthy|nutritious|"
        r"safe|certified|period|duration|last|long|seller|selling|"
        r"popular|trending|recommend|suggest)\b", q,
    ):
        return False
    return bool(re.search(
        r"\b(fruits?|vegetables?|veg|leafy|greens?|beverages?|drinks?|juice|"
        r"snacks?|groceries|grocery|fast food|fastfood|homemade|home made|"
        r"services?|rent|rental|products?|items?|catalogue|catalog)\b",
        q,
    ))


def looks_like_recommendation_query(q: str) -> bool:
    return bool(re.search(
        r"\b(recommend|suggest|good for|best for|best seller|best selling|"
        r"most popular|top selling|trending|ingredients? for|"
        r"what (do|should) i (need|buy|get|use)|what.*good for|"
        r"healthy|nutritio|diet|rich in|suitable for)\b", q,
    ))


def extract_product_name_from_query(query: str) -> str:
    q = query.lower().strip().rstrip("?").strip()
    q = preprocess_fractions(q)
    q = re.sub(r"[^\w\s.]", " ", q)
    for phrase in PRODUCT_QUERY_PHRASES:
        q = q.replace(phrase, " ")
    q = re.sub(r"(under|below|less than|at most|maximum|max|within)\s*₹?\s*\d+.*$", " ", q)
    q = re.sub(rf"\b\d+(?:\.\d+)?\s*{UNIT_WORDS_RE}\b", " ", q)
    keep_words = {
        "onion", "onions", "tomato", "tomatoes", "potato", "potatoes",
        "ginger", "garlic", "leafy", "spinach", "methi", "coriander", "palak",
        "dishwash", "cleaning", "sweeping", "mopping",
    }
    words  = [
        w for w in q.split()
        if (w not in PRODUCT_FILLER_WORDS and w not in STOP_WORDS) or w in keep_words
    ]
    result = " ".join(words).strip().title()
    return normalise_service_name(result) if result else result


def extract_grocery_items(query: str) -> list:
    q = query.lower().strip()
    q = re.sub(r"\bnow\s*[.!?]?$", "", q).strip()
    q = re.sub(
        r"^(also|and|plus|also add|and add|can you also|"
        r"can i buy|can i get|add|get|buy|order|give me|i want|i need|"
        r"i would like|want to buy|want to order)\s+", "", q,
    )
    q = re.sub(r"\b(from|at|on|in)\s+sahachari\b", "", q)
    q = preprocess_fractions(q)
    q = re.sub(r"[^\w\s.]", " ", q)
    q = q.strip().replace(",", " and ")
    # FIX: ensure digit immediately followed by unit gets a space inserted
    q = re.sub(rf"(\d)({UNIT_WORDS_RE})(?=\s|$)", r"\1 \2", q)

    segments = re.split(r"\band\b|\bplus\b", q)
    items    = []

    # FIX: Updated pattern — multi-char units listed first in UNIT_WORDS_RE
    pattern  = (
        rf"\b{NUM_WORDS_RE}\b"
        rf"\s*(?:{UNIT_WORDS_RE})?"
        r"\s*(?:of\s+)?([a-z][a-z]+(?:\s+[a-z][a-z]+){0,3})"
    )

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        matches = list(re.finditer(pattern, segment))
        if not matches:
            # No quantity specified — treat whole segment as item name
            clean_words = [w for w in segment.split() if w.strip() not in STOP_WORDS]
            if clean_words:
                item_name = " ".join(clean_words).strip()
                if item_name and all(len(w) >= 2 for w in item_name.split()):
                    items.append({"item": item_name.title(), "quantity": None, "unit": None})
            continue
        for m in matches:
            raw_qty, raw_unit, raw_item = m.group(1), m.group(2), m.group(3).strip()
            item_words = [w for w in raw_item.split() if w not in STOP_WORDS and len(w) >= 2]
            if not item_words:
                continue
            try:
                quantity = float(raw_qty) if raw_qty.replace(".", "", 1).isdigit() \
                           else float(WORD_TO_NUM.get(raw_qty, 1.0))
            except (ValueError, TypeError):
                quantity = 1.0
            unit = UNIT_NORMALISE.get(raw_unit, "piece") if raw_unit else "piece"
            items.append({"item": " ".join(item_words).title(), "quantity": quantity, "unit": unit})
    return items


def parse_quantity_reply(query: str):
    q = preprocess_fractions(query.lower().strip())
    m = re.search(rf"{NUM_WORDS_RE}\s*(?:{UNIT_WORDS_RE})?", q)
    if not m:
        return None, None
    raw_qty  = m.group(1)
    raw_unit = m.group(2)
    try:
        quantity = float(raw_qty) if raw_qty.replace(".", "", 1).isdigit() \
                   else float(WORD_TO_NUM.get(raw_qty, 1.0))
    except (ValueError, TypeError):
        quantity = 1.0
    unit = UNIT_NORMALISE.get(raw_unit, "piece") if raw_unit else "piece"
    return quantity, unit


def format_qty(quantity: float, unit: str) -> str:
    qty_str = int(quantity) if quantity == int(quantity) else quantity
    return f"{qty_str} {unit}" if unit != "piece" or quantity != 1 else "1"


# ─────────────────────────────────────────────────────────────────────────────
# Pronoun resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_pronoun(query: str, history: list, session: dict | None = None) -> str:
    q_lower     = query.lower()
    has_pronoun = any(ref in q_lower for ref in PRONOUN_REFS)
    has_action  = bool(re.search(r'\b(add|buy|order|get|remove|delete|update|set|change|make)\b', q_lower))
    if not has_pronoun or not has_action:
        return query

    if session:
        mentioned = session.get("last_mentioned_items", [])
        if mentioned:
            most_recent = mentioned[-1]
            for ref in sorted(PRONOUN_REFS, key=len, reverse=True):
                if ref in q_lower:
                    log.info(f"Pronoun '{ref}' → '{most_recent}' via session entities")
                    return query.lower().replace(ref, most_recent.lower())

    for msg in reversed(history):
        if msg["role"] != "assistant":
            continue
        content = msg["content"]
        cheapest_match = re.search(
            r'cheapest.*?(?:is|option.*?is)\s+\*{0,2}([A-Za-z][A-Za-z\s]+?)\*{0,2}[,\s₹\n]',
            content, re.IGNORECASE,
        )
        if cheapest_match:
            matched_name = cheapest_match.group(1).strip()
            for ref in sorted(PRONOUN_REFS, key=len, reverse=True):
                if ref in q_lower:
                    return query.lower().replace(ref, matched_name.lower())
        product_match = re.search(r'🛍️\s+\*{0,2}\d+\.\s+([A-Z][^\n*]+?)(?:\*{0,2})?\n', content)
        if product_match:
            matched_name = product_match.group(1).strip()
            for ref in sorted(PRONOUN_REFS, key=len, reverse=True):
                if ref in q_lower:
                    return query.lower().replace(ref, matched_name.lower())
    return query


def update_entity_tracker(session: dict, items: list[str]):
    existing = session.get("last_mentioned_items", [])
    for item in items:
        if item and item not in existing:
            existing.append(item)
    session["last_mentioned_items"] = existing[-5:]


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection — scored multi-signal approach
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentScore:
    intent:  str
    score:   float         = 0.0
    signals: list[str]     = field(default_factory=list)


CHECKOUT_EXACT_PHRASES = {
    "checkout", "check out", "place order", "place my order",
    "confirm order", "proceed to payment", "proceed to checkout",
    "complete my order", "complete order",
}


def detect_intent(query: str, session: dict | None = None, memory: dict | None = None) -> tuple[str, float]:
    log.info(f"detect_intent raw input: {repr(query)}")
    q = normalize_query(query.lower().strip())
    log.info(f"detect_intent normalized q: {repr(q)}")

    # ── Fast-path exact matches ──────────────────────────────────────────────
    if q in {"hi", "hello", "hey", "namaste", "good morning", "good evening", "good afternoon"}:
        return "greeting", 1.0
    if q in CLOSING_WORDS:
        return "courtesy", 1.0
    if q in CHECKOUT_EXACT_PHRASES or query.strip().lower() in CHECKOUT_EXACT_PHRASES:
        return "checkout", 1.0
    if PREFERENCE_STATEMENT_RE.search(q):
        return "preference", 1.0
    if REPAIR_SIGNALS.search(q):
        return "repair", 1.0

    scores: dict[str, IntentScore] = {i: IntentScore(i) for i in INTENT_LABELS}

    def boost(intent: str, amount: float, reason: str):
        if intent in scores:
            scores[intent].score += amount
            scores[intent].signals.append(reason)

    # ── Keyword signals ──────────────────────────────────────────────────────
    for intent, keywords in INTENT_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in q)
        if matches:
            boost(intent, matches * 0.4, f"keyword_match:{matches}")

    if any(kw in q for kw in INTENT_KEYWORDS["checkout"]):
        boost("checkout", 1.2, "checkout_keyword_extra_boost")

    # ── Regex signals ────────────────────────────────────────────────────────
    if COMPLAINT_PATTERNS.search(q):               boost("complaint",       1.5, "complaint_regex")
    if REORDER_PATTERNS.search(q):                 boost("reorder",         1.5, "reorder_regex")
    if RECIPE_PATTERNS.search(q):                  boost("recipe",          1.5, "recipe_regex")
    if FAQ_PATTERNS.search(q):                     boost("rag",             1.2, "faq_regex")
    if CHEAPEST_PATTERNS.search(q):                boost("browse",          1.0, "cheapest_regex")
    if SERVICE_QUERY_PATTERN.search(q):            boost("order",           0.8, "service_pattern")
    if looks_like_recommendation_query(q):         boost("rag",             0.9, "recommendation_query")
    if looks_like_service_query(q):                boost("rag",             0.8, "denied_service")
    if looks_like_category_browse(q):              boost("browse",          0.9, "category_browse")
    if looks_like_price_filter(q):                 boost("browse",          0.7, "price_filter")

    if re.search(r"\b(add|put|get|buy|order)\b.{0,20}\b(into|to|in)\b.{0,10}\b(my\s+)?cart\b", q):
        boost("order",     0.8, "add_to_cart_phrase")
        boost("view_cart", -0.8, "suppress_view_cart_on_add")

    if re.search(r"\bcart\b", q) and not re.search(
        r"\b(add|put|remove|delete|clear|empty|drop|buy|order|get)\b", q
    ):
        boost("view_cart", 1.0, "cart_keyword_passive")

    # FIX: Expanded show_all orders detection
    if re.search(
        r"\b(status of|show|list).{0,10}\b(all|every|my)?\s*orders?\b", q
    ) or re.search(
        r"\b(show my orders|my orders|show orders|order history|past orders|previous orders)\b", q
    ):
        boost("status", 1.8, "all_orders_regex")

    if re.search(r"\b(track|where is|status of|check).{0,20}\border\b", q) or \
       re.search(r"\border\b.{0,20}\b(status|tracking|whereabouts)\b", q):
        boost("status", 1.5, "order_status_regex")

    if re.search(
        r"\b(how (can|do|to) i (track|cancel|order|buy|get|check|view)|"
        r"can i track|can i cancel|can i check|can i view my order|"
        r"steps to (cancel|track|order)|process to cancel|"
        r"want to know how|how does .* work)\b", q,
    ):
        boost("rag", 1.1, "how_to_query")

    if re.search(r"\b(increase|update|change|set|make it)\b.+\b(quantity|qty|amount|to)\b", q):
        boost("update_cart", 1.5, "update_qty_regex")

    if re.search(
        r"\b(remove|clear|empty|delete)\b.{0,20}\b(everything|all items?|all from|my cart|the cart)\b", q,
    ):
        boost("clear_cart", 1.5, "clear_all_regex")

    if re.search(
        r"\b(show|see|view|get|display)\b.{0,15}\b(my\s+)?(last|latest|recent|current|previous)?\s*order\b", q,
    ):
        boost("status", 1.0, "show_order_regex")

    if re.search(r"\b(remove|delete|take out|drop)\b.*\bcart\b", q):
        boost("delete_cart", 1.2, "remove_from_cart")

    # ── Structural signals ───────────────────────────────────────────────────
    if re.match(r"^(add|buy|order|get|i want|i need)\b", q):
        boost("order", 0.6, "action_verb_start")
    if re.match(r"^(show|list|display)\b", q) and "order" not in q and "cart" not in q:
        boost("browse", 0.4, "browse_verb_start")
    if re.match(r"^(remove|delete)\s+\w+", q) and "wishlist" not in q:
        boost("delete_cart", 0.6, "remove_verb_start")
    if len(q.split()) <= 2 and not any(kw in q for kw in _SHORT_QUERY_EXEMPT):
        boost("clarify", 0.8, "very_short_query")
    if re.match(r"^(what|who|why|how)\b", q) and "cart" not in q \
            and "wishlist" not in q and "order" not in q:
        boost("rag", 0.5, "question_word_start")
    if re.search(r"\b(available|in stock|price|cost|how much|do you have|is there)\b", q):
        boost("product_info", 0.5, "availability_query")

    # ── Session context signals ───────────────────────────────────────────────
    if session:
        if session.get("pending_item"):
            boost("order", 0.5, "has_pending_item")
        if session.get("last_browse_category"):
            boost("browse", 0.3, "continuing_browse")
        analytics = session.get("analytics", {})
        prev_intents = analytics.get("intents_this_session", [])
        if "product_info" in prev_intents[-3:]:
            boost("order", 0.2, "after_product_info")

    # ── Memory signals ────────────────────────────────────────────────────────
    if memory:
        top_items = list(memory.get("frequent_items", {}).keys())
        if any(item.lower() in q for item in top_items):
            boost("order", 0.4, "matches_frequent_item")

    # ── Quantity pattern — strong order signal ─────────────────────────────
    if re.search(rf"\b{NUM_WORDS_RE}\b.*{UNIT_WORDS_RE}", q):
        boost("order", 0.7, "quantity_unit_pattern")

    winner = max(scores.values(), key=lambda s: s.score)
    non_zero = {k: round(v.score, 2) for k, v in scores.items() if v.score > 0}
    log.info(f"Intent scores: {non_zero}")

    if winner.score == 0:
        return "rag", 0.3

    scores_list = sorted([s.score for s in scores.values()], reverse=True)
    sole_winner = len(scores_list) < 2 or scores_list[1] == 0
    raw_confidence = min(winner.score / 2.0, 1.0)
    if sole_winner:
        raw_confidence = max(raw_confidence, 0.5)
    if winner.intent == "checkout":
        raw_confidence = max(raw_confidence, 0.5)
    return winner.intent, raw_confidence


# ─────────────────────────────────────────────────────────────────────────────
# Session analytics helpers
# ─────────────────────────────────────────────────────────────────────────────

def record_intent(session: dict, intent: str):
    analytics = session.setdefault("analytics", {**DEFAULT_SESSION["analytics"]})
    intents   = analytics.setdefault("intents_this_session", [])
    intents.append(intent)
    if len(intents) > 50:
        analytics["intents_this_session"] = intents[-50:]


def detect_cart_abandonment_risk(session: dict) -> bool:
    return session.get("analytics", {}).get("cart_abandonment", 0) >= 2


def record_product_viewed(session: dict, item_name: str):
    analytics = session.setdefault("analytics", {})
    viewed    = analytics.setdefault("items_viewed", [])
    if item_name not in viewed:
        viewed.append(item_name)
    analytics["items_viewed"] = viewed[-10:]


# ─────────────────────────────────────────────────────────────────────────────
# Response personalisation layer
# ─────────────────────────────────────────────────────────────────────────────

def personalise_response(
    response: str,
    memory: dict,
    session: dict,
    intent: str,
    items_just_added: list[str] | None = None,
) -> str:
    name = memory.get("name")

    if name and not session.get("_greeted_with_name") and intent == "greeting":
        session["_greeted_with_name"] = True
        response = response.replace("Welcome to Sahachari", f"Welcome to Sahachari, {name}", 1)
        response = response.replace("Welcome back to Sahachari", f"Welcome back, {name}!", 1)

    freq = memory.get("frequent_items", {})
    if freq and items_just_added and intent == "order":
        top = max(freq, key=freq.get)
        # FIX: Use stem comparison to avoid suggesting item just added (e.g. "Apples" vs "Apple")
        added_stems = {simple_stem(i) for i in items_just_added}
        if simple_stem(top) not in added_stems and "added!" in response:
            safe = memory_store.get_safe_suggestions(memory, [top]) if memory_store else [top]
            if safe:
                response += f"\n\n💡 You usually also order **{safe[0]}** — want me to add it?"

    if intent == "clear_cart":
        if detect_cart_abandonment_risk(session):
            response += "\n\n💬 Having trouble finding what you need? Just tell me what you're looking for!"

    entities  = memory.get("entities", {})
    allergies = entities.get("allergies", [])
    if allergies and items_just_added:
        allergen_triggered = [
            a for a in allergies
            if any(a.lower() in item.lower() for item in items_just_added)
        ]
        if allergen_triggered:
            response = (
                f"⚠️ **Allergy alert**: You've mentioned being allergic to "
                f"**{', '.join(allergen_triggered)}**. "
                f"Please double-check before consuming.\n\n" + response
            )

    return response


# ─────────────────────────────────────────────────────────────────────────────
# Wishlist helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wishlist_line(item: dict) -> str:
    if item.get("quantity") is not None:
        return f"{item['item']} ({format_qty(item['quantity'], item.get('unit') or 'piece')})"
    return item["item"]


def add_to_wishlist(items: list, curr_session: dict) -> str:
    for new in items:
        if not any(w["item"].lower() == new["item"].lower() for w in curr_session["wishlist"]):
            curr_session["wishlist"].append(new.copy())
    count = len(curr_session["wishlist"])
    lines = [f"💛 Wishlist ({count} items)", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
    for idx, item in enumerate(curr_session["wishlist"], 1):
        lines.append(f"  {idx}. {_wishlist_line(item)}")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


def view_wishlist(curr_session: dict) -> str:
    if not curr_session["wishlist"]:
        return "Your wishlist is empty."
    lines = ["💛 Your Wishlist:"]
    for i, item in enumerate(curr_session["wishlist"], 1):
        lines.append(f"  {i}. {_wishlist_line(item)}")
    return "\n".join(lines)


def remove_from_wishlist(query: str, curr_session: dict) -> str:
    q = query.lower()
    if any(p in q for p in ["clear wishlist", "empty wishlist"]):
        curr_session["wishlist"].clear()
        return "💛 Your wishlist has been cleared."
    item_name = extract_item_name(query)
    if not item_name:
        return "Which item should I remove from your wishlist?"
    before = len(curr_session["wishlist"])
    curr_session["wishlist"] = [
        w for w in curr_session["wishlist"]
        if item_name.lower() not in w["item"].lower()
        and w["item"].lower() not in item_name.lower()
    ]
    if len(curr_session["wishlist"]) == before:
        return f"❌ '{item_name}' was not found in your wishlist."
    return f"🗑️ Removed '{item_name}' from your wishlist.\n\n{view_wishlist(curr_session)}"


# ─────────────────────────────────────────────────────────────────────────────
# Model / DB globals
# ─────────────────────────────────────────────────────────────────────────────

embedding_model: Optional[SentenceTransformer] = None
reranker        = None
chroma_client   = None
collection      = None
qwen_pipeline   = None
mongo_client    = None
db              = None
session_store:  Optional[SessionStore] = None
memory_store:   Optional[MemoryStore]  = None
http_client:    Optional[httpx.AsyncClient] = None

service_status = {
    "embedding_model":   False,
    "reranker":          False,
    "chroma_collection": False,
    "qwen_pipeline":     False,
    "mongo":             False,
}


def load_embedding_model():
    global embedding_model
    try:
        log.info("Loading embedding model BAAI/bge-base-en-v1.5 ...")
        embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        service_status["embedding_model"] = True
        log.info("Embedding model loaded.")
    except Exception as e:
        log.error(f"Failed to load embedding model: {e}")


def load_reranker():
    global reranker
    if FlagReranker is None:
        log.error("FlagEmbedding not importable; reranker disabled.")
        return
    try:
        log.info("Loading reranker BAAI/bge-reranker-base ...")
        reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=False)
        service_status["reranker"] = True
        log.info("Reranker loaded.")
    except Exception as e:
        log.error(f"Failed to load reranker: {e}")


def load_chroma_collection():
    global chroma_client, collection
    try:
        log.info(f"Connecting to ChromaDB at {CHROMA_PATH} ...")
        chroma_client = PersistentClient(path=CHROMA_PATH)
        collection    = chroma_client.get_collection("sahachari_docs")
        service_status["chroma_collection"] = True
        log.info("ChromaDB collection loaded.")
    except Exception as e:
        log.error(f"Failed to load ChromaDB: {e}. Run ingest.py to rebuild. RAG disabled.")


def load_qwen_pipeline():
    global qwen_pipeline
    try:
        log.info(f"Loading local LLM {QWEN_MODEL_NAME} ...")
        qwen_pipeline = pipeline(
            "text-generation",
            model=QWEN_MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        service_status["qwen_pipeline"] = True
        log.info("Local LLM loaded.")
    except Exception as e:
        log.error(f"Failed to load Qwen ({QWEN_MODEL_NAME}): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mongo_client, db, session_store, memory_store, http_client

    http_client  = httpx.AsyncClient(timeout=15.0)
    mongo_client = AsyncIOMotorClient(
        MONGO_URI,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=10000,
    )
    db = mongo_client[MONGO_DB]
    try:
        await mongo_client.admin.command("ping")
        service_status["mongo"] = True
        log.info(f"MongoDB connected. DB: {MONGO_DB}")
    except Exception as e:
        log.error(f"MongoDB FAILED: {type(e).__name__}: {e}")

    session_store = SessionStore(db)
    memory_store  = MemoryStore(db)
    if service_status["mongo"]:
        await session_store.ensure_indexes()
        await memory_store.ensure_indexes()

    loop = asyncio.get_running_loop()
    await asyncio.gather(
        loop.run_in_executor(None, load_embedding_model),
        loop.run_in_executor(None, load_reranker),
        loop.run_in_executor(None, load_chroma_collection),
        loop.run_in_executor(None, load_qwen_pipeline),
    )
    log.info(f"Startup complete. Status: {service_status}")
    yield

    await http_client.aclose()
    if mongo_client is not None:
        mongo_client.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "services": service_status}


@app.get("/ready")
async def ready():
    is_ready = service_status["embedding_model"] and service_status["mongo"]
    if is_ready:
        return JSONResponse({"status": "ready",   "services": service_status}, status_code=200)
    return JSONResponse({"status": "loading", "services": service_status}, status_code=503)


# ─────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ─────────────────────────────────────────────────────────────────────────────

async def find_product_id(item_name: str, auth_token: str):
    item_name = normalise_service_name(item_name)
    cache_key = item_name.strip().lower()
    if cache_key in _product_cache:
        log.info(f"Product cache hit: {cache_key!r}")
        return _product_cache[cache_key]

    clean_name = cache_key
    if not clean_name:
        return None, None
    query_words = set(w for w in re.findall(r"[a-z]+", clean_name) if len(w) >= 3)

    def is_relevant(candidate_name: str) -> bool:
        if not candidate_name:
            return False
        cand = candidate_name.lower()
        if clean_name in cand or cand in clean_name:
            return True
        cand_words = set(w for w in re.findall(r"[a-z]+", cand) if len(w) >= 3)
        if not query_words or not cand_words:
            return False
        return len(query_words & cand_words) > 0

    headers = {"Authorization": auth_token}

    # FIX: More comprehensive plural/singular alternates
    alts = []
    if clean_name.endswith("oes"):
        alts.append(clean_name[:-2])          # tomatoes → tomato
    elif clean_name.endswith("ies"):
        alts.append(clean_name[:-3] + "y")    # berries → berry
    elif clean_name.endswith("s") and not clean_name.endswith("ss"):
        alts.append(clean_name[:-1])           # onions → onion
    else:
        alts.append(clean_name + "s")          # onion → onions

    # Also add stem without last char if ends in s (catches "carrots" → "carrot")
    if clean_name.endswith("s") and len(clean_name) > 3:
        stem = clean_name[:-1]
        if stem not in alts:
            alts.append(stem)

    search_terms = [t for t in [item_name] + [a.title() for a in alts] if t]

    for term in search_terms:
        try:
            response = await http_client.get(
                f"{NESTJS_BACKEND_URL}/customer/products",
                params={"search": term},
                headers=headers,
            )
            if response.status_code == 200:
                data     = response.json()
                products = data if isinstance(data, list) else (
                    data.get("data") or data.get("products") or data.get("items") or []
                )
                for prod in products:
                    name = prod.get("name", "")
                    if is_non_grocery_mismatch(name, item_name):
                        continue
                    if is_relevant(name):
                        result = (prod.get("_id") or prod.get("id"), name or clean_name)
                        _product_cache[cache_key] = result
                        return result
        except Exception as e:
            log.warning(f"find_product_id failed for '{term}': {e}")

    _product_cache[cache_key] = (None, None)
    return None, None


async def fetch_store_name_cache(store_id: str, auth_token: str, cache: dict) -> str:
    return "Sahachari"


def extract_product_name(item: dict) -> str:
    for key in ("product", "productId"):
        nested = item.get(key)
        if isinstance(nested, dict):
            for field in ("name", "title", "itemName"):
                if nested.get(field):
                    return nested[field]
    for field in ("name", "productName", "title"):
        if item.get(field):
            return item[field]
    return "Unknown Item"


def _parse_price_field(val) -> float:
    try:
        cleaned = re.sub(r"[₹,\s]", "", re.sub(r"/.*$", "", str(val))).strip()
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0


def _build_cart_summary_from_data(cart_data: dict, memory: dict | None = None) -> str:
    items = cart_data.get("items", [])
    if not items:
        return "🛒 Your cart is empty.\n\nAdd some fresh items! 🛍️"

    grand_total = 0.0
    item_lines  = []

    for idx, item in enumerate(items, 1):
        name         = extract_product_name(item)
        qty          = item.get("quantity", 1)
        product_data = item.get("productId") if isinstance(item.get("productId"), dict) else item
        price        = 0.0
        for f in ("finalPrice", "price", "sellingPrice", "mrp"):
            val = product_data.get(f)
            if val is not None:
                price = _parse_price_field(val)
                break

        if price > 0:
            for offer in product_data.get("offers", []):
                if not offer.get("isActive"):
                    continue
                try:
                    end_date = datetime.fromisoformat(offer["endDate"].replace("Z", "+00:00"))
                    if end_date >= datetime.now(timezone.utc):
                        if offer.get("type") == "PERCENTAGE":
                            price = price * (1 - offer["value"] / 100)
                        elif offer.get("type") == "FLAT":
                            price = max(0.0, price - offer["value"])
                        break
                except Exception:
                    continue

        item_total   = price * qty
        grand_total += item_total
        item_lines.append(
            f"🛍️ **{idx}. {name}**\n"
            f"   📦 Quantity: **{qty}**\n"
            f"   💰 Amount: **₹{item_total:,.2f}**\n"
        )

    lines = [
        "🛒 **Your Cart**",
        f"📋 {len(items)} item{'s' if len(items) > 1 else ''}",
        "═" * 30,
        "",
    ]
    lines.extend(item_lines)
    lines.append("")
    lines.append(f"💰 **Total: ₹{grand_total:,.2f}**")
    lines.append("\n✨ Ready to checkout?")

    cart_names = [extract_product_name(i) for i in items]
    cross_hint = build_cross_sell_hint(cart_names, memory)
    if cross_hint:
        lines.append("\n" + cross_hint)

    return "\n".join(lines)


async def fetch_nestjs_cart_summary(auth_token: str, memory: dict | None = None) -> str:
    if not auth_token:
        return "⚠️ Please log in to view your cart."
    token = clean_auth_token(auth_token)
    try:
        response = await http_client.get(
            f"{NESTJS_BACKEND_URL}/customer/cart",
            headers={"Authorization": token},
        )
        if response.status_code != 200:
            return "🛒 Unable to fetch your cart right now."
        return _build_cart_summary_from_data(response.json(), memory)
    except Exception as e:
        log.warning(f"fetch_nestjs_cart_summary failed: {e}")
        return "🛒 Your cart is ready!"


async def _add_single_item_to_cart(
    item: dict,
    product_id,
    matched_name: str,
    existing_cart: dict,
    headers: dict,
    token: str,
) -> tuple[str, bool]:
    if not product_id:
        return f"❌ '{item['item']}' not found in Sahachari store.", False

    raw_qty = float(item["quantity"]) if item["quantity"] is not None else 1.0
    unit    = item.get("unit") or "piece"

    if unit == "g":
        raw_qty /= 1000;  unit = "kg"
    elif unit == "ml":
        raw_qty /= 1000;  unit = "litre"

    cart_quantity = round(raw_qty, 3) if raw_qty != int(raw_qty) else int(raw_qty)
    if cart_quantity <= 0:
        cart_quantity = 1

    try:
        existing_entry = existing_cart.get(str(product_id))
        if existing_entry:
            existing_item_id, current_qty = existing_entry
            new_qty  = current_qty + cart_quantity
            response = await http_client.patch(
                f"{NESTJS_BACKEND_URL}/customer/cart/{existing_item_id}",
                json={"quantity": new_qty},
                headers=headers,
            )
            if response.status_code in OK_STATUSES:
                qty_label = format_qty(raw_qty, unit)
                total_label = format_qty(new_qty, unit)
                return f"✅ {matched_name} — {qty_label} added! (Total in cart: {total_label})", True
            else:
                log.warning(f"Cart PATCH failed — body: {response.text}")
                return f"❌ Failed to update {matched_name} ({response.status_code}).", False
        else:
            response = await http_client.post(
                f"{NESTJS_BACKEND_URL}/customer/cart",
                json={"productId": product_id, "quantity": cart_quantity},
                headers=headers,
            )
            if response.status_code in OK_STATUSES:
                qty_label = format_qty(raw_qty, unit)
                return f"✅ {matched_name} — {qty_label} added!", True
            else:
                log.warning(f"Cart POST failed — body: {response.text}")
                return f"❌ Failed to add {matched_name} ({response.status_code}).", False

    except httpx.TimeoutException:
        return f"❌ Timed out adding {matched_name}. Please try again.", False
    except httpx.ConnectError:
        return f"❌ Could not reach server adding {matched_name}. Please try again.", False
    except Exception as e:
        return f"❌ Error on {item['item']} ({type(e).__name__}). Please try again.", False


async def forward_to_nestjs_cart(
    items: list,
    auth_token: str,
    memory: dict | None = None,
) -> str:
    if not auth_token:
        return "⚠️ Error: Authorization token missing. Please log into Sahachari."

    token   = clean_auth_token(auth_token)
    headers = {"Authorization": token, "Content-Type": "application/json"}

    cart_fetch      = http_client.get(f"{NESTJS_BACKEND_URL}/customer/cart", headers={"Authorization": token})
    product_lookups = [find_product_id(item["item"], token) for item in items]

    results         = await asyncio.gather(cart_fetch, *product_lookups, return_exceptions=True)
    cart_res        = results[0]
    product_results = results[1:]

    existing_cart: dict[str, tuple] = {}
    if not isinstance(cart_res, Exception) and cart_res.status_code == 200:
        for ci in cart_res.json().get("items", []):
            pid     = ci.get("productId")
            pid_str = str(pid.get("_id") if isinstance(pid, dict) else pid)
            existing_cart[pid_str] = (ci.get("_id") or ci.get("id"), ci.get("quantity", 0))
    else:
        log.warning("Could not pre-fetch cart for duplicate check.")

    add_tasks = []
    for item, prod_result in zip(items, product_results):
        product_id, matched_name = (
            (None, item["item"]) if isinstance(prod_result, Exception) else prod_result
        )
        add_tasks.append(
            _add_single_item_to_cart(
                item, product_id, matched_name or item["item"], existing_cart, headers, token
            )
        )

    add_results = await asyncio.gather(*add_tasks)
    added_lines = []
    any_added   = False
    for msg, success in add_results:
        added_lines.append(msg)
        if success:
            any_added = True

    if not added_lines:
        return "No items could be processed."

    result = "\n".join(added_lines)
    if any_added:
        try:
            updated_cart_res = await http_client.get(
                f"{NESTJS_BACKEND_URL}/customer/cart", headers={"Authorization": token},
            )
            cart_summary = (
                _build_cart_summary_from_data(updated_cart_res.json(), memory)
                if updated_cart_res.status_code == 200
                else await fetch_nestjs_cart_summary(token, memory)
            )
        except Exception:
            cart_summary = await fetch_nestjs_cart_summary(token, memory)

        result += f"\n\n{cart_summary}\n\n💡 Say 'checkout' to place your order or keep adding items!"
    else:
        result += "\n\nThis item isn't available in Sahachari's catalogue yet."
    return result


async def update_nestjs_cart_quantity(query: str, auth_token: str) -> str:
    token = clean_auth_token(auth_token)
    m = (
        re.search(rf"(?:of|for)\s+([a-z ]+?)\s+to\s+{NUM_WORDS_RE}\s*(?:{UNIT_WORDS_RE})?", query.lower())
        or re.search(rf"(?:update|set|change|make(?:\s+it)?)\s+([a-z ]+?)\s+to\s+{NUM_WORDS_RE}\s*(?:{UNIT_WORDS_RE})?", query.lower())
        or re.search(rf"([a-z][a-z\s]+?)\s+to\s+{NUM_WORDS_RE}\s*(?:{UNIT_WORDS_RE})?", query.lower())
    )
    if not m:
        return "❌ Please specify the item and quantity, e.g. 'update apple to 5 kg'."

    filler   = {"update", "set", "change", "make", "it", "the", "my", "quantity", "qty"}
    raw_item = " ".join(w for w in m.group(1).strip().lower().split() if w not in filler).title()
    if not raw_item:
        return "❌ Please specify the item and quantity, e.g. 'update apple to 5 kg'."

    raw_qty  = m.group(2)
    raw_unit = m.group(3) if m.lastindex >= 3 else None
    try:
        quantity = float(raw_qty) if raw_qty.replace(".", "", 1).isdigit() \
                   else float(WORD_TO_NUM.get(raw_qty, 1.0))
    except (ValueError, TypeError):
        quantity = 1.0
    unit          = UNIT_NORMALISE.get(raw_unit, "piece") if raw_unit else "piece"
    cart_quantity = int(quantity) if quantity == int(quantity) else quantity

    headers = {"Authorization": token}
    try:
        cart_res = await http_client.get(f"{NESTJS_BACKEND_URL}/customer/cart", headers=headers)
        if cart_res.status_code != 200:
            return "❌ Could not fetch your cart."
        items  = cart_res.json().get("items", [])
        target = next(
            (i for i in items if raw_item.lower() in extract_product_name(i).lower()
             or extract_product_name(i).lower() in raw_item.lower()), None,
        )
        if not target:
            return f"❌ '{raw_item}' not found in your cart."
        item_id = target.get("_id") or target.get("id")
        res = await http_client.patch(
            f"{NESTJS_BACKEND_URL}/customer/cart/{item_id}",
            json={"quantity": cart_quantity},
            headers={**headers, "Content-Type": "application/json"},
        )
        if res.status_code in OK_STATUSES:
            updated_res = await http_client.get(f"{NESTJS_BACKEND_URL}/customer/cart", headers=headers)
            summary = (
                _build_cart_summary_from_data(updated_res.json())
                if updated_res.status_code == 200
                else await fetch_nestjs_cart_summary(token)
            )
            return f"✅ Updated {raw_item} to {format_qty(quantity, unit)}!\n\n{summary}"
        return f"❌ Failed to update {raw_item} ({res.status_code})."
    except Exception as e:
        return f"❌ Error updating cart: {str(e)}"


async def delete_from_nestjs_cart(item_name: str, auth_token: str) -> str:
    if not auth_token:
        return "⚠️ Error: Authorization token missing."
    token   = clean_auth_token(auth_token)
    headers = {"Authorization": token}
    try:
        response = await http_client.get(f"{NESTJS_BACKEND_URL}/customer/cart", headers=headers)
        if response.status_code != 200:
            return "❌ Could not fetch your cart."
        items = response.json().get("items", [])
        if not items:
            return "🛒 Your cart is already empty."

        target_item  = None
        matched_name = item_name
        for item in items:
            name = extract_product_name(item)
            if name.lower() == item_name.lower():
                target_item  = item;  matched_name = name;  break
        if not target_item:
            for item in items:
                name = extract_product_name(item)
                if name != "Unknown Item" and (
                    item_name.lower() in name.lower() or name.lower() in item_name.lower()
                ):
                    target_item  = item;  matched_name = name;  break
        if not target_item:
            return f"❌ '{item_name}' not found in your cart."

        item_id      = target_item.get("_id") or target_item.get("id")
        del_response = await http_client.delete(
            f"{NESTJS_BACKEND_URL}/customer/cart/{item_id}", headers=headers,
        )
        if del_response.status_code in OK_STATUSES:
            updated_res = await http_client.get(f"{NESTJS_BACKEND_URL}/customer/cart", headers=headers)
            summary = (
                _build_cart_summary_from_data(updated_res.json())
                if updated_res.status_code == 200
                else await fetch_nestjs_cart_summary(token)
            )
            return f"🗑️ {matched_name} removed from cart!\n\n{summary}"
        return f"❌ Failed to remove {matched_name} ({del_response.status_code})"
    except Exception as e:
        return f"❌ Error removing item: {str(e)}"


async def clear_nestjs_cart(auth_token: str) -> str:
    if not auth_token:
        return "⚠️ Error: Authorization token missing."
    token   = clean_auth_token(auth_token)
    headers = {"Authorization": token}
    try:
        bulk_res = await http_client.delete(f"{NESTJS_BACKEND_URL}/customer/cart", headers=headers)
        if bulk_res.status_code in OK_STATUSES:
            return "🗑️ Your cart has been cleared!"
    except Exception:
        pass
    try:
        cart_res = await http_client.get(f"{NESTJS_BACKEND_URL}/customer/cart", headers=headers)
        if cart_res.status_code != 200:
            return "❌ Could not fetch your cart to clear it."
        items = cart_res.json().get("items", [])
        if not items:
            return "🛒 Your cart is already empty."
        errors = []
        for item in items:
            item_id = item.get("_id") or item.get("id")
            try:
                await http_client.delete(
                    f"{NESTJS_BACKEND_URL}/customer/cart/{item_id}", headers=headers,
                )
            except Exception as e:
                errors.append(str(e))
        if errors:
            return f"⚠️ Cart partially cleared. Errors: {'; '.join(errors[:3])}"
        return "🗑️ Your cart has been cleared!"
    except Exception as e:
        return f"❌ Error clearing cart: {str(e)}"


async def fetch_nestjs_order_status(
    auth_token: str,
    order_id: str | None = None,
    item_name: str | None = None,
) -> str:
    token = clean_auth_token(auth_token)
    icons = {
        "PLACED": "⏳", "ACCEPTED": "✅", "READY": "📦",
        "PICKED_UP": "🛵", "DELIVERED": "✅", "CANCELLED": "❌", "FAILED": "⚠️",
    }
    try:
        response = await http_client.get(
            f"{NESTJS_BACKEND_URL}/customer/orders",
            headers={"Authorization": token},
        )
        if response.status_code != 200:
            return "❌ Could not fetch your orders right now."
        orders = response.json()
        if not orders:
            return "No orders found yet."

        target = None
        if order_id:
            for o in orders:
                oid = str(o.get("_id") or o.get("id") or "")
                if order_id.lower() in oid.lower():
                    target = o;  break
            if not target:
                return f"❌ No order found matching ID '{order_id}'."
        elif item_name:
            query_words = set(w for w in re.findall(r"[a-z]+", item_name.lower()) if len(w) >= 3)
            for o in orders:
                for it in o.get("items", []):
                    name       = extract_product_name(it).lower()
                    cand_words = set(w for w in re.findall(r"[a-z]+", name) if len(w) >= 3)
                    if item_name.lower() in name or name in item_name.lower() \
                            or (query_words & cand_words):
                        target = o;  break
                if target:
                    break
            if not target:
                return f"❌ No order found containing '{item_name}'."
        else:
            target = orders[0]

        status        = target.get("status", "Processing")
        order_id_disp = target.get("_id") or target.get("id")
        icon          = icons.get(status, "•")
        date_str      = target.get("createdAt", "")
        if date_str:
            try:
                dt       = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                date_str = dt.strftime("%d %b %Y, %I:%M %p")
            except Exception:
                pass

        items       = target.get("items", [])
        item_lines  = []
        grand_total = 0.0
        for idx, it in enumerate(items, 1):
            name         = extract_product_name(it)
            qty          = it.get("quantity", 1)
            product_data = it.get("productId") if isinstance(it.get("productId"), dict) else it
            price        = 0.0
            for f in ("finalPrice", "price", "sellingPrice", "mrp"):
                val = product_data.get(f)
                if val is not None:
                    price = _parse_price_field(val)
                    if price:
                        break
            item_total   = price * qty
            grand_total += item_total
            price_part   = f"₹{item_total:.2f}" if price > 0 else "N/A"
            item_lines.append(f"  {idx}. {name:<18} x{qty:<3} {price_part}")

        address    = target.get("deliveryAddress") or target.get("address") or {}
        addr_parts = []
        if isinstance(address, dict):
            for field in ("street", "area", "city", "pincode", "landmark"):
                val = address.get(field)
                if val:
                    addr_parts.append(str(val))
        addr_line      = ", ".join(addr_parts) if addr_parts else None
        payment_method = target.get("paymentMethod") or target.get("payment") or None
        payment_status = target.get("paymentStatus") or None

        lines = [f"📦 Order #{str(order_id_disp)[-6:]}  ({order_id_disp})", "━" * 28]
        if date_str:       lines.append(f"📅 Placed    : {date_str}")
        lines.append(f"{icon} Status    : {status}")
        if payment_method:
            lines.append(f"💳 Payment   : {payment_method}" + (f" ({payment_status})" if payment_status else ""))
        if addr_line:      lines.append(f"📍 Address   : {addr_line}")
        lines.append("━" * 28)
        lines.append("🛒 Items:")
        lines.extend(item_lines)
        lines.append("━" * 28)
        if grand_total > 0:
            lines.append(f"💰 Total     : ₹{grand_total:.2f}")
        return "\n".join(lines)
    except Exception as e:
        log.warning(f"fetch_nestjs_order_status failed: {e}")
    return "No orders found yet."


async def fetch_all_orders_status(auth_token: str) -> str:
    token = clean_auth_token(auth_token)
    icons = {
        "PLACED": "⏳", "ACCEPTED": "✅", "READY": "📦",
        "PICKED_UP": "🛵", "DELIVERED": "✅", "CANCELLED": "❌", "FAILED": "⚠️",
    }
    try:
        response = await http_client.get(
            f"{NESTJS_BACKEND_URL}/customer/orders",
            headers={"Authorization": token},
        )
        if response.status_code != 200:
            return "❌ Could not fetch your orders right now."
        orders = response.json()
        if not orders:
            return "You have no orders yet."

        lines = [f"📋 All Your Orders ({len(orders)} total)", "━" * 28]
        for idx, order in enumerate(orders, 1):
            order_id  = order.get("_id") or order.get("id")
            status    = order.get("status", "Unknown")
            icon      = icons.get(status, "•")
            date      = order.get("createdAt", "")
            if date:
                try:
                    dt   = datetime.fromisoformat(date.replace("Z", "+00:00"))
                    date = dt.strftime("%d %b %Y, %I:%M %p")
                except Exception:
                    pass
            items     = order.get("items", [])
            item_list = ", ".join(extract_product_name(i) for i in items[:3])
            if len(items) > 3:
                item_list += f" + {len(items) - 3} more"
            lines.append(f"  {idx}. Order #{str(order_id)[-6:]}")
            if date:       lines.append(f"     📅 {date}")
            lines.append(f"     {icon} {status}")
            if item_list:  lines.append(f"     🛒 {item_list}")
            lines.append("")
        lines.append("━" * 28)
        return "\n".join(lines)
    except Exception as e:
        log.warning(f"fetch_all_orders_status failed: {e}")
        return "❌ Could not fetch your orders right now."


async def forward_nestjs_order_placement(auth_token: str):
    if not auth_token:
        return "⚠️ Error: Authorization token missing.", False
    token = clean_auth_token(auth_token)
    try:
        cart_res = await http_client.get(
            f"{NESTJS_BACKEND_URL}/customer/cart", headers={"Authorization": token},
        )
        if cart_res.status_code == 200:
            if not cart_res.json().get("items"):
                return "🛒 Your cart is empty. Add some items first!", False
        else:
            return "❌ Could not verify your cart right now. Please try again.", False
    except Exception:
        return "❌ Could not reach Sahachari server. Please try again shortly.", False
    return "🛒 Taking you to checkout now!", True


async def forward_nestjs_order_cancellation(auth_token: str) -> str:
    token = clean_auth_token(auth_token)
    try:
        orders_res = await http_client.get(
            f"{NESTJS_BACKEND_URL}/customer/orders", headers={"Authorization": token},
        )
        if orders_res.status_code != 200:
            return "❌ Could not fetch your orders. Please try from the Orders screen."
        orders = orders_res.json()
        if not orders:
            return "🛒 You have no orders to cancel."

        last_order    = orders[0]
        last_order_id = last_order.get("_id") or last_order.get("id")
        status        = last_order.get("status", "")
        if status in ("DELIVERED", "CANCELLED", "FAILED"):
            return (
                f"❌ Your last order is already {status} and cannot be cancelled.\n"
                "Please check the Orders screen for more details."
            )

        cancelled = False
        for url in [
            f"{NESTJS_BACKEND_URL}/customer/orders/{last_order_id}/cancel",
            f"{NESTJS_BACKEND_URL}/customer/orders/{last_order_id}",
        ]:
            try:
                for method in ("POST", "PATCH"):
                    res = await http_client.request(
                        method, url,
                        headers={"Authorization": token},
                        json={"status": "CANCELLED"} if method == "PATCH" else {},
                    )
                    if res.status_code in OK_STATUSES:
                        cancelled = True;  break
                if cancelled:
                    break
            except Exception:
                continue

        if cancelled:
            return f"✅ Your order ({last_order_id}) has been cancelled successfully."
        return (
            "❌ Could not cancel your order automatically.\n"
            "Please cancel from the Orders screen or contact support."
        )
    except Exception as e:
        log.warning(f"forward_nestjs_order_cancellation failed: {e}")
        return "❌ Could not reach Sahachari server. Please try from the Orders screen."


def extract_order_query_target(query: str) -> tuple[str | None, str | None]:
    q = query.lower().strip()
    id_match = ORDER_ID_RE.search(q)
    if id_match:
        return id_match.group(1), None
    stripped = q
    for phrase in INTENT_KEYWORDS["status"]:
        stripped = stripped.replace(phrase, " ")
    stripped = re.sub(r"\b(status|of|my|the|order|for|with|containing|like)\b", " ", stripped)
    stripped = re.sub(r"[^a-z0-9\s]", " ", stripped)
    words = [w for w in stripped.split() if w not in STOP_WORDS and len(w) >= 2]
    words = [w for w in words if w not in NON_ITEM_WORDS]
    return None, " ".join(words).strip().title() if words else None


# ─────────────────────────────────────────────────────────────────────────────
# Product info / catalogue browsing
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_product_details(query: str, auth_token: str) -> str:
    if not auth_token:
        return "⚠️ Please log in to check product availability."
    token           = clean_auth_token(auth_token)
    item_candidates = extract_grocery_items(query)
    item_names      = [c["item"] for c in item_candidates if c["item"]]

    if len(item_names) > 1:
        blocks = await asyncio.gather(
            *[_fetch_single_product_block(n, token) for n in item_names]
        )
        return "\n\n".join(blocks) + "\n\nWould you like to add any of these to your cart?"

    clean_query = re.sub(r"\b(from|at|on|in)\s+sahachari\b", "", query, flags=re.IGNORECASE)
    clean_query = re.sub(r"[^\w\s]", " ", clean_query).strip()
    item_name   = item_names[0] if item_names else extract_product_name_from_query(clean_query)
    item_name   = normalise_service_name(item_name)
    return await _fetch_single_product_block(item_name, token)


async def _fetch_single_product_block(item_name: str, token: str) -> str:
    item_name  = normalise_service_name(item_name)
    clean_name = item_name.lower()

    # FIX: More comprehensive alternates
    alts = []
    if clean_name.endswith("oes"):
        alts.append(clean_name[:-2])
    elif clean_name.endswith("ies"):
        alts.append(clean_name[:-3] + "y")
    elif clean_name.endswith("s") and not clean_name.endswith("ss"):
        alts.append(clean_name[:-1])
    else:
        alts.append(clean_name + "s")

    if clean_name.endswith("s") and len(clean_name) > 3:
        stem = clean_name[:-1]
        if stem not in alts:
            alts.append(stem)

    search_terms = [item_name] + [a.title() for a in alts]

    try:
        products = []
        for term in search_terms:
            response = await http_client.get(
                f"{NESTJS_BACKEND_URL}/customer/products",
                params={"search": term},
                headers={"Authorization": token},
            )
            if response.status_code != 200:
                continue
            data       = response.json()
            candidates = data if isinstance(data, list) else (
                data.get("data") or data.get("products") or data.get("items") or []
            )
            if candidates:
                products = candidates;  break

        if not products:
            return f"❌ '{item_name}' is not available in Sahachari's catalogue right now."

        query_words = set(w for w in re.findall(r"[a-z]+", item_name.lower()) if len(w) >= 3)
        matched = []
        for prod in products:
            name = prod.get("name", "")
            if is_non_grocery_mismatch(name, item_name):
                continue
            cand       = name.lower()
            cand_words = set(w for w in re.findall(r"[a-z]+", cand) if len(w) >= 3)
            if item_name.lower() in cand or cand in item_name.lower() \
                    or len(query_words & cand_words) > 0:
                matched.append(prod)

        if not matched:
            return f"❌ '{item_name}' is not available right now."

        deduped     = {}
        store_cache = {}
        for prod in matched:
            key = (prod.get("name", "").lower(), str(prod.get("price", "")))
            if key not in deduped:
                deduped[key] = {"prod": prod.copy(), "total_stock": 0, "stores": []}
            deduped[key]["total_stock"] += prod.get("quantity", 0)
            store_id = prod.get("storeId", "")
            if isinstance(store_id, dict):
                store_name = store_id.get("name") or store_id.get("email") or "Unknown Store"
            elif isinstance(store_id, str) and store_id:
                store_name = await fetch_store_name_cache(store_id, token, store_cache)
            else:
                store_name = None
            if store_name and store_name not in deduped[key]["stores"]:
                deduped[key]["stores"].append(store_name)

        lines = []
        for _, entry in list(deduped.items())[:3]:
            prod             = entry["prod"]
            prod["quantity"] = entry["total_stock"]
            store_names      = entry["stores"]
            name             = prod.get("name", "Unknown")
            price            = prod.get("price", "N/A")
            final_price      = prod.get("finalPrice")
            quantity         = prod.get("quantity", 0)
            description      = prod.get("description", "")
            category         = prod.get("category", "")
            offers           = prod.get("offers", [])
            stock            = f"✅ In Stock ({quantity} available)" if quantity > 0 else "❌ Out of Stock"

            price_clean = re.sub(r"[₹,\s]", "", str(price))
            price_num   = re.sub(r"/.*$", "", price_clean)
            try:
                final_num = float(final_price) if final_price else None
                base_num  = float(price_num) if price_num else None
            except (ValueError, TypeError):
                final_num = base_num = None

            if final_num and base_num and final_num != base_num:
                price_line = (
                    f"₹{int(final_num) if final_num == int(final_num) else final_num}"
                    f" (was ₹{int(base_num) if base_num == int(base_num) else base_num})"
                )
            elif price_num:
                unit_match = re.search(r"/(.+)$", price_clean)
                suffix     = f"/{unit_match.group(1)}" if unit_match else ""
                price_line = f"₹{price_num}{suffix}"
            else:
                price_line = f"₹{price}"

            active_offers = [o for o in offers if o.get("isActive")]
            offer_line    = ""
            if active_offers:
                o = active_offers[0]
                offer_line = "\n  🏷️ Offer: " + (
                    f"{o['value']}% off" if o["type"] == "PERCENTAGE"
                    else f"₹{o['value']} flat off"
                )
            store_line = (
                f"\n  🏪 Store  : {store_names[0]}" if len(store_names) == 1
                else (f"\n  🏪 Stores : {', '.join(store_names)}" if store_names else "")
            )
            block = f"📦 {name}\n  💰 Price : {price_line}\n  📊 Stock : {stock}{store_line}"
            if category:    block += f"\n  🗂️ Category: {category}"
            if description: block += f"\n  📝 {description}"
            block += offer_line
            lines.append(block)

        return "\n\n".join(lines)

    except httpx.TimeoutException:
        return "❌ Request timed out. Please try again."
    except httpx.ConnectError:
        return "❌ Could not reach Sahachari server. Please try again shortly."
    except Exception as e:
        log.warning(f"_fetch_single_product_block failed for {item_name!r}: {e}")
        return "❌ Could not fetch product details right now."


async def browse_catalog(query: str, auth_token: str) -> str:
    if not auth_token:
        return "⚠️ Please log in to browse the catalogue."

    token     = clean_auth_token(auth_token)
    q         = query.lower().strip()
    cache_key = hashlib.md5(q.encode()).hexdigest()
    if cache_key in _catalog_cache:
        log.info(f"Catalog cache hit for: {q!r}")
        return _catalog_cache[cache_key]

    want_fruits    = any(x in q for x in CATEGORY_SYNONYMS["fruit"])
    want_leafy     = any(x in q for x in CATEGORY_SYNONYMS["leafy"])
    want_veg       = want_leafy or any(x in q for x in CATEGORY_SYNONYMS["vegetable"])
    want_beverages = any(x in q for x in CATEGORY_SYNONYMS["beverages"])
    want_snacks    = any(x in q for x in CATEGORY_SYNONYMS["snacks"])
    want_fastfood  = any(x in q for x in CATEGORY_SYNONYMS["fastfood"])
    want_food      = any(x in q for x in CATEGORY_SYNONYMS["food"])
    want_groceries = any(x in q for x in CATEGORY_SYNONYMS["groceries"])
    want_service   = any(x in q for x in CATEGORY_SYNONYMS["service"])
    want_rent      = any(x in q for x in CATEGORY_SYNONYMS["rent"])

    price_cap = None
    pm = re.search(
        r"(?:under|below|less than|at most|max|within)\s*(?:₹|rs\.?|rupees?)?\s*(\d+(?:\.\d+)?)", q,
    )
    if pm:
        try:
            price_cap = float(pm.group(1))
        except Exception:
            pass

    sort_mode  = None
    stock_only = bool(re.search(r"\b(currently in stock|in stock|available)\b", q))
    if re.search(r"\b(cheapest|lowest price|least expensive)\b", q):
        sort_mode = "asc"
    elif re.search(r"\b(most expensive|highest price|costliest)\b", q):
        sort_mode = "desc"

    try:
        response = await http_client.get(
            f"{NESTJS_BACKEND_URL}/customer/products",
            headers={"Authorization": token},
        )
        if response.status_code != 200:
            return "❌ Could not fetch catalogue right now. Please try again."
        data     = response.json()
        products = data if isinstance(data, list) else (
            data.get("data") or data.get("products") or data.get("items") or []
        )
        if not products:
            return "❌ The catalogue is empty right now."

        def norm_price(p):
            for key in ("finalPrice", "price"):
                v = p.get(key)
                if v is not None:
                    val = _parse_price_field(v)
                    if val:
                        return val
            return None

        filtered = []
        for p in products:
            name  = str(p.get("name", "")).strip()
            pname = name.lower()
            cat   = str(p.get("category", "")).lower().strip()

            is_fruit    = any(f in pname for f in KNOWN_FRUITS)
            is_veg      = any(v in pname for v in KNOWN_VEGETABLES)
            is_leafy    = any(l in pname for l in CATEGORY_SYNONYMS["leafy"])
            is_beverage = any(b in pname for b in ["beverage", "drink", "juice"]) or "beverage" in cat
            is_snack    = "snack" in pname or "snack" in cat
            is_fastfood = any(f in pname for f in ["burger", "shawarma", "sandwich", "fast food"]) \
                          or "fast food" in cat
            is_food     = "food" in cat or "homemade" in cat or "home made" in cat
            is_grocery  = "groceries" in cat or "grocery" in cat
            is_service  = "service" in cat or "service" in pname
            is_rent     = "rent" in cat or "rent" in pname

            if want_fruits    and not is_fruit:    continue
            if want_leafy     and not is_leafy:    continue
            if want_veg       and not is_veg:      continue
            if want_beverages and not is_beverage: continue
            if want_snacks    and not is_snack:    continue
            if want_fastfood  and not is_fastfood: continue
            if want_food      and not is_food:     continue
            if want_groceries and not is_grocery:  continue
            if want_service   and not is_service:  continue
            if want_rent      and not is_rent:     continue

            if price_cap is not None:
                pv = norm_price(p)
                if pv is None or pv > price_cap:
                    continue
            if stock_only and (p.get("quantity", 0) or 0) <= 0:
                continue
            filtered.append(p)

        seen_names = set()
        deduped    = []
        for p in filtered:
            norm = p.get("name", "").strip().lower()
            if norm not in seen_names:
                seen_names.add(norm);  deduped.append(p)
        filtered = deduped

        if sort_mode == "asc":
            filtered.sort(key=lambda x: norm_price(x) if norm_price(x) is not None else float("inf"))
        elif sort_mode == "desc":
            filtered.sort(key=lambda x: norm_price(x) if norm_price(x) is not None else float("-inf"), reverse=True)

        if not filtered:
            return "❌ No matching items were found in the catalogue for that filter."

        if want_leafy:       header = "🥬 **Fresh Leafy Vegetables**"
        elif want_fruits:    header = "🥭 **Fresh Fruits**"
        elif want_veg and price_cap:
            header = f"🥦 **Vegetables under ₹{int(price_cap) if price_cap == int(price_cap) else price_cap}**"
        elif want_veg:       header = "🥦 **Fresh Vegetables**"
        elif want_beverages: header = "🥤 **Beverages & Drinks**"
        elif want_snacks:    header = "🍿 **Snacks**"
        elif want_fastfood:  header = "🍔 **Fast Food**"
        elif want_food:      header = "🍱 **Homemade Food**"
        elif want_groceries: header = "🛒 **Groceries**"
        elif want_service:   header = "🔧 **Services**"
        elif want_rent:      header = "📦 **Items for Rent**"
        elif price_cap:
            header = f"🛍️ **Items under ₹{int(price_cap) if price_cap == int(price_cap) else price_cap}**"
        elif sort_mode == "asc":   header = "🛍️ **Cheapest Items**"
        elif sort_mode == "desc":  header = "🛍️ **Most Expensive Items**"
        elif stock_only:           header = "🛍️ **Items Currently In Stock**"
        else:                      header = "🛍️ **Matching Items**"

        lines = [f"{header}\n"]
        for p in filtered[:12]:
            name  = p.get("name", "Unknown").strip()
            price = p.get("finalPrice") or p.get("price") or "N/A"
            qty   = int(p.get("quantity", 0))
            try:
                price_str = f"₹{float(price):.0f}" if float(price) == int(float(price)) else f"₹{price}"
            except Exception:
                price_str = f"₹{price}"
            stock_line = f"✅ **{qty}** in stock" if qty > 0 else "❌ Out of stock"
            lines.append(f"**{name}**")
            lines.append(f"💰 {price_str}     {stock_line}")
            lines.append("─" * 35)
            lines.append("")

        result = "\n".join(lines)
        _catalog_cache[cache_key] = result
        return result

    except httpx.TimeoutException:
        return "❌ Request timed out. Please try again."
    except httpx.ConnectError:
        return "❌ Could not reach Sahachari server. Please try again shortly."
    except Exception as e:
        log.warning(f"browse_catalog failed: {e}")
        return "❌ Could not fetch catalogue right now. Please try again."


# ─────────────────────────────────────────────────────────────────────────────
# RAG — query expansion + hybrid retrieval + hallucination guard
# ─────────────────────────────────────────────────────────────────────────────

_GROUNDING_STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "did", "you", "your", "have", "has",
    "what", "how", "why", "where", "when", "which", "who", "this", "that", "it",
    "of", "to", "in", "on", "for", "and", "or", "with", "about", "my", "me",
    "please", "can", "could", "would", "will", "we", "us", "our", "any",
}

_HALLUCINATION_MARKERS = re.compile(
    r"\b(our website|visit us at|call us at|phone number|email us|"
    r"instagram|facebook|twitter|download our app|app store|play store|"
    r"promo code|voucher code|discount code|referral code)\b",
    re.IGNORECASE,
)

def _query_grounded_in_context(query: str, context: str, min_ratio: float = 0.5) -> bool:
    query_words = set(
        w for w in re.findall(r"[a-z]+", query.lower())
        if len(w) >= 3 and w not in _GROUNDING_STOPWORDS
    )
    if not query_words:
        return True
    matched = sum(1 for w in query_words if w in context.lower())
    return (matched / len(query_words)) >= min_ratio

def _llm_response_is_safe(text: str) -> bool:
    return not _HALLUCINATION_MARKERS.search(text)

async def _expand_query_for_rag(query: str, memory: dict, session: dict) -> str:
    context_terms = []
    entities      = memory.get("entities", {})

    if entities.get("household_size"):
        context_terms.append(entities["household_size"])

    history = memory.get("conversation_history", [])
    for msg in reversed(history[-4:]):
        if msg["role"] == "assistant":
            product_mentions = re.findall(
                r"\*\*([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\*\*", msg["content"]
            )
            if product_mentions:
                context_terms.extend(product_mentions[:2]);  break

    if context_terms:
        return f"{query} {' '.join(context_terms)}"
    return query

async def answer_from_history(query: str, conversation_history: list) -> str:
    if qwen_pipeline is None:
        return "ℹ️ I'm not sure what you're referring to. Could you give more details?"
    last_assistant_msg = next(
        (m["content"] for m in reversed(conversation_history) if m["role"] == "assistant"), "",
    )
    if not last_assistant_msg:
        return "ℹ️ I'm not sure what you're referring to. Could you give more details?"
    loop     = asyncio.get_running_loop()
    messages = [
        {"role": "system", "content": (
            "You are Sahachari grocery assistant. "
            "Answer the follow-up question using ONLY the previous response provided. "
            "Do not invent any information. Keep answer under 2 sentences."
        )},
        {"role": "assistant", "content": last_assistant_msg},
        {"role": "user",      "content": query},
    ]
    def _gen():
        return qwen_pipeline(
            messages, max_new_tokens=100, return_full_text=False, do_sample=False,
        )[0]["generated_text"]
    try:
        return (await loop.run_in_executor(None, _gen)).strip()
    except Exception as e:
        log.warning(f"answer_from_history failed: {e}")
        return "ℹ️ I'm not sure what you're referring to. Could you give more details?"

async def get_rag_response(
    query: str,
    memory: dict | None = None,
    session: dict | None = None,
    conversation_history: list | None = None,
    is_faq: bool = False,
) -> str:
    if memory is None:   memory = {}
    if session is None:  session = {}
    if conversation_history is None:  conversation_history = []

    if BLOCKED_RAG_TOPICS.search(query):
        return SERVICE_DENIAL_RESPONSE

    if is_faq:
        faq_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        if faq_key in _faq_answer_cache:
            log.info(f"FAQ cache hit for: {query!r}")
            return _faq_answer_cache[faq_key]

    if collection is None or embedding_model is None:
        return (
            "ℹ️ My knowledge base isn't loaded right now. "
            "I can help you order groceries, check product availability, or track your order!"
        )

    try:
        loop = asyncio.get_running_loop()

        expanded_query = await _expand_query_for_rag(query, memory, session)

        vec_original = await loop.run_in_executor(None, embedding_model.encode, query)
        vec_expanded = (
            await loop.run_in_executor(None, embedding_model.encode, expanded_query)
            if expanded_query != query
            else vec_original
        )

        results_a = collection.query(query_embeddings=[vec_original.tolist()], n_results=3)
        results_b = collection.query(query_embeddings=[vec_expanded.tolist()],  n_results=3)

        seen           = set()
        merged_chunks  = []
        docs_a = (results_a.get("documents") or [[]])[0]
        docs_b = (results_b.get("documents") or [[]])[0]
        for chunk in docs_a + docs_b:
            if chunk not in seen:
                seen.add(chunk);  merged_chunks.append(chunk)

        if not merged_chunks:
            if is_faq:
                return (
                    "📋 For detailed policy information, please contact Sahachari support "
                    "or visit the Help section in the app.\n\n"
                    "I can help you with orders, product availability, or your cart!"
                )
            return "ℹ️ I'm not quite sure about that. Is there a specific grocery item or order I can help you with?"

        if reranker is not None:
            pairs  = [[query, chunk] for chunk in merged_chunks]
            scores = await loop.run_in_executor(None, reranker.compute_score, pairs)
            if isinstance(scores, float):
                scores = [scores]
            ranked_chunks        = sorted(zip(merged_chunks, scores), key=lambda x: x[1], reverse=True)
            top_chunk, top_score = ranked_chunks[0]
            log.info(f"RAG top score: {top_score:.3f} for query: {query!r} (is_faq={is_faq})")

            effective_threshold = RELEVANCE_THRESHOLD + 3.0 if is_faq else RELEVANCE_THRESHOLD
            if top_score < effective_threshold:
                if is_faq:
                    return (
                        "📋 For detailed information about our refund policy, delivery charges, "
                        "payment methods, and cancellation policy, please contact Sahachari "
                        "support or check the Help section in the app."
                    )
                if conversation_history:
                    return await answer_from_history(query, conversation_history)
                return (
                    "ℹ️ Sahachari is a grocery delivery service. I can help you with:\n"
                    "  • Ordering fresh groceries\n"
                    "  • Checking product prices and availability\n"
                    "  • Viewing your cart or tracking your order\n\n"
                    "Try: 'show me fruits' or 'what is the price of onions'?"
                )
        else:
            top_chunk = merged_chunks[0]

        cleaned = re.sub(r"#[^\n]*\n?", "", top_chunk)
        cleaned = re.sub(r"^={5,}.*?={5,}\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"-{5,}", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        context = cleaned.strip()

        if not _query_grounded_in_context(query, context):
            log.info(f"RAG guard: query not grounded for {query!r}")
            if is_faq:
                return (
                    "📋 For detailed policy information, please contact Sahachari support "
                    "or visit the Help section in the app."
                )
            return (
                "ℹ️ Sahachari is a grocery delivery service. I can help you with:\n"
                "  • Ordering fresh groceries\n"
                "  • Checking product prices and availability\n"
                "  • Viewing your cart or tracking your order\n\n"
                "Try: 'show me fruits' or 'what is the price of onions'?"
            )

        if qwen_pipeline is not None:
            if memory_store:
                llm_messages = memory_store.build_llm_context(memory, session, query)
            else:
                llm_messages = [{"role": "system", "content": (
                    "You are Sahachari, a delivery assistant. "
                    "Answer ONLY from the provided Context. "
                    "NEVER invent phone numbers, promo codes, or features not in Context. "
                    "Keep answer under 2 sentences."
                )}]

            llm_messages.append({
                "role":    "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            })

            def _gen():
                return qwen_pipeline(
                    llm_messages, max_new_tokens=150, return_full_text=False, do_sample=False,
                )[0]["generated_text"]
            raw_answer = (await loop.run_in_executor(None, _gen)).strip()

            if not _llm_response_is_safe(raw_answer):
                log.warning(f"LLM hallucination detected for query: {query!r}")
                raw_answer = context

            if is_faq:
                _faq_answer_cache[faq_key] = raw_answer
            return raw_answer

        if is_faq:
            _faq_answer_cache[faq_key] = context
        return context

    except Exception as e:
        log.exception(f"get_rag_response failed for query={query!r}")
        return "ℹ️ I'm having trouble retrieving that right now. How can I help you with your shopping?"


# ─────────────────────────────────────────────────────────────────────────────
# Conversation repair handler
# ─────────────────────────────────────────────────────────────────────────────

async def handle_repair(query: str, memory: dict, session: dict) -> str:
    history = memory.get("conversation_history", [])
    if len(history) >= 2:
        last_user_query = history[-2]["content"]
        return (
            "Sorry about that! Let me try again.\n\n"
            f"You said: *'{last_user_query}'*\n"
            "Could you rephrase so I can get it right?"
        )
    return "Sorry about the confusion! Could you rephrase what you need?"


# ─────────────────────────────────────────────────────────────────────────────
# Normalize confirmation string (strip punctuation/spaces)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_confirmation(text: str) -> str:
    """Remove punctuation and normalize spaces for confirmation matching."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower().strip())
    return re.sub(r"\s+", " ", cleaned).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Chat endpoint
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query:      str
    session_id: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest, authorization: str = Header(None)):
    query        = req.query.strip()
    session_id   = req.session_id
    cart_updated = False
    action       = None

    try:
        if session_store is None or memory_store is None:
            return {
                "response":     "⚠️ Service is still starting up. Please try again in a moment.",
                "cart_updated": False,
                "action":       None,
            }

        current_user_session, current_memory = await asyncio.gather(
            session_store.get(session_id),
            memory_store.get(session_id),
        )

        query = resolve_pronoun(query, current_memory.get("conversation_history", []), current_user_session)
        log.info(f"Resolved query: {query!r}")

        # ── Early-exit guards ────────────────────────────────────────────────

        if SERVICE_DENIAL_PATTERNS.search(query):
            response_text = SERVICE_DENIAL_RESPONSE
            memory_store.add_turn(current_memory, query, response_text)
            await asyncio.gather(
                session_store.save(session_id, current_user_session),
                memory_store.save(session_id, current_memory),
            )
            return {"response": response_text, "cart_updated": False, "action": None}

        if META_QUESTION_PATTERNS.search(query):
            mem_summary = memory_store.build_context_summary(current_memory)
            response_text = (
                "🧠 Here's what I remember about you:\n\n"
                + mem_summary
                + "\n\nThis helps me personalise your Sahachari experience! 😊"
            ) if mem_summary else META_QUESTION_RESPONSE
            memory_store.add_turn(current_memory, query, response_text)
            await asyncio.gather(
                session_store.save(session_id, current_user_session),
                memory_store.save(session_id, current_memory),
            )
            return {"response": response_text, "cart_updated": False, "action": None}

        memory_store.extract_preferences(current_memory, query)
        memory_context = memory_store.build_context_summary(current_memory)

        if len(current_memory.get("conversation_history", [])) > memory_store.SUMMARY_THRESHOLD * 2:
            asyncio.create_task(memory_store.maybe_summarise(current_memory))

        response_text    = "I can help you with grocery orders, product info, your cart, wishlist, or order tracking. What would you like to do?"
        items_just_added: list[str] = []

        # ── Pending quantity / confirmation reply ─────────────────────────────
        if current_user_session["pending_item"] is not None:
            # FIX: normalize confirmation strings to handle punctuation like "yes,add all"
            q_lower = normalize_confirmation(query)

            # FIX: checkout_confirm pending intent — "yes" after viewing cart
            if current_user_session.get("pending_intent") == "checkout_confirm":
                current_user_session["pending_item"]   = None
                current_user_session["pending_intent"] = None
                if q_lower in {"yes", "yeah", "yep", "ok", "okay", "sure", "alright",
                               "checkout", "place order", "confirm", "yes please",
                               "go ahead", "proceed"}:
                    response_text, cart_updated = await forward_nestjs_order_placement(authorization)
                    if cart_updated:
                        action = "checkout"
                        current_memory["last_order_summary"] = (
                            f"Checked out on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
                        )
                else:
                    response_text = "No problem! Let me know if you need anything else. 😊"
                response_text = personalise_response(
                    response_text, current_memory, current_user_session,
                    "checkout", items_just_added,
                )
                memory_store.add_turn(current_memory, query, response_text)
                await asyncio.gather(
                    session_store.save(session_id, current_user_session),
                    memory_store.save(session_id, current_memory),
                )
                return {"response": response_text, "cart_updated": cart_updated, "action": action}

            # FIX: recipe_confirm with normalized confirmation
            if current_user_session.get("pending_intent") == "recipe_confirm":
                RECIPE_YES = {
                    "yes", "yeah", "yep", "ok", "okay", "sure", "alright",
                    "yes add all", "add all", "add them all", "yes please",
                    "yes add them", "add all of them", "add everything",
                }
                if q_lower in RECIPE_YES:
                    recipe_items = current_user_session.get("recipe_items") or []
                    if recipe_items:
                        response_text = await forward_to_nestjs_cart(
                            recipe_items, authorization, current_memory
                        )
                        cart_updated  = True
                        for it in recipe_items:
                            memory_store.record_item_ordered(current_memory, it["item"])
                            items_just_added.append(it["item"])
                        current_user_session["pending_item"]   = None
                        current_user_session["pending_intent"] = None
                        current_user_session["recipe_items"]   = None
                    else:
                        response_text = "I couldn't find the recipe items. Try searching individually."
                        current_user_session["pending_item"]   = None
                        current_user_session["pending_intent"] = None
                else:
                    current_user_session["pending_item"]   = None
                    current_user_session["pending_intent"] = None
                    current_user_session["recipe_items"]   = None
                    response_text = "No problem! What else can I help you with?"
                response_text = personalise_response(
                    response_text, current_memory, current_user_session,
                    "order", items_just_added,
                )
                memory_store.add_turn(current_memory, query, response_text)
                await asyncio.gather(
                    session_store.save(session_id, current_user_session),
                    memory_store.save(session_id, current_memory),
                )
                return {"response": response_text, "cart_updated": cart_updated, "action": action}

            if q_lower in {"yes", "yeah", "yep", "ok", "okay", "sure", "alright"}:
                pending    = current_user_session["pending_item"]
                rem_qty, rem_unit = memory_store.get_preferred_quantity(current_memory, pending["item"])
                response_text = (
                    f"How much **{pending['item']}** would you like? "
                    f"Last time you ordered {format_qty(rem_qty, rem_unit)}. Say a quantity or 'same'."
                    if rem_qty
                    else f"How much {pending['item']} would you like? (e.g. '1 kg', '500 g')"
                )
                memory_store.add_turn(current_memory, query, response_text)
                await asyncio.gather(
                    session_store.save(session_id, current_user_session),
                    memory_store.save(session_id, current_memory),
                )
                return {"response": response_text, "cart_updated": False, "action": None}

            elif q_lower.strip() in {"same", "same as before", "same quantity", "usual"}:
                pending           = current_user_session["pending_item"]
                rem_qty, rem_unit = memory_store.get_preferred_quantity(current_memory, pending["item"])
                if rem_qty:
                    pending["quantity"] = rem_qty
                    pending["unit"]     = rem_unit or "piece"
                    current_user_session["pending_item"]   = None
                    current_user_session["pending_intent"] = None
                    response_text = await forward_to_nestjs_cart([pending], authorization, current_memory)
                    cart_updated  = True
                    memory_store.record_item_ordered(current_memory, pending["item"], rem_qty, rem_unit or "piece")
                    items_just_added.append(pending["item"])
                else:
                    response_text = (
                        f"I don't have a saved quantity for {pending['item']} yet. "
                        f"How much would you like? (e.g. '1 kg', '500 g')"
                    )
                response_text = personalise_response(
                    response_text, current_memory, current_user_session,
                    "order", items_just_added,
                )
                memory_store.add_turn(current_memory, query, response_text)
                await asyncio.gather(
                    session_store.save(session_id, current_user_session),
                    memory_store.save(session_id, current_memory),
                )
                return {"response": response_text, "cart_updated": cart_updated, "action": action}

            else:
                quantity, unit   = parse_quantity_reply(query)
                pending          = current_user_session["pending_item"]
                intent_pending   = current_user_session["pending_intent"]
                pending_resolved = True

                if quantity is not None:
                    pending["quantity"] = quantity
                    pending["unit"]     = unit
                    current_user_session["pending_item"]   = None
                    current_user_session["pending_intent"] = None

                    # FIX: After resolving first item, check slot_items for more pending items
                    slot_items = current_user_session.get("slot_items", [])
                    remaining  = [i for i in slot_items
                                  if i["item"].lower() != pending["item"].lower()]

                    if intent_pending == "wishlist":
                        response_text = add_to_wishlist([pending], current_user_session)
                    else:
                        response_text = await forward_to_nestjs_cart(
                            [pending], authorization, current_memory
                        )
                        cart_updated  = True
                        memory_store.record_item_ordered(current_memory, pending["item"], quantity, unit)
                        items_just_added.append(pending["item"])

                    # Ask for next item in the queue
                    if remaining:
                        next_pending = remaining[0]
                        current_user_session["slot_items"]     = remaining
                        current_user_session["pending_item"]   = next_pending
                        current_user_session["pending_intent"] = intent_pending or "order"
                        response_text += (
                            f"\n\nHow much **{next_pending['item']}** would you like? "
                            f"(e.g. '1 kg', '500 g')"
                        )
                    else:
                        current_user_session["slot_items"] = []

                else:
                    new_intent, new_conf = detect_intent(query, current_user_session, current_memory)
                    if (new_conf >= 0.8 and new_intent not in ("order", "wishlist")) \
                            or looks_like_service_query(query.lower()) \
                            or new_intent == "rag":
                        current_user_session["pending_item"]   = None
                        current_user_session["pending_intent"] = None
                        current_user_session["slot_items"]     = []
                        pending_resolved = False
                    else:
                        response_text = (
                            f"Sorry, I didn't catch that. "
                            f"How much {pending['item']} would you like? "
                            f"(e.g. '1 kg', '500 g', '2 pieces')"
                        )

                if pending_resolved:
                    response_text = personalise_response(
                        response_text, current_memory, current_user_session,
                        "order", items_just_added,
                    )
                    memory_store.add_turn(current_memory, query, response_text)
                    await asyncio.gather(
                        session_store.save(session_id, current_user_session),
                        memory_store.save(session_id, current_memory),
                    )
                    return {"response": response_text, "cart_updated": cart_updated, "action": action}

        # ── Cheapest follow-up context injection ──────────────────────────────
        if CHEAPEST_PATTERNS.search(query):
            last_cat = current_user_session.get("last_browse_category")
            if last_cat:
                cat_keyword = CAT_TO_BROWSE_KEYWORD.get(last_cat, last_cat)
                query       = f"cheapest {cat_keyword}"
                log.info(f"Cheapest injection → {query!r}")

        # ── Intent routing ────────────────────────────────────────────────────
        intent, confidence = detect_intent(query, current_user_session, current_memory)

        # FIX: Clear stale pending state when a clear new order intent arrives
        if intent == "order" and current_user_session.get("pending_intent") in (
            "recipe_confirm", "checkout_confirm"
        ):
            log.info(f"Clearing stale pending_intent={current_user_session['pending_intent']!r} on new order")
            current_user_session["pending_item"]   = None
            current_user_session["pending_intent"] = None
            current_user_session["recipe_items"]   = None
            current_user_session["slot_items"]     = []

        if confidence < 0.3 and intent != "checkout":
            intent = "rag"
        elif confidence < 0.3 and intent == "checkout":
            log.info("Checkout intent preserved despite low confidence")

        log.info(f"Intent: {intent!r} (confidence={confidence:.2f}) for query: {query!r}")
        record_intent(current_user_session, intent)

        # ── Handlers ──────────────────────────────────────────────────────────

        if intent == "repair":
            response_text = await handle_repair(query, current_memory, current_user_session)

        elif intent == "greeting":
            time_greet = get_time_greeting()
            sentiment  = current_memory.get("sentiment_score", 0.0)
            tone_prefix = "Welcome back! Hope things are going better today. " if sentiment < -0.3 else ""
            freq = current_memory.get("frequent_items", {})
            if freq:
                top_items = memory_store.get_top_items(current_memory, n=1)
                if top_items:
                    top_item  = top_items[0]
                    rem_qty, rem_unit = memory_store.get_preferred_quantity(current_memory, top_item)
                    qty_hint  = f" ({format_qty(rem_qty, rem_unit)} as usual)" if rem_qty else ""
                    base_msg  = (
                        f"{time_greet}! Welcome back to Sahachari 😊 "
                        f"Shall I add your usual **{top_item}**{qty_hint} to the cart?"
                    )
                    response_text = tone_prefix + base_msg
                    current_user_session["pending_item"]   = {"item": top_item, "quantity": None, "unit": None}
                    current_user_session["pending_intent"] = "order"
                else:
                    response_text = (
                        f"{tone_prefix}{time_greet}! Welcome back to Sahachari 😊 "
                        "What can I get for you today?"
                    )
            else:
                response_text = (
                    f"{tone_prefix}{time_greet}! Welcome to Sahachari 😊 "
                    "What can I get for you today? You can say things like:\n"
                    "  • 'Show me vegetables'\n"
                    "  • 'I want 2 kg onions'\n"
                    "  • 'What's the price of mango?'"
                )
            response_text = personalise_response(
                response_text, current_memory, current_user_session, "greeting",
            )

        elif intent == "courtesy":
            response_text = "You're very welcome! 😊 Always happy to help you shop fresh with Sahachari."

        elif intent == "clarify":
            top_items = memory_store.get_top_items(current_memory, n=3)
            if top_items:
                response_text = (
                    f"I can help with orders, product info, cart, wishlist, or order tracking.\n\n"
                    f"Based on your history, you often order: **{', '.join(top_items)}**. Want any of these?"
                )
            else:
                response_text = (
                    "I can help you with grocery orders, product info, your cart, "
                    "wishlist, or order tracking. What would you like to do?"
                )

        elif intent == "view_cart":
            response_text = await fetch_nestjs_cart_summary(authorization, current_memory)
            # After showing cart, offer checkout confirmation if cart not empty
            if "empty" not in response_text.lower():
                current_user_session["pending_item"]   = {"item": "__checkout__", "quantity": 1, "unit": "piece"}
                current_user_session["pending_intent"] = "checkout_confirm"

        elif intent == "wishlist_remove":
            response_text = remove_from_wishlist(query, current_user_session)

        elif intent == "wishlist":
            if any(p in query.lower() for p in ["show", "view", "my wishlist", "see"]):
                response_text = view_wishlist(current_user_session)
            else:
                items = extract_grocery_items(query)
                if items:
                    missing_qty = [i for i in items if i["quantity"] is None]
                    has_qty     = [i for i in items if i["quantity"] is not None]
                    lines       = []
                    if has_qty:
                        lines.append(add_to_wishlist(has_qty, current_user_session))
                    if missing_qty:
                        pending = missing_qty[0]
                        current_user_session["pending_item"]   = pending
                        current_user_session["pending_intent"] = "wishlist"
                        lines.append(
                            f"How much {pending['item']} would you like to save? "
                            f"(e.g. '1 kg', '2 pieces')"
                        )
                    response_text = "\n\n".join(lines)
                else:
                    response_text = "Tell me what to save, e.g. add 1 kg apples to wishlist."

        elif intent == "status":
            order_id, item_name = extract_order_query_target(query)
            q_lower_status = query.lower()

            # FIX: Expanded show_all detection
            show_all = bool(re.search(
                r"\b(all my orders|all orders|every order|order history|list.*orders|"
                r"show.*all.*orders|show my orders|my orders|show orders|"
                r"list.*my.*orders|past orders|previous orders)\b",
                q_lower_status,
            )) and not re.search(
                r"\b(last|latest|recent|current|previous order|my order\b)\b",
                q_lower_status,
            )

            if show_all:
                response_text = await fetch_all_orders_status(authorization)
            else:
                response_text = await fetch_nestjs_order_status(
                    authorization, order_id=order_id, item_name=item_name,
                )

        elif intent == "cancel":
            response_text = await forward_nestjs_order_cancellation(authorization)
            cart_updated  = True

        elif intent == "clear_cart":
            response_text = await clear_nestjs_cart(authorization)
            cart_updated  = True
            current_user_session.setdefault("analytics", {})["cart_abandonment"] = (
                current_user_session["analytics"].get("cart_abandonment", 0) + 1
            )
            response_text = personalise_response(
                response_text, current_memory, current_user_session, "clear_cart",
            )

        elif intent == "delete_cart":
            item_name = extract_item_name(query)
            if item_name:
                response_text = await delete_from_nestjs_cart(item_name, authorization)
                cart_updated  = True
            else:
                response_text = "Which item would you like to remove? (e.g. 'remove onion from cart')"

        elif intent == "update_cart":
            response_text = await update_nestjs_cart_quantity(query, authorization)
            cart_updated  = True

        elif intent == "checkout":
            response_text, cart_updated = await forward_nestjs_order_placement(authorization)
            if cart_updated:
                action = "checkout"
                current_memory["last_order_summary"] = (
                    f"Checked out on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
                )

        elif intent == "service_booking":
            service_match = SERVICE_QUERY_PATTERN.search(query.lower())
            if service_match:
                service_name  = normalise_service_name(service_match.group(0).strip())
                items_to_add  = [{"item": service_name, "quantity": 1, "unit": "piece"}]
                response_text = await forward_to_nestjs_cart(
                    items_to_add, authorization, current_memory
                )
                cart_updated  = True
            else:
                response_text = (
                    "🔧 Which service would you like to add? "
                    "Say 'show me services' to see what's available."
                )

        elif intent == "browse":
            response_text = await browse_catalog(query, authorization)
            q_lower = query.lower()
            new_cat = next(
                (cat for cat, syns in [
                    ("fruits",     ["fruit", "fruits"]),
                    ("vegetables", ["vegetable", "vegetables", "veg"]),
                    ("leafy",      ["leafy", "greens"]),
                    ("beverages",  ["beverage", "drink", "juice"]),
                    ("snacks",     ["snack", "snacks"]),
                    ("fastfood",   ["fast food", "fastfood"]),
                    ("food",       ["food", "homemade"]),
                    ("groceries",  ["grocery", "groceries"]),
                    ("service",    ["service", "services"]),
                    ("rent",       ["rent", "rental"]),
                ] if any(s in q_lower for s in syns)),
                None,
            )
            if new_cat:
                current_user_session["last_browse_category"] = new_cat
                log.info(f"Set last_browse_category={new_cat!r} for session {session_id}")

        elif intent == "product_info":
            response_text = await fetch_product_details(query, authorization)
            mentioned_names = [c["item"] for c in extract_grocery_items(query) if c.get("item")]
            if not mentioned_names:
                name_candidate = extract_product_name_from_query(query)
                if name_candidate:
                    mentioned_names = [name_candidate]
            update_entity_tracker(current_user_session, mentioned_names)
            for mn in mentioned_names:
                record_product_viewed(current_user_session, mn)
            entities = current_memory.setdefault("entities", {})
            existing_seen = entities.get("last_seen_products", [])
            for mn in mentioned_names:
                if mn not in existing_seen:
                    existing_seen.append(mn)
            entities["last_seen_products"] = existing_seen[-5:]

        elif intent == "preference":
            prefs    = current_memory.get("preferences", {})
            likes    = prefs.get("likes", [])
            dislikes = prefs.get("dislikes", [])
            diet     = prefs.get("diet", [])

            ack_parts = []
            if likes:    ack_parts.append(f"love **{', '.join(likes[-2:])}**")
            if dislikes: ack_parts.append(f"avoid **{', '.join(dislikes[-2:])}**")

            if ack_parts:
                response_text = (
                    f"Got it! I've noted that you {' and '.join(ack_parts)}. 😊 "
                    "I'll keep this in mind when making suggestions."
                )
            else:
                response_text = "Thanks for sharing that! I'll keep your preferences in mind. 😊"

            if diet:
                response_text += f"\n\nAlso noted your dietary preferences: **{', '.join(diet)}**."

            entities = current_memory.get("entities", {})
            if entities.get("allergies"):
                response_text += (
                    f"\n\n⚠️ Allergy noted: **{', '.join(entities['allergies'])}**. "
                    "I'll make sure not to suggest these."
                )
            if current_memory.get("organic_preference"):
                response_text += "\n🌿 I'll prioritise organic options for you."
            sp = current_memory.get("spice_preference", "medium")
            if sp != "medium":
                response_text += f"\n🌶️ Noted your spice preference: {sp}."

            response_text += (
                "\n\nWould you like to see what we have available? "
                "Try 'show me vegetables' or 'show me fruits'."
            )

        elif intent == "complaint":
            q_lower = query.lower()
            if any(w in q_lower for w in ["rotten", "spoiled", "expired", "bad quality",
                                           "damaged", "dirty", "unhygienic", "broken"]):
                response_text = (
                    "😔 We're really sorry to hear that! That's not the experience we want for you.\n\n"
                    "To get a refund or replacement, please:\n"
                    "  1. Go to **My Orders** in the app\n"
                    "  2. Select the affected order\n"
                    "  3. Tap **Report an Issue** and attach a photo if possible\n\n"
                    "Our support team will resolve it within 24 hours. "
                    "You can also reach us at support@sahachari.in 📧"
                )
            elif any(w in q_lower for w in ["wrong order", "wrong item",
                                             "someone else's order", "someone else"]):
                response_text = (
                    "😮 Oh no, that's a serious mix-up! We apologize for this.\n\n"
                    "Please **do not consume** the items. Here's what to do:\n"
                    "  1. Go to **My Orders** → tap **Report an Issue**\n"
                    "  2. Select 'Wrong order delivered'\n"
                    "  3. Our team will arrange an immediate pickup and redeliver your correct order\n\n"
                    "You'll receive a full refund or re-delivery within 24 hours. "
                    "For urgent help: support@sahachari.in 📧"
                )
            elif any(w in q_lower for w in ["didn't show up", "no show", "not arrived",
                                             "didn't arrive", "cleaning person", "helper"]):
                response_text = (
                    "😔 We sincerely apologize — this should not have happened.\n\n"
                    "Please contact us immediately:\n"
                    "  • 📧 support@sahachari.in\n"
                    "  • Or tap **Help** in the app → 'Service not arrived'\n\n"
                    "We will either reschedule at the earliest available slot "
                    "or process a full refund for the booking."
                )
            elif any(w in q_lower for w in ["refund", "charged twice", "double charged", "overcharged"]):
                response_text = (
                    "💰 We're sorry about the payment issue!\n\n"
                    "To raise a refund request:\n"
                    "  1. Go to **My Orders** → select the order\n"
                    "  2. Tap **Report an Issue** → 'Payment problem'\n\n"
                    "Refunds are typically processed within **3–5 business days** "
                    "back to your original payment method. "
                    "For urgent cases: support@sahachari.in 📧"
                )
            else:
                response_text = (
                    "😔 We're sorry you're facing an issue!\n\n"
                    "Please reach out to our support team:\n"
                    "  • 📧 support@sahachari.in\n"
                    "  • Or tap **Help** in the app to report the problem\n\n"
                    "We'll make sure it gets resolved as quickly as possible."
                )

        elif intent == "reorder":
            try:
                token      = clean_auth_token(authorization)
                orders_res = await http_client.get(
                    f"{NESTJS_BACKEND_URL}/customer/orders",
                    headers={"Authorization": token},
                )
                if orders_res.status_code == 200:
                    orders = orders_res.json()
                    if orders:
                        last_items    = orders[0].get("items", [])
                        reorder_list  = [
                            {"item": extract_product_name(it), "quantity": it.get("quantity", 1), "unit": "piece"}
                            for it in last_items
                            if extract_product_name(it) != "Unknown Item"
                        ]
                        if reorder_list:
                            cart_response = await forward_to_nestjs_cart(
                                reorder_list, authorization, current_memory
                            )
                            cart_updated  = True
                            names = ", ".join(i["item"] for i in reorder_list[:3])
                            if len(reorder_list) > 3:
                                names += f" + {len(reorder_list) - 3} more"
                            response_text = f"✅ Reordering your last order ({names})!\n\n{cart_response}"
                            for it in reorder_list:
                                memory_store.record_item_ordered(current_memory, it["item"])
                                items_just_added.append(it["item"])
                        else:
                            response_text = "I couldn't find any items in your last order to reorder."
                    else:
                        response_text = "You don't have any previous orders to reorder from."
                else:
                    response_text = "I couldn't fetch your previous orders right now. Please try again."
            except Exception as e:
                log.warning(f"Reorder failed: {e}")
                response_text = "I couldn't process the reorder right now. Please try again shortly."

        elif intent == "recipe":
            q_lower    = query.lower()
            dish_match = re.search(
                r"(?:for making|for cooking|for|to make|ingredients? for)\s+([a-z][a-z\s]+?)(?:\s+for\s+\d+|\s*$|\?)",
                q_lower,
            )
            dish = dish_match.group(1).strip().title() if dish_match else "your dish"

            ingredient_list = None
            for key, ingredients in RECIPE_INGREDIENTS.items():
                if key in q_lower:
                    ingredient_list = ingredients;  dish = key.title();  break

            if ingredient_list:
                ingredient_list = memory_store.get_safe_suggestions(current_memory, ingredient_list)

                product_lookups = await asyncio.gather(
                    *[find_product_id(ing, clean_auth_token(authorization)) for ing in ingredient_list],
                    return_exceptions=True,
                )
                available_items = []
                missing_items   = []
                for ing, result in zip(ingredient_list, product_lookups):
                    if isinstance(result, Exception) or result[0] is None:
                        missing_items.append(ing)
                    else:
                        available_items.append({"item": result[1] or ing, "quantity": 1, "unit": "piece"})

                if available_items:
                    names = ", ".join(i["item"] for i in available_items)
                    response_text = (
                        f"🍳 For **{dish}**, I found these ingredients on Sahachari:\n"
                        f"  ✅ Available: {names}\n"
                    )
                    if missing_items:
                        response_text += f"  ❌ Not in stock: {', '.join(missing_items)}\n"
                    response_text += "\nShall I add the available ones to your cart? Say **'yes, add all'** or specify quantities."
                    current_user_session["pending_item"]   = {"item": "recipe_items", "quantity": 1, "unit": "piece"}
                    current_user_session["pending_intent"] = "recipe_confirm"
                    current_user_session["recipe_items"]   = available_items
                else:
                    response_text = (
                        f"🍳 For **{dish}**, you'd typically need: {', '.join(ingredient_list)}.\n\n"
                        "However, none seem to be in stock right now. "
                        "Try searching individually: 'do you have onions?'"
                    )
            else:
                response_text = (
                    f"🍳 I'd love to help you shop for **{dish}**!\n\n"
                    "Could you tell me which specific ingredients you need? "
                    "For example: 'I need 2 kg rice, 1 kg onions, and 500 g tomatoes for biryani.'\n\n"
                    "Or try: 'ingredients for biryani' / 'ingredients for curry'"
                )

        elif intent == "order":
            q_lower       = query.lower()

            # FIX: "now" at end of string triggers buy_now
            wants_checkout = bool(BUY_NOW_PATTERNS.search(query))

            if "wishlist" in query.lower():
                if current_user_session["wishlist"]:
                    response_text = await forward_to_nestjs_cart(
                        current_user_session["wishlist"], authorization, current_memory,
                    )
                    for w in current_user_session["wishlist"]:
                        memory_store.record_item_ordered(current_memory, w["item"])
                        items_just_added.append(w["item"])
                    current_user_session["wishlist"].clear()
                    cart_updated = True
                else:
                    response_text  = "Your wishlist is empty. Nothing to add!"
                    wants_checkout = False
            else:
                items = extract_grocery_items(query)

                service_match = SERVICE_QUERY_PATTERN.search(q_lower)
                if service_match:
                    service_name = normalise_service_name(service_match.group(0).strip())
                    if not any(service_name.lower() in i["item"].lower() for i in items):
                        items.append({"item": service_name, "quantity": 1, "unit": "piece"})

                if items:
                    for it in items:
                        if it["quantity"] is None and not is_service_item(it["item"]):
                            rem_qty, rem_unit = memory_store.get_preferred_quantity(
                                current_memory, it["item"]
                            )
                            if rem_qty:
                                it["quantity"] = rem_qty
                                it["unit"]     = rem_unit or "piece"
                                log.info(f"Auto-filled qty for {it['item']}: {rem_qty} {rem_unit}")

                    service_items = [i for i in items if is_service_item(i["item"]) or i["quantity"] is not None]
                    missing_qty   = [i for i in items if i["quantity"] is None and not is_service_item(i["item"])]

                    lines = []
                    if service_items:
                        res = await forward_to_nestjs_cart(service_items, authorization, current_memory)
                        lines.append(res)
                        cart_updated = True
                        for it in service_items:
                            memory_store.record_item_ordered(
                                current_memory, it["item"],
                                it.get("quantity", 1), it.get("unit", "piece"),
                            )
                            items_just_added.append(it["item"])
                        update_entity_tracker(current_user_session, [i["item"] for i in service_items])

                    # FIX: Queue ALL missing-qty items in slot_items, ask for first
                    if missing_qty:
                        current_user_session["slot_items"]     = missing_qty
                        pending = missing_qty[0]
                        current_user_session["pending_item"]   = pending
                        current_user_session["pending_intent"] = "order"
                        lines.append(
                            f"How much **{pending['item']}** would you like? "
                            f"(e.g. '1 kg', '500 g')"
                        )
                        wants_checkout = False

                    response_text = "\n\n".join(lines) if lines else "I couldn't figure out which items you'd like. Try: 'I want 2 kg onions'."
                else:
                    response_text  = "I couldn't figure out which items you'd like. Try: 'I want 2 kg onions'."
                    wants_checkout = False

            # FIX: "now" checkout — add items first, then trigger checkout
            if wants_checkout and cart_updated:
                checkout_msg, checkout_ok = await forward_nestjs_order_placement(authorization)
                if checkout_ok:
                    action = "checkout"
                    response_text += f"\n\n{checkout_msg}"
                    current_memory["last_order_summary"] = (
                        f"Checked out on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
                    )

            response_text = personalise_response(
                response_text, current_memory, current_user_session, "order", items_just_added,
            )

        else:  # rag
            is_faq_query  = bool(FAQ_PATTERNS.search(query))
            response_text = await get_rag_response(
                query,
                memory=current_memory,
                session=current_user_session,
                conversation_history=current_memory.get("conversation_history", []),
                is_faq=is_faq_query,
            )

        memory_store.add_turn(current_memory, query, response_text)
        await asyncio.gather(
            session_store.save(session_id, current_user_session),
            memory_store.save(session_id, current_memory),
        )
        return {"response": response_text, "cart_updated": cart_updated, "action": action}

    except Exception as e:
        log.exception("chat_endpoint failed")
        return {"response": f"System Error: {str(e)}", "cart_updated": False, "action": None}


if __name__ == "__main__":
    log.info("Starting Sahachari AI Service...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
