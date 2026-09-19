"""Paths, seeds and tunable constants shared by the whole pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EVENTS_PATH = DATA_DIR / "events.csv.gz"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = ROOT / "sample_submission.csv"
SUBMISSION_PATH = ROOT / "submission.csv"
RESULTS_PATH = ROOT / "results.jsonl"
CACHE_DIR = ROOT / "artifacts"

SEED = 42
N_FOLDS = 5
N_SEEDS = 5
TARGET_RECALL = 0.70

# A gap longer than this starts a new browsing session.
SESSION_GAP_S = 1800

# Population comparisons are made inside a fixed span so that the training pools and
# the scored batch cover the same number of days; the test split is exactly one span.
POOL_SPAN_DAYS = 7

# Inter-event gaps below these thresholds are counted as "machine fast".
FAST_GAP_THRESHOLDS_S = (2, 5, 10, 30)

# Search pages above this are treated as deep pagination sweeps.
DEEP_PAGE = 5

# Number of most frequent event bigrams kept as explicit features.
TOP_BIGRAMS = 25

# Components kept from the TF-IDF of event n-grams.
SVD_COMPONENTS = 16

EVENT_NAMES = (
    "search_results_view",
    "item_view",
    "photo_swipe",
    "seller_page_view",
    "contact_phone_show",
    "contact_chat_open",
    "contact_message_sent",
    "favorite_add",
    "login",
    "captcha_shown",
)

ENGAGEMENT_EVENTS = ("photo_swipe", "favorite_add", "login")
CONTACT_EVENTS = ("contact_phone_show", "contact_chat_open", "contact_message_sent")

SCRIPT_CLIENTS = (
    "curl",
    "Scrapy",
    "node-fetch",
    "python-requests",
    "python-urllib3",
    "Go-http-client",
)

PLATFORMS = ("web", "desktop", "android", "ios")
