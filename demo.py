# demo.py
import pandas as pd
from numerical_engine import EcomData, analyze_store
from pprint import pprint

# --- Fake orders: note refund_flag carefully ---
orders = pd.DataFrame([
    # Normal hoodie & mug orders (NOT refunded)
    {"order_id": "1001", "order_date": "2025-11-01", "product_id": "sku-hoodie", "quantity": 1, "total_revenue": 40.0, "refund_flag": 0},
    {"order_id": "1002", "order_date": "2025-11-01", "product_id": "sku-mug",    "quantity": 2, "total_revenue": 30.0, "refund_flag": 0},
    {"order_id": "1003", "order_date": "2025-11-02", "product_id": "sku-hoodie", "quantity": 1, "total_revenue": 40.0, "refund_flag": 0},
    {"order_id": "1004", "order_date": "2025-11-03", "product_id": "sku-hoodie", "quantity": 2, "total_revenue": 80.0, "refund_flag": 0},

    # A few more normal hoodie & mug orders
    {"order_id": "1005", "order_date": "2025-11-04", "product_id": "sku-hoodie", "quantity": 1, "total_revenue": 40.0, "refund_flag": 0},
    {"order_id": "1006", "order_date": "2025-11-04", "product_id": "sku-mug",    "quantity": 1, "total_revenue": 15.0, "refund_flag": 0},

    # High-refund product: POSTER (these ARE refunded)
    {"order_id": "1007", "order_date": "2025-11-05", "product_id": "sku-poster", "quantity": 3, "total_revenue": 60.0, "refund_flag": 1},
    {"order_id": "1008", "order_date": "2025-11-06", "product_id": "sku-poster", "quantity": 2, "total_revenue": 40.0, "refund_flag": 1},
])

orders["order_date"] = pd.to_datetime(orders["order_date"])

# --- Fake products ---
products = pd.DataFrame([
    {"product_id": "sku-hoodie", "product_name": "Cloud Hoodie",   "cost_of_goods": 18.0, "price": 40.0},
    {"product_id": "sku-mug",    "product_name": "Galaxy Mug",     "cost_of_goods": 9.0,  "price": 15.0},
    {"product_id": "sku-poster", "product_name": "Starfield Poster","cost_of_goods": 8.0, "price": 20.0},
])

# --- Fake ads (one good, one bad, one low-CTR) ---
ads = pd.DataFrame([
    {"date": "2025-11-03", "platform": "meta",   "campaign_name": "Hoodie Launch",  "spend": 40.0, "impressions": 8000,  "clicks": 240, "conversions": 6, "revenue_attributed": 240.0},
    {"date": "2025-11-04", "platform": "google", "campaign_name": "Brand Search",   "spend": 15.0, "impressions": 2000,  "clicks": 80,  "conversions": 2, "revenue_attributed": 80.0},
    {"date": "2025-11-05", "platform": "meta",   "campaign_name": "Bad Promo",      "spend": 50.0, "impressions": 10000, "clicks": 300, "conversions": 1, "revenue_attributed": 40.0},
    {"date": "2025-11-06", "platform": "meta",   "campaign_name": "Awareness Only", "spend": 20.0, "impressions": 15000, "clicks": 50,  "conversions": 0, "revenue_attributed": 0.0},
])
ads["date"] = pd.to_datetime(ads["date"])

# --- Fake traffic (keep it simple for now) ---
traffic = pd.DataFrame([
    {"date": "2025-11-01", "channel": "meta_paid",   "sessions": 300, "purchases": 3},
    {"date": "2025-11-02", "channel": "direct",      "sessions": 150, "purchases": 2},
    {"date": "2025-11-03", "channel": "direct",      "sessions": 200, "purchases": 3},
    {"date": "2025-11-04", "channel": "direct",      "sessions": 180, "purchases": 2},
    {"date": "2025-11-05", "channel": "meta_paid",   "sessions": 400, "purchases": 3},
    {"date": "2025-11-06", "channel": "meta_paid",   "sessions": 420, "purchases": 2},
])
traffic["date"] = pd.to_datetime(traffic["date"])

# Build EcomData and analyze
data = EcomData.from_dataframes(orders, products, ads, traffic)
analysis = analyze_store(data)

print("\n=== Product KPIs (raw) ===")
for row in analysis["product_kpis"]:
    print(row)

print("\n=== Product insights ===")
pprint(analysis["insights"]["products"])

print("\n=== Campaign insights ===")
pprint(analysis["insights"]["campaigns"])

print("\n=== Daily insights ===")
pprint(analysis["insights"]["daily"])
