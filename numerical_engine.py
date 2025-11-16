# numerical_engine.py

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


# ---------- Data container ----------

@dataclass
class EcomData:
    orders: pd.DataFrame       # orders.csv
    products: pd.DataFrame     # products.csv
    ads: pd.DataFrame          # ads.csv
    traffic: pd.DataFrame      # traffic.csv

    @staticmethod
    def from_csv(
        orders_path: str,
        products_path: str,
        ads_path: str,
        traffic_path: str,
    ) -> "EcomData":
        orders = pd.read_csv(orders_path, parse_dates=["order_date"])
        products = pd.read_csv(products_path, parse_dates=["launch_date"], dayfirst=False, infer_datetime_format=True)
        ads = pd.read_csv(ads_path, parse_dates=["date"])
        traffic = pd.read_csv(traffic_path, parse_dates=["date"])
        return EcomData(orders=orders, products=products, ads=ads, traffic=traffic)


# ---------- KPI computations ----------

def compute_daily_kpis(data: EcomData) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per day and columns like:
    date, revenue, orders, aov, sessions, conversion_rate, ad_spend, roas, etc.
    """
    # Base daily revenue / orders
    orders = data.orders.copy()
    orders["date"] = orders["order_date"].dt.date

    daily_orders = (
        orders.groupby("date")
        .agg(
            revenue=("total_revenue", "sum"),
            orders=("order_id", "nunique"),
        )
        .reset_index()
    )

    # AOV
    daily_orders["aov"] = np.where(
        daily_orders["orders"] > 0,
        daily_orders["revenue"] / daily_orders["orders"],
        0.0,
    )

    # Traffic (sessions, purchases if provided)
    traffic = data.traffic.copy()
    traffic["date"] = traffic["date"].dt.date

    daily_traffic = (
        traffic.groupby("date")
        .agg(
            sessions=("sessions", "sum"),
            purchases=("purchases", "sum") if "purchases" in traffic.columns else ("sessions", "size"),
        )
        .reset_index()
    )

    # Join revenue + traffic
    daily = pd.merge(daily_orders, daily_traffic, on="date", how="outer").fillna(0)

    # Conversion rate (use purchases from traffic if present, else orders)
    daily["conversion_rate"] = np.where(
        daily["sessions"] > 0,
        daily["purchases"] / daily["sessions"],
        0.0,
    )

    # Ad spend and attributed revenue
    ads = data.ads.copy()
    ads["date"] = ads["date"].dt.date

    daily_ads = (
        ads.groupby("date")
        .agg(
            ad_spend=("spend", "sum"),
            ad_revenue=("revenue_attributed", "sum") if "revenue_attributed" in ads.columns else ("spend", "sum"),
        )
        .reset_index()
    )

    daily = pd.merge(daily, daily_ads, on="date", how="left").fillna(
        {"ad_spend": 0.0, "ad_revenue": 0.0}
    )

    daily["roas"] = np.where(
        daily["ad_spend"] > 0,
        daily["ad_revenue"] / daily["ad_spend"],
        np.nan,
    )

    # Sort and add moving averages + z-scores for anomaly detection
    daily = daily.sort_values("date").reset_index(drop=True)

    # 7-day moving averages
    for col in ["revenue", "conversion_rate", "roas"]:
        if col in daily.columns:
            daily[f"{col}_ma7"] = daily[col].rolling(window=7, min_periods=3).mean()
            daily[f"{col}_std7"] = daily[col].rolling(window=7, min_periods=3).std()
            daily[f"{col}_z"] = (daily[col] - daily[f"{col}_ma7"]) / daily[f"{col}_std7"]

    return daily


def compute_product_kpis(data: EcomData) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per product_id, including:
    units_sold, revenue, cogs, profit, margin, refund_rate (if available), etc.
    """
    orders = data.orders.copy()
    products = data.products.copy()

    # Aggregate per product
    product_agg = (
        orders.groupby("product_id")
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("total_revenue", "sum"),
            orders=("order_id", "nunique"),
            refunds=("refund_flag", "sum") if "refund_flag" in orders.columns else ("order_id", "size"),
        )
        .reset_index()
    )

    # Join with product metadata
    products_kpis = pd.merge(products, product_agg, on="product_id", how="left").fillna(
        {"units_sold": 0, "revenue": 0.0, "orders": 0, "refunds": 0}
    )

    # Profit and margin
    if "cost_of_goods" in products_kpis.columns:
        products_kpis["cogs_total"] = products_kpis["cost_of_goods"] * products_kpis["units_sold"]
        products_kpis["profit"] = products_kpis["revenue"] - products_kpis["cogs_total"]
        products_kpis["margin"] = np.where(
            products_kpis["revenue"] > 0,
            products_kpis["profit"] / products_kpis["revenue"],
            np.nan,
        )

    # Refund rate
    products_kpis["refund_rate"] = np.where(
        products_kpis["orders"] > 0,
        products_kpis["refunds"] / products_kpis["orders"],
        0.0,
    )

    return products_kpis


def compute_campaign_kpis(data: EcomData) -> pd.DataFrame:
    """
    Returns one row per campaign per platform with ROAS and efficiency metrics.
    """
    ads = data.ads.copy()

    group_cols = ["platform", "campaign_name"]
    campaign = (
        ads.groupby(group_cols)
        .agg(
            spend=("spend", "sum"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum") if "conversions" in ads.columns else ("clicks", "sum"),
            revenue_attributed=("revenue_attributed", "sum") if "revenue_attributed" in ads.columns else ("spend", "sum"),
        )
        .reset_index()
    )

    # Derived KPIs
    campaign["ctr"] = np.where(
        campaign["impressions"] > 0,
        campaign["clicks"] / campaign["impressions"],
        0.0,
    )
    campaign["cpc"] = np.where(
        campaign["clicks"] > 0,
        campaign["spend"] / campaign["clicks"],
        np.nan,
    )
    campaign["cpm"] = np.where(
        campaign["impressions"] > 0,
        campaign["spend"] * 1000 / campaign["impressions"],
        np.nan,
    )
    campaign["roas"] = np.where(
        campaign["spend"] > 0,
        campaign["revenue_attributed"] / campaign["spend"],
        np.nan,
    )

    return campaign


# ---------- Insight detection ----------

def detect_daily_insights(daily: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Look at daily KPIs + z-scores and emit "insight objects" like:
    - revenue anomaly
    - conversion drop
    - roas anomaly
    """
    insights: List[Dict[str, Any]] = []

    for _, row in daily.iterrows():
        date = row["date"]

        # Revenue anomaly
        if "revenue_z" in row and pd.notna(row["revenue_z"]):
            if row["revenue_z"] <= -2:
                insights.append(
                    {
                        "type": "revenue_drop",
                        "severity": "high",
                        "date": str(date),
                        "revenue": float(row["revenue"]),
                        "revenue_ma7": float(row["revenue_ma7"]),
                        "z_score": float(row["revenue_z"]),
                        "message": f"Revenue on {date} is significantly below the 7-day trend.",
                    }
                )
            elif row["revenue_z"] >= 2:
                insights.append(
                    {
                        "type": "revenue_spike",
                        "severity": "medium",
                        "date": str(date),
                        "revenue": float(row["revenue"]),
                        "revenue_ma7": float(row["revenue_ma7"]),
                        "z_score": float(row["revenue_z"]),
                        "message": f"Revenue on {date} is unusually high compared to recent days.",
                    }
                )

        # Conversion anomaly
        if "conversion_rate_z" in row and pd.notna(row["conversion_rate_z"]):
            if row["conversion_rate_z"] <= -2:
                insights.append(
                    {
                        "type": "conversion_drop",
                        "severity": "high",
                        "date": str(date),
                        "conversion_rate": float(row["conversion_rate"]),
                        "conversion_rate_ma7": float(row["conversion_rate_ma7"]),
                        "z_score": float(row["conversion_rate_z"]),
                        "message": f"Conversion rate on {date} is much lower than the recent average.",
                    }
                )

        # ROAS anomaly
        if "roas_z" in row and pd.notna(row["roas_z"]):
            if row["roas_z"] <= -2:
                insights.append(
                    {
                        "type": "roas_drop",
                        "severity": "medium",
                        "date": str(date),
                        "roas": float(row["roas"]),
                        "roas_ma7": float(row["roas_ma7"]),
                        "z_score": float(row["roas_z"]),
                        "message": f"ROAS on {date} is significantly lower than the recent average.",
                    }
                )

    return insights


def detect_product_insights(products_kpis: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Emit product-level insights like:
    - hero products
    - low-margin products
    - high refund rate
    """
    insights: List[Dict[str, Any]] = []

    # Hero products: high revenue & high margin
    if "revenue" in products_kpis.columns and "margin" in products_kpis.columns:
        revenue_q75 = products_kpis["revenue"].quantile(0.75)
        margin_q75 = products_kpis["margin"].quantile(0.75)

        hero_mask = (products_kpis["revenue"] >= revenue_q75) & (products_kpis["margin"] >= margin_q75)
        for _, row in products_kpis[hero_mask].iterrows():
            insights.append(
                {
                    "type": "hero_product",
                    "severity": "info",
                    "product_id": row["product_id"],
                    "product_name": row.get("product_name", ""),
                    "revenue": float(row["revenue"]),
                    "margin": float(row["margin"]),
                    "message": f"Product {row.get('product_name', row['product_id'])} is a hero product with high revenue and strong margin.",
                }
            )

    # Low margin products
    low_margin_mask = (products_kpis.get("margin") < 0.15) & (products_kpis["revenue"] > 0)
    for _, row in products_kpis[low_margin_mask].iterrows():
        insights.append(
            {
                "type": "low_margin_product",
                "severity": "medium",
                "product_id": row["product_id"],
                "product_name": row.get("product_name", ""),
                "revenue": float(row["revenue"]),
                "margin": float(row["margin"]),
                "message": f"Product {row.get('product_name', row['product_id'])} has low margin ({row['margin']:.2%}) despite generating revenue.",
            }
        )

    # High refund rate
    if "refund_rate" in products_kpis.columns:
        high_refund_mask = products_kpis["refund_rate"] >= 0.1
        for _, row in products_kpis[high_refund_mask].iterrows():
            insights.append(
                {
                    "type": "high_refund_product",
                    "severity": "high",
                    "product_id": row["product_id"],
                    "product_name": row.get("product_name", ""),
                    "refund_rate": float(row["refund_rate"]),
                    "orders": int(row["orders"]),
                    "message": f"Product {row.get('product_name', row['product_id'])} has a high refund rate ({row['refund_rate']:.1%}).",
                }
            )

    return insights


def detect_campaign_insights(campaigns: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Emit campaign-level insights:
    - low ROAS campaigns
    - high ROAS (scale)
    - poor CTR (creative issue)
    """
    insights: List[Dict[str, Any]] = []

    for _, row in campaigns.iterrows():
        platform = row["platform"]
        campaign_name = row["campaign_name"]
        roas = row["roas"]
        ctr = row["ctr"]
        spend = row["spend"]

        # Low ROAS with significant spend
        if pd.notna(roas) and spend > 10 and roas < 1.0:
            insights.append(
                {
                    "type": "low_roas_campaign",
                    "severity": "high",
                    "platform": platform,
                    "campaign_name": campaign_name,
                    "spend": float(spend),
                    "roas": float(roas),
                    "message": f"Campaign '{campaign_name}' on {platform} has low ROAS ({roas:.2f}) with meaningful spend (${spend:.2f}). Consider pausing or changing creative/targeting.",
                }
            )

        # Very good ROAS (scale)
        if pd.notna(roas) and spend >= 5 and roas >= 3.0:
            insights.append(
                {
                    "type": "high_roas_campaign",
                    "severity": "info",
                    "platform": platform,
                    "campaign_name": campaign_name,
                    "spend": float(spend),
                    "roas": float(roas),
                    "message": f"Campaign '{campaign_name}' on {platform} is performing very well (ROAS {roas:.2f}). Consider increasing budget.",
                }
            )

        # Low CTR (creative issue)
        if pd.notna(ctr) and row["impressions"] > 1000 and ctr < 0.01:
            insights.append(
                {
                    "type": "low_ctr_campaign",
                    "severity": "medium",
                    "platform": platform,
                    "campaign_name": campaign_name,
                    "ctr": float(ctr),
                    "message": f"Campaign '{campaign_name}' on {platform} has a low CTR ({ctr:.2%}). The creative or targeting may not be resonating.",
                }
            )

    return insights


# ---------- Top-level orchestrator ----------

def analyze_store(data: EcomData) -> Dict[str, Any]:
    """
    Run the full numerical analysis and return structured results that
    you can feed into an LLM or render directly.
    """
    daily = compute_daily_kpis(data)
    products_kpis = compute_product_kpis(data)
    campaigns = compute_campaign_kpis(data)

    daily_insights = detect_daily_insights(daily)
    product_insights = detect_product_insights(products_kpis)
    campaign_insights = detect_campaign_insights(campaigns)

    result = {
        "daily_kpis": daily.to_dict(orient="records"),
        "product_kpis": products_kpis.to_dict(orient="records"),
        "campaign_kpis": campaigns.to_dict(orient="records"),
        "insights": {
            "daily": daily_insights,
            "products": product_insights,
            "campaigns": campaign_insights,
        },
    }
    return result


# Example usage (you can delete this in production)
if __name__ == "__main__":
    data = EcomData.from_csv(
        "orders.csv",
        "products.csv",
        "ads.csv",
        "traffic.csv",
    )
    analysis = analyze_store(data)
    # For debugging:
    import json
    print(json.dumps(analysis["insights"], indent=2, default=str))
