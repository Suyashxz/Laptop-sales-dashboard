import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================
# Page config
# =============================
st.set_page_config(
    page_title="Laptop Sales Dashboard",
    page_icon="💻",
    layout="wide",
)

# =============================
# Light-blue theme (CSS)
# =============================
st.markdown(
    """
    <style>
      /* App background */
      .stApp {
        background: linear-gradient(180deg, #f4fbff 0%, #ffffff 55%, #f7fbff 100%);
      }

      /* Sidebar */
      section[data-testid="stSidebar"] {
        background: #eef8ff;
        border-right: 1px solid #d8efff;
      }

      /* Titles */
      h1, h2, h3, h4, h5, h6 { color: #0b1f33; }
      .subtle { color: #274a63; opacity: 0.9; }

      /* KPI cards */
      .kpi {
        background: #ffffff;
        border: 1px solid #d8efff;
        border-radius: 16px;
        padding: 14px 14px;
        box-shadow: 0 1px 10px rgba(10, 30, 60, 0.06);
      }
      .kpi .label { font-size: 12px; color: #355a73; letter-spacing: 0.2px; }
      .kpi .value { font-size: 26px; font-weight: 750; margin-top: 2px; color: #0b1f33; }
      .kpi .delta { font-size: 12px; color: #1f6fb2; margin-top: 4px; }

      /* Section cards */
      .card {
        background: #ffffff;
        border: 1px solid #d8efff;
        border-radius: 16px;
        padding: 16px 16px 10px 16px;
        box-shadow: 0 1px 12px rgba(10, 30, 60, 0.06);
      }

      /* Small helper text */
      .hint { font-size: 12px; color: #355a73; opacity: 0.9; }
      .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #e6f5ff;
        border: 1px solid #cfeeff;
        color: #1f6fb2;
        font-size: 12px;
        margin-right: 6px;
        margin-top: 6px;
      }

      /* Dataframe rounding */
      div[data-testid="stDataFrame"] {
        border: 1px solid #d8efff;
        border-radius: 16px;
        overflow: hidden;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Helpers
# =============================
def parse_money(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_gb(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    # Interpret TB as 1024 GB
    if "TB" in s:
        val *= 1024
    return val

def parse_inches(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan

def parse_ghz(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan

def normalize_brand(b):
    if pd.isna(b):
        return "Unknown"
    s = str(b).strip()
    if not s:
        return "Unknown"
    # Keep Apple/HP/MSI as common casing; title-case others
    mapping = {
        "DELL": "Dell",
        "dell": "Dell",
        "acer": "Acer",
        "LENOVO": "Lenovo",
        "ASUS": "ASUS",
        "HP": "HP",
        "MSI": "MSI",
        "LG": "LG",
        "APPLE": "Apple",
        "Apple": "Apple",
    }
    if s in mapping:
        return mapping[s]
    return s.title()

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardize core text fields
    for col in ["brand", "model", "screen_size", "color", "harddisk", "cpu", "ram", "OS",
                "special_features", "graphics", "graphics_coprocessor", "cpu_speed"]:
        if col in df.columns:
            df[col] = df[col].astype("object")

    df["brand_norm"] = df["brand"].apply(normalize_brand)
    df["model"] = df["model"].fillna("Unknown model").astype(str).str.strip()
    df["OS"] = df["OS"].fillna("Unknown").astype(str).str.strip()
    df["cpu"] = df["cpu"].fillna("Unknown").astype(str).str.strip()
    df["graphics"] = df["graphics"].fillna("Unknown").astype(str).str.strip()
    df["color"] = df["color"].fillna("Unknown").astype(str).str.strip()

    # Numeric parsing
    df["price_usd"] = df["Price"].apply(parse_money)
    df["total_sales_usd"] = df["Total Sales"].apply(parse_money)
    df["ram_gb"] = df["ram"].apply(parse_gb)
    df["disk_gb"] = df["harddisk"].apply(parse_gb)
    df["screen_in"] = df["screen_size"].apply(parse_inches)
    df["cpu_ghz"] = df["cpu_speed"].apply(parse_ghz)
    df["units_sold"] = pd.to_numeric(df["Sale Product Count"], errors="coerce")
    df["stock"] = pd.to_numeric(df["Available Stock"], errors="coerce")
    df["rating"] = pd.to_numeric(df.get("rating", np.nan), errors="coerce")

    # Derived metrics (safe)
    df["revenue_per_unit"] = np.where(df["units_sold"] > 0, df["total_sales_usd"] / df["units_sold"], np.nan)
    df["stock_value_usd"] = df["stock"] * df["price_usd"]
    df["sales_to_stock"] = np.where(df["stock"] > 0, df["units_sold"] / df["stock"], np.nan)

    return df

def fmt_money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"${x:,.2f}"

def fmt_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{int(round(x)):,}"

def kpi(label, value, delta_text=None):
    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          {'<div class="delta">'+delta_text+'</div>' if delta_text else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

def recommendation_engine(d: pd.DataFrame) -> dict:
    """
    Generate practical actions:
      - Reorder: high sales, low stock
      - Discount: high stock, low sales
      - Premium push: high rating & high price but low sales (possible awareness gap)
      - Quality watch: low rating but high sales (risk of returns/brand damage)
    """
    out = {}

    dd = d.copy()
    dd = dd.dropna(subset=["total_sales_usd", "price_usd", "units_sold", "stock"], how="any")
    if dd.empty:
        return {"note": "Not enough valid numeric rows after filtering to generate recommendations."}

    # Robust thresholds (percentiles)
    sales_p75 = dd["total_sales_usd"].quantile(0.75)
    sales_p25 = dd["total_sales_usd"].quantile(0.25)
    stock_p25 = dd["stock"].quantile(0.25)
    stock_p75 = dd["stock"].quantile(0.75)
    price_p75 = dd["price_usd"].quantile(0.75)

    # Reorder list
    reorder = dd[(dd["total_sales_usd"] >= sales_p75) & (dd["stock"] <= stock_p25)].copy()
    reorder["priority_score"] = (reorder["total_sales_usd"] / (reorder["stock"] + 1))
    reorder = reorder.sort_values("priority_score", ascending=False).head(10)

    # Discount list
    discount = dd[(dd["total_sales_usd"] <= sales_p25) & (dd["stock"] >= stock_p75)].copy()
    discount["overstock_score"] = (discount["stock"] / (discount["total_sales_usd"] + 1))
    discount = discount.sort_values("overstock_score", ascending=False).head(10)

    # Premium push: expensive, highly rated, but not selling strongly
    premium = dd.copy()
    if premium["rating"].notna().any():
        rating_p75 = premium["rating"].quantile(0.75)
        premium = premium[(premium["price_usd"] >= price_p75) & (premium["rating"] >= rating_p75)]
        premium = premium[premium["total_sales_usd"] <= sales_p25].copy()
        premium["opp_score"] = (premium["rating"] * premium["price_usd"]) / (premium["total_sales_usd"] + 1)
        premium = premium.sort_values("opp_score", ascending=False).head(10)
    else:
        premium = premium.iloc[0:0]

    # Quality watch: low rating but high sales
    quality = dd.copy()
    if quality["rating"].notna().any():
        rating_p25 = quality["rating"].quantile(0.25)
        quality = quality[(quality["rating"] <= rating_p25) & (quality["total_sales_usd"] >= sales_p75)].copy()
        quality["risk_score"] = (quality["total_sales_usd"] / (quality["rating"] + 0.1))
        quality = quality.sort_values("risk_score", ascending=False).head(10)
    else:
        quality = quality.iloc[0:0]

    out["reorder"] = reorder
    out["discount"] = discount
    out["premium"] = premium
    out["quality"] = quality
    out["thresholds"] = {
        "sales_p75": sales_p75,
        "sales_p25": sales_p25,
        "stock_p25": stock_p25,
        "stock_p75": stock_p75,
        "price_p75": price_p75,
    }
    return out

# =============================
# Load data
# =============================
DATA_PATH = "660b2599-ba9c-4921-b4d0-e90e8ebd9440.csv"  # put CSV in same folder as app.py
df = load_data(DATA_PATH)

# =============================
# Sidebar filters
# =============================
st.sidebar.markdown("## Filters")
st.sidebar.caption("Use slicers to drill into brands, specs, and performance.")

# Categorical filters
brand_opts = sorted(df["brand_norm"].dropna().unique().tolist())
os_opts = sorted(df["OS"].dropna().unique().tolist())
cpu_opts = sorted(df["cpu"].dropna().unique().tolist())
gfx_opts = sorted(df["graphics"].dropna().unique().tolist())
color_opts = sorted(df["color"].dropna().unique().tolist())

brand_sel = st.sidebar.multiselect("Brand", brand_opts, default=brand_opts[:10] if len(brand_opts) > 10 else brand_opts)
os_sel = st.sidebar.multiselect("Operating System", os_opts, default=os_opts)
cpu_sel = st.sidebar.multiselect("CPU", cpu_opts, default=cpu_opts[:12] if len(cpu_opts) > 12 else cpu_opts)
gfx_sel = st.sidebar.multiselect("Graphics", gfx_opts, default=gfx_opts)
color_sel = st.sidebar.multiselect("Color", color_opts, default=color_opts)

# Numeric filters
price_min, price_max = float(np.nanmin(df["price_usd"])), float(np.nanmax(df["price_usd"]))
sales_min, sales_max = float(np.nanmin(df["total_sales_usd"])), float(np.nanmax(df["total_sales_usd"]))
rating_min, rating_max = float(np.nanmin(df["rating"])) if df["rating"].notna().any() else 0.0, float(np.nanmax(df["rating"])) if df["rating"].notna().any() else 5.0

price_range = st.sidebar.slider("Price (USD)", min_value=float(price_min), max_value=float(price_max),
                                value=(float(price_min), float(price_max)))
sales_range = st.sidebar.slider("Total Sales (USD)", min_value=float(sales_min), max_value=float(sales_max),
                                value=(float(sales_min), float(sales_max)))
rating_range = st.sidebar.slider("Rating", min_value=0.0, max_value=5.0,
                                 value=(float(rating_min), float(rating_max)))

st.sidebar.markdown("---")
top_n = st.sidebar.slider("Top N for rankings", 5, 25, 10)
show_table = st.sidebar.checkbox("Show filtered table", value=False)

# Apply filters
f = df.copy()
f = f[f["brand_norm"].isin(brand_sel)]
f = f[f["OS"].isin(os_sel)]
f = f[f["cpu"].isin(cpu_sel)]
f = f[f["graphics"].isin(gfx_sel)]
f = f[f["color"].isin(color_sel)]
f = f[(f["price_usd"].between(price_range[0], price_range[1], inclusive="both")) | (f["price_usd"].isna())]
f = f[(f["total_sales_usd"].between(sales_range[0], sales_range[1], inclusive="both")) | (f["total_sales_usd"].isna())]
f = f[(f["rating"].between(rating_range[0], rating_range[1], inclusive="both")) | (f["rating"].isna())]

# =============================
# Header
# =============================
st.markdown("# Laptop Sales Dashboard")
st.markdown(
    "<div class='subtle'>Interactive analysis of laptop listings, pricing, inventory, and sales performance. "
    "Built for quick executive insights and operational actions.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<span class='pill'>Rows: {len(f):,}</span>"
    f"<span class='pill'>Brands: {f['brand_norm'].nunique():,}</span>"
    f"<span class='pill'>Models: {f['model'].nunique():,}</span>",
    unsafe_allow_html=True,
)

# =============================
# KPIs
# =============================
c1, c2, c3, c4, c5 = st.columns(5)

total_revenue = float(f["total_sales_usd"].sum(skipna=True))
units = float(f["units_sold"].sum(skipna=True))
avg_price = float(f["price_usd"].mean(skipna=True))
avg_rating = float(f["rating"].mean(skipna=True)) if f["rating"].notna().any() else np.nan
stock_units = float(f["stock"].sum(skipna=True))

with c1:
    kpi("Total Revenue", fmt_money(total_revenue))
with c2:
    kpi("Units Sold (sum)", fmt_int(units))
with c3:
    kpi("Avg Price", fmt_money(avg_price))
with c4:
    kpi("Avg Rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "—")
with c5:
    kpi("Stock Units (sum)", fmt_int(stock_units))

st.markdown("")

# =============================
# Tabs
# =============================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🧩 Mix & Segments", "📦 Inventory Actions", "🧾 Conclusions"])

# ---------- Tab 1: Performance ----------
with tab1:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Top Brands by Revenue")
        top_brand = (
            f.groupby("brand_norm", as_index=False)["total_sales_usd"].sum()
            .sort_values("total_sales_usd", ascending=False)
            .head(top_n)
        )
        fig = px.bar(
            top_brand,
            x="total_sales_usd",
            y="brand_norm",
            orientation="h",
            labels={"total_sales_usd": "Revenue (USD)", "brand_norm": "Brand"},
            template="plotly_white",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Revenue vs Price (Bubble = Units Sold)")
        scat = f.dropna(subset=["price_usd", "total_sales_usd", "units_sold"], how="any").copy()
        if scat.empty:
            st.info("Not enough numeric rows after filters to render scatter.")
        else:
            fig2 = px.scatter(
                scat,
                x="price_usd",
                y="total_sales_usd",
                size="units_sold",
                color="brand_norm",
                hover_data=["model", "OS", "ram", "harddisk", "rating", "stock"],
                labels={"price_usd": "Price (USD)", "total_sales_usd": "Total Sales (USD)", "brand_norm": "Brand"},
                template="plotly_white",
            )
            fig2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="Brand")
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    cA, cB = st.columns(2, gap="large")

    with cA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Top Models by Revenue")
        top_model = (
            f.groupby(["brand_norm", "model"], as_index=False)["total_sales_usd"].sum()
            .sort_values("total_sales_usd", ascending=False)
            .head(top_n)
        )
        top_model["label"] = top_model["brand_norm"] + " • " + top_model["model"]
        fig3 = px.bar(
            top_model.sort_values("total_sales_usd"),
            x="total_sales_usd",
            y="label",
            orientation="h",
            labels={"total_sales_usd": "Revenue (USD)", "label": "Model"},
            template="plotly_white",
        )
        fig3.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Price Distribution")
        hist = f.dropna(subset=["price_usd"]).copy()
        if hist.empty:
            st.info("No price values after filters.")
        else:
            fig4 = px.histogram(
                hist,
                x="price_usd",
                nbins=30,
                labels={"price_usd": "Price (USD)"},
                template="plotly_white",
            )
            fig4.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Tab 2: Mix & Segments ----------
with tab2:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Revenue Mix by OS")
        os_mix = (
            f.groupby("OS", as_index=False)["total_sales_usd"].sum()
            .sort_values("total_sales_usd", ascending=False)
            .head(15)
        )
        fig = px.pie(
            os_mix,
            values="total_sales_usd",
            names="OS",
            hole=0.45,
            template="plotly_white",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("CPU Mix (Top 15)")
        cpu_mix = (
            f.groupby("cpu", as_index=False)["total_sales_usd"].sum()
            .sort_values("total_sales_usd", ascending=False)
            .head(15)
        )
        fig = px.bar(
            cpu_mix.sort_values("total_sales_usd"),
            x="total_sales_usd",
            y="cpu",
            orientation="h",
            labels={"total_sales_usd": "Revenue (USD)", "cpu": "CPU"},
            template="plotly_white",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Screen Size vs Revenue (Median)")
        ss = f.dropna(subset=["screen_in", "total_sales_usd"]).copy()
        if ss.empty:
            st.info("No screen size values after filters.")
        else:
            ss_agg = ss.groupby("screen_in", as_index=False).agg(
                median_sales=("total_sales_usd", "median"),
                listings=("screen_in", "size"),
            )
            fig = px.line(
                ss_agg.sort_values("screen_in"),
                x="screen_in",
                y="median_sales",
                markers=True,
                hover_data=["listings"],
                labels={"screen_in": "Screen Size (in)", "median_sales": "Median Sales (USD)"},
                template="plotly_white",
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Rating vs Revenue")
        rr = f.dropna(subset=["rating", "total_sales_usd"]).copy()
        if rr.empty:
            st.info("No ratings available after filters.")
        else:
            fig = px.box(
                rr,
                x=pd.cut(rr["rating"], bins=[0, 3, 4, 4.5, 5.01], include_lowest=True).astype(str),
                y="total_sales_usd",
                labels={"x": "Rating Band", "total_sales_usd": "Total Sales (USD)"},
                template="plotly_white",
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Tab 3: Inventory Actions ----------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Actionable Recommendations")
    rec = recommendation_engine(f)

    if "note" in rec:
        st.warning(rec["note"])
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        t = rec["thresholds"]
        st.markdown(
            f"<div class='hint'>Thresholds (robust percentiles on filtered data): "
            f"Sales P75={fmt_money(t['sales_p75'])}, Sales P25={fmt_money(t['sales_p25'])}, "
            f"Stock P25={fmt_int(t['stock_p25'])}, Stock P75={fmt_int(t['stock_p75'])}, "
            f"Price P75={fmt_money(t['price_p75'])}.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        sub1, sub2 = st.columns(2, gap="large")

        with sub1:
            st.markdown("#### ✅ Reorder / Restock (High Sales, Low Stock)")
            rr = rec["reorder"].copy()
            if rr.empty:
                st.info("No items match the reorder rule within the current filters.")
            else:
                show_cols = ["brand_norm", "model", "price_usd", "total_sales_usd", "units_sold", "stock", "rating"]
                st.dataframe(
                    rr[show_cols].rename(columns={
                        "brand_norm": "Brand", "model": "Model", "price_usd": "Price",
                        "total_sales_usd": "Revenue", "units_sold": "Units Sold", "stock": "Stock", "rating": "Rating"
                    }).assign(
                        Price=lambda d: d["Price"].map(fmt_money),
                        Revenue=lambda d: d["Revenue"].map(fmt_money),
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Operational action: increase replenishment or prioritize supplier lead times for these SKUs.")

        with sub2:
            st.markdown("#### 🟦 Discount / Bundle (High Stock, Low Sales)")
            dd = rec["discount"].copy()
            if dd.empty:
                st.info("No items match the discount rule within the current filters.")
            else:
                show_cols = ["brand_norm", "model", "price_usd", "total_sales_usd", "units_sold", "stock", "rating"]
                st.dataframe(
                    dd[show_cols].rename(columns={
                        "brand_norm": "Brand", "model": "Model", "price_usd": "Price",
                        "total_sales_usd": "Revenue", "units_sold": "Units Sold", "stock": "Stock", "rating": "Rating"
                    }).assign(
                        Price=lambda d: d["Price"].map(fmt_money),
                        Revenue=lambda d: d["Revenue"].map(fmt_money),
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Operational action: consider targeted discounts, bundles, or repositioning for these SKUs.")

        st.markdown("")
        sub3, sub4 = st.columns(2, gap="large")

        with sub3:
            st.markdown("#### ⭐ Premium Push (Expensive + Highly Rated, But Under-selling)")
            pp = rec["premium"].copy()
            if pp.empty:
                st.info("No items match the premium push rule (or ratings are missing).")
            else:
                show_cols = ["brand_norm", "model", "price_usd", "total_sales_usd", "units_sold", "stock", "rating"]
                st.dataframe(
                    pp[show_cols].rename(columns={
                        "brand_norm": "Brand", "model": "Model", "price_usd": "Price",
                        "total_sales_usd": "Revenue", "units_sold": "Units Sold", "stock": "Stock", "rating": "Rating"
                    }).assign(
                        Price=lambda d: d["Price"].map(fmt_money),
                        Revenue=lambda d: d["Revenue"].map(fmt_money),
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Marketing action: improve visibility (ads/SEO/listing content), not necessarily discounting.")

        with sub4:
            st.markdown("#### ⚠️ Quality Watch (Low Rated, Yet High Sales)")
            qw = rec["quality"].copy()
            if qw.empty:
                st.info("No items match the quality watch rule (or ratings are missing).")
            else:
                show_cols = ["brand_norm", "model", "price_usd", "total_sales_usd", "units_sold", "stock", "rating"]
                st.dataframe(
                    qw[show_cols].rename(columns={
                        "brand_norm": "Brand", "model": "Model", "price_usd": "Price",
                        "total_sales_usd": "Revenue", "units_sold": "Units Sold", "stock": "Stock", "rating": "Rating"
                    }).assign(
                        Price=lambda d: d["Price"].map(fmt_money),
                        Revenue=lambda d: d["Revenue"].map(fmt_money),
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Risk action: verify supplier quality, review return rates, and improve after-sales support.")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Tab 4: Conclusions ----------
with tab4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Executive Summary (Auto-generated from current filters)")

    # Key insights
    top_brand_name = (
        f.groupby("brand_norm")["total_sales_usd"].sum().sort_values(ascending=False).head(1).index.tolist()
    )
    top_brand_name = top_brand_name[0] if top_brand_name else "—"

    best_os = (
        f.groupby("OS")["total_sales_usd"].sum().sort_values(ascending=False).head(1).index.tolist()
    )
    best_os = best_os[0] if best_os else "—"

    asp = f["price_usd"].mean(skipna=True)
    sell_through = f["sales_to_stock"].median(skipna=True)

    st.markdown(
        f"""
        **What stands out**
        - The current selection is led by **{top_brand_name}** in revenue contribution.
        - The strongest OS segment by revenue is **{best_os}**.
        - Average listed price (ASP) is **{fmt_money(asp)}**.
        - Median *sales-to-stock* (units_sold / stock) is **{sell_through:.2f}** (higher suggests faster moving stock).

        **Recommendations**
        - Restock high-demand SKUs (see *Inventory Actions → Reorder*) to avoid lost sales.
        - Reduce overstock exposure using targeted discounts/bundles for slow movers.
        - Push premium, highly rated but under-selling models via better listing visibility and positioning.
        - Investigate low-rated high sellers to prevent brand damage and future revenue erosion.

        **Notes**
        - This dataset appears to be a snapshot (no date/time column). Trend-over-time metrics are not possible unless you add a date field.
        """.strip()
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Optional: Table + Download
# =============================
if show_table:
    st.markdown("### Filtered Records")
    show_cols = [
        "brand_norm", "model", "price_usd", "total_sales_usd", "units_sold", "stock",
        "rating", "OS", "cpu", "ram", "harddisk", "screen_size", "graphics", "color"
    ]
    out = f[show_cols].copy()
    out = out.rename(columns={
        "brand_norm": "Brand",
        "model": "Model",
        "price_usd": "Price (USD)",
        "total_sales_usd": "Total Sales (USD)",
        "units_sold": "Sale Product Count",
        "stock": "Available Stock",
        "rating": "Rating",
    })
    st.dataframe(out, use_container_width=True, hide_index=True)

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_laptop_sales.csv", mime="text/csv")
