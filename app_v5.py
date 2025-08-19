# import streamlit as st
# import pandas as pd
# from pymongo import MongoClient
# from st_aggrid import AgGrid, GridOptionsBuilder
# import os
# from dotenv import load_dotenv
# import json
# import numpy as np

# # Load environment variables
# load_dotenv()

# mongo_uri = os.getenv("MONGO_URI")
# db_name = os.getenv("MONGO_DB_NAME")
# collection_name = os.getenv("MONGO_COLLECTION_NAME")

# # MongoDB Connection
# mongo_uri = st.secrets["MONGO"]["URI"]
# db_name = st.secrets["MONGO"]["DB_NAME"]
# collection_name = st.secrets["MONGO"]["COLLECTION_NAME"]


# client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True, serverSelectionTimeoutMS=5000)
# db = client[db_name]
# collection = db[collection_name]

# # Page config
# st.set_page_config(layout="wide")

# # Get URL parameters
# query_params = st.query_params
# page = int(query_params.get("page", ["1"])[0])
# scroll_target = query_params.get("scroll", [""])[0]

# # Title
# st.title('Price and Availability Survey Database')
# st.markdown("""
# ##### Swathi Iyengar 
# ##### · Version 1.0 - 28 April 2025
# ###### · Built with Python, Semantic Scholar, MongoDB, and Streamlit
# """, unsafe_allow_html=True)


# # Add header with usage instructions
# st.markdown("---")
# st.markdown("""
# **Instructions:**
# - Use the sidebar filters to narrow down your search results
# - Add additional columns to the table by selecting them in the sidebar
# - Click any column header to sort the table or search within that column
# - Click on a record to view more details about the paper
# - Additional details will be displayed below the table
# - Use the download button to export the filtered results as a CSV file
# """)

# # Sidebar filters
# st.sidebar.header('Search Filters')
# search_term = st.sidebar.text_input("Search Title or Abstract")

# try:
#     years = sorted([y for y in collection.distinct("year") if y is not None], reverse=True)
#     year_filter = st.sidebar.multiselect("Select Year(s)", years)
    
#     open_access_filter = st.sidebar.multiselect("Open Access", ["Yes", "No"])
#     countries = sorted([c for c in collection.distinct("official_country_name") if c is not None])
#     country_filter = st.sidebar.multiselect("Select Country/Countries", countries)
    
#     regions = sorted([r for r in collection.distinct("WHO_region") if r is not None])
#     region_filter = st.sidebar.multiselect("Select WHO Region(s)", regions)
# except Exception as e:
#     st.error(f"Error loading filters: {e}")
#     st.stop()

# # Column selection
# available_columns = [
#     "title", "authors", "year", "isOpenAccess",
#     "official_country_name", "WHO_region", "WB_income_group"
# ]
# default_columns = ["title", "authors", "year", "isOpenAccess", "official_country_name", "WHO_region"]

# selected_columns = st.sidebar.multiselect(
#     "Select columns to display in table:",
#     available_columns,
#     default=default_columns
# )

# # Build MongoDB query
# query = {}
# if search_term:
#     query["$or"] = [
#         {"title": {"$regex": search_term, "$options": "i"}},
#         {"abstract": {"$regex": search_term, "$options": "i"}},
#     ]
# if year_filter:
#     query["year"] = {"$in": year_filter}
# if open_access_filter:
#     if "Yes" in open_access_filter and "No" not in open_access_filter:
#         query["isOpenAccess"] = True
#     elif "No" in open_access_filter and "Yes" not in open_access_filter:
#         query["isOpenAccess"] = False
# if country_filter:
#     query["official_country_name"] = {"$in": country_filter}
# if region_filter:
#     query["WHO_region"] = {"$in": region_filter}

# # Query the database
# papers_cursor = collection.find(query)
# papers_df = pd.DataFrame(list(papers_cursor))

# # Handle scroll after page change
# if scroll_target == "grid":
#     st.query_params.update({"page": str(page)})
#     st.markdown("<script>window.scrollTo(0, 500);</script>", unsafe_allow_html=True)

# # Reset page if sidebar search triggered
# if any([search_term, year_filter, open_access_filter, country_filter, region_filter, selected_columns != default_columns]) and page != 1:
#     st.query_params.update({"page": "1"})
#     page = 1

# # If papers exist
# if not papers_df.empty:
#     st.success(f"Found {papers_df.shape[0]} papers matching your criteria.")
    
#     # Preprocessing
#     def format_authors(authors_list):
#         if not authors_list or not isinstance(authors_list, list):
#             return ""
#         if len(authors_list) == 1:
#             return authors_list[0].get("name", "")
#         authors = authors_list[0].get("name", "") if authors_list else ""
#         return f"{authors} +{len(authors_list)-1}" if authors else ""

# def format_authors(authors_list):
#     if not authors_list or not isinstance(authors_list, list):
#         return ""
#     if len(authors_list) == 1:
#         return authors_list[0].get("name", "")
#     first = authors_list[0].get("name", "")
#     return f"{first} +{len(authors_list)-1}" if first else ""

# if "authors" in papers_df.columns:
#     # Build full list first (for details panel)
#     papers_df["full_authors"] = papers_df["authors"].apply(
#         lambda lst: ", ".join(a.get("name", "") for a in lst) if isinstance(lst, list) else ""
#     )
#     # Then collapse for the table
#     papers_df["authors"] = papers_df["authors"].apply(format_authors)

#     if "year" in papers_df.columns:
#         papers_df["year"] = pd.to_numeric(papers_df["year"], errors="coerce").fillna(0).astype(int)

#     def extract_openaccess_url(row):
#         if row.get("isOpenAccess") and row.get("openAccessPdf") and row["openAccessPdf"].get("url"):
#             return row["openAccessPdf"]["url"]
#         return None

#     if "openAccessPdf" in papers_df.columns:
#         papers_df["open_access_link"] = papers_df.apply(extract_openaccess_url, axis=1)
#     else:
#         papers_df["open_access_link"] = None

#     if "isOpenAccess" in papers_df.columns:
#         papers_df["isOpenAccess"] = papers_df["isOpenAccess"].map({True: "Yes", False: "No"})


#     # Pagination setup
#     items_per_page = 300
#     total_pages = max(1, (len(papers_df) + items_per_page - 1) // items_per_page)
#     start_idx = (page - 1) * items_per_page
#     end_idx = min(start_idx + items_per_page, len(papers_df))
#     page_df = papers_df.iloc[start_idx:end_idx]
    
#     # 🆕 Sort by year descending BEFORE showing
#     page_df = page_df.sort_values(by="year", ascending=False)

#     # AgGrid setup
#     display_df = page_df[selected_columns]
    
#     display_df = display_df.copy()

#     def to_display(x):
#         # Scalars stay as-is
#         if isinstance(x, (str, int, float, bool, np.number)) or x is None:
#             return x
#         # Simple lists -> "A, B, C"
#         if isinstance(x, list) and all(
#             (isinstance(i, (str, int, float, bool, np.number)) or i is None) for i in x
#         ):
#             return ", ".join(str(i) for i in x if i not in (None, ""))
#         # Dicts or nested lists -> JSON
#         if isinstance(x, (list, dict)):
#             try:
#                 return json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)
#             except Exception:
#                 return str(x)
#         # Fallback
#         return str(x)

#     for col in display_df.columns:
#         display_df[col] = display_df[col].map(to_display)

    

#     gb = GridOptionsBuilder.from_dataframe(display_df)
#     gb.configure_selection(selection_mode="single", use_checkbox=False)
#     gb.configure_default_column(filter=True)  # Enable column search
#     if "title" in selected_columns:
#         gb.configure_column("title", wrapText=True, autoHeight=True, flex=4, minWidth=250)
#     grid_options = gb.build()

#     ag_grid = AgGrid(
#         display_df,
#         gridOptions=grid_options,
#         update_mode="SELECTION_CHANGED",
#         data_return_mode="AS_INPUT",
#         fit_columns_on_grid_load=True,
#         theme="streamlit",
#         pagination=True,
#         paginationPageSize=items_per_page,
#     )


#     # Show selected row details
#     selected_rows = ag_grid["selected_rows"]
#     if selected_rows is not None and not selected_rows.empty:
#         selected_row_info = selected_rows.iloc[0]

#         selected_title = selected_row_info["title"]
#         selected_year = selected_row_info["year"]

#         selected_full_row = page_df[
#             (page_df["title"] == selected_title) & (page_df["year"] == selected_year)
#         ].iloc[0]

#         st.markdown("---")
#         st.subheader(f"Details for: {selected_full_row.get('title', '')}")

#         col1, col2 = st.columns(2)
#         if selected_full_row.get("url"):
#             col1.markdown(f"🔗 [Open Paper]({selected_full_row['url']})")
#         if selected_full_row.get("open_access_link"):
#             col2.markdown(f"📄 [Download PDF]({selected_full_row['open_access_link']})")

#         if selected_full_row.get("full_authors"):
#             st.markdown(f"**Authors:** {selected_full_row['full_authors']}")

#         col1, col2, col3 = st.columns(3)
#         if selected_full_row.get("official_country_name"):
#             col1.markdown(f"**Country:** {selected_full_row['official_country_name']}")
#         if selected_full_row.get("WHO_region"):
#             col2.markdown(f"**WHO Region:** {selected_full_row['WHO_region']}")
#         if selected_full_row.get("WB_income_group"):
#             col3.markdown(f"**Income Group:** {selected_full_row['WB_income_group']}")

#         if selected_full_row.get("abstract"):
#             st.markdown("**Abstract:**")
#             st.markdown(selected_full_row["abstract"])

#     # Download CSV
#     # Define exactly the columns you want in the download
#     download_columns = [
#         "externalIds", 
#         "corpusID",
#         "url",
#         "title",
#         "abstract",
#         "year",
#         "isOpenAccess",
#         "open_access_link",
#         "official_country_name",
#         "WHO_region",
#         "WB_income_group",
#         "full_authors"
#     ]

#     # Check if each download column exists in papers_df
#     available_download_columns = [col for col in download_columns if col in papers_df.columns]

#     csv = papers_df[available_download_columns].to_csv(index=False)
#     st.download_button("Download as CSV", csv, "papers_export.csv", "text/csv")

# else:
#     st.info("No papers found for the selected filters.")

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from st_aggrid import AgGrid, GridOptionsBuilder
from dotenv import load_dotenv

# -----------------------------
# Setup & Mongo connection
# -----------------------------
st.set_page_config(layout="wide")
load_dotenv()

# Secrets first, then fall back to env (robust)
try:
    mongo_uri = st.secrets["MONGO"]["URI"]
    db_name = st.secrets["MONGO"]["DB_NAME"]
    collection_name = st.secrets["MONGO"]["COLLECTION_NAME"]
except Exception:
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    collection_name = os.getenv("MONGO_COLLECTION_NAME")

if not all([mongo_uri, db_name, collection_name]):
    st.error("MongoDB connection details are missing. Check st.secrets or environment variables.")
    st.stop()

try:
    client = MongoClient(
        mongo_uri,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=5000,
    )
    db = client[db_name]
    collection = db[collection_name]
    # Trigger connection early
    client.admin.command("ping")
except Exception as e:
    st.error(f"Error connecting to MongoDB: {e}")
    st.stop()

# -----------------------------
# Header
# -----------------------------
st.title("Price and Availability Survey Database")
st.markdown(
    """
##### Swathi Iyengar 
##### · Version 1.0 - 28 April 2025
###### · Built with Python, Semantic Scholar, MongoDB, and Streamlit
""",
    unsafe_allow_html=True,
)

st.markdown("---")
st.markdown(
    """
**Instructions:**
- Use the sidebar filters to narrow down your search results
- Add additional columns to the table by selecting them in the sidebar
- Click any column header to sort the table or search within that column
- Click on a record to view more details about the paper
- Additional details will be displayed below the table
- Use the download button to export the filtered results as a CSV file
"""
)

# -----------------------------
# URL params (robust)
# -----------------------------
page_str = st.query_params.get("page", "1")
try:
    page = int(page_str)
    if page < 1:
        page = 1
except Exception:
    page = 1

scroll_target = st.query_params.get("scroll", "")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Search Filters")
search_term = st.sidebar.text_input("Search Title or Abstract")

try:
    years_raw = [y for y in collection.distinct("year") if y is not None]
    # normalize years -> ints if possible
    def _as_int(y):
        try:
            return int(y)
        except Exception:
            return y
    years = sorted({_as_int(y) for y in years_raw if y != ""}, reverse=True)

    open_access_filter = st.sidebar.multiselect("Open Access", ["Yes", "No"])

    countries = sorted([c for c in collection.distinct("official_country_name") if c is not None])
    region_vals = sorted([r for r in collection.distinct("WHO_region") if r is not None])

    country_filter = st.sidebar.multiselect("Select Country/Countries", countries)
    region_filter = st.sidebar.multiselect("Select WHO Region(s)", region_vals)
except Exception as e:
    st.error(f"Error loading filters: {e}")
    st.stop()

# -----------------------------
# Column selection
# -----------------------------
available_columns = [
    "title",
    "authors",
    "year",
    "isOpenAccess",
    "official_country_name",
    "WHO_region",
    "WB_income_group",
]
default_columns = [
    "title",
    "authors",
    "year",
    "isOpenAccess",
    "official_country_name",
    "WHO_region",
]
selected_columns = st.sidebar.multiselect(
    "Select columns to display in table:", available_columns, default=default_columns
)

# -----------------------------
# Build MongoDB query
# -----------------------------
query = {}
if search_term:
    query["$or"] = [
        {"title": {"$regex": search_term, "$options": "i"}},
        {"abstract": {"$regex": search_term, "$options": "i"}},
    ]

if year_filter := st.sidebar.multiselect("Select Year(s)", years):
    query["year"] = {"$in": year_filter}

if open_access_filter:
    if "Yes" in open_access_filter and "No" not in open_access_filter:
        query["isOpenAccess"] = True
    elif "No" in open_access_filter and "Yes" not in open_access_filter:
        query["isOpenAccess"] = False

if country_filter:
    query["official_country_name"] = {"$in": country_filter}

if region_filter:
    query["WHO_region"] = {"$in": region_filter}

# -----------------------------
# Fetch & DataFrame
# -----------------------------
try:
    papers_cursor = collection.find(query)
    papers_df = pd.DataFrame(list(papers_cursor))
except Exception as e:
    st.error(f"Error querying MongoDB: {e}")
    st.stop()

# Scroll handling
if scroll_target == "grid":
    st.query_params["page"] = str(page)
    st.markdown("<script>window.scrollTo(0, 500);</script>", unsafe_allow_html=True)

# Reset page if filters changed
if any(
    [
        bool(search_term),
        bool(year_filter),
        bool(open_access_filter),
        bool(country_filter),
        bool(region_filter),
        selected_columns != default_columns,
    ]
) and page != 1:
    st.query_params["page"] = "1"
    page = 1

# -----------------------------
# Helpers
# -----------------------------
def format_authors(authors_list):
    if not authors_list or not isinstance(authors_list, list):
        return ""
    if len(authors_list) == 1:
        return authors_list[0].get("name", "")
    first = authors_list[0].get("name", "")
    return f"{first} +{len(authors_list)-1}" if first else ""

def extract_openaccess_url(row):
    try:
        if row.get("isOpenAccess") and row.get("openAccessPdf") and row["openAccessPdf"].get("url"):
            return row["openAccessPdf"]["url"]
    except Exception:
        pass
    return None

def to_display(x):
    # Scalars stay as-is
    if isinstance(x, (str, int, float, bool, np.number)) or x is None:
        return x
    # Simple lists -> "A, B, C"
    if isinstance(x, list) and all(
        (isinstance(i, (str, int, float, bool, np.number)) or i is None) for i in x
    ):
        return ", ".join(str(i) for i in x if i not in (None, ""))
    # Dicts or nested lists -> JSON
    if isinstance(x, (list, dict)):
        try:
            return json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return str(x)
    # Fallback
    return str(x)

# -----------------------------
# Main table
# -----------------------------
if papers_df.empty:
    st.info("No papers found for the selected filters.")
else:
    st.success(f"Found {papers_df.shape[0]} papers matching your criteria.")

    # Normalize types / derived cols
    if "year" in papers_df.columns:
        papers_df["year"] = pd.to_numeric(papers_df["year"], errors="coerce").fillna(0).astype(int)

    if "authors" in papers_df.columns:
        # Full list (details)
        papers_df["full_authors"] = papers_df["authors"].apply(
            lambda lst: ", ".join(a.get("name", "") for a in lst) if isinstance(lst, list) else ""
        )
        # Compact for grid
        papers_df["authors"] = papers_df["authors"].apply(format_authors)

    if "openAccessPdf" in papers_df.columns:
        papers_df["open_access_link"] = papers_df.apply(extract_openaccess_url, axis=1)
    else:
        papers_df["open_access_link"] = None

    if "isOpenAccess" in papers_df.columns:
        papers_df["isOpenAccess"] = papers_df["isOpenAccess"].map({True: "Yes", False: "No"})

    # Create a stable id for selection lookup
    if "_id" in papers_df.columns:
        papers_df["doc_id"] = papers_df["_id"].astype(str)
    else:
        papers_df["doc_id"] = papers_df.index.astype(str)

    # Pagination
    items_per_page = 300
    total_pages = max(1, (len(papers_df) + items_per_page - 1) // items_per_page)
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(papers_df))
    page_df = papers_df.iloc[start_idx:end_idx].copy()

    # Sort by year desc for display
    if "year" in page_df.columns:
        page_df = page_df.sort_values(by="year", ascending=False)

    # Build display df
    cols_present = [c for c in selected_columns if c in page_df.columns]
    # Always include hidden doc_id to map back selection
    display_cols = cols_present + (["doc_id"] if "doc_id" in page_df.columns else [])
    display_df = page_df[display_cols].copy()

    # Make hash-safe & pretty for lists
    for col in display_df.columns:
        display_df[col] = display_df[col].map(to_display)

    # Grid
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_default_column(filter=True)
    if "title" in display_df.columns:
        gb.configure_column("title", wrapText=True, autoHeight=True, flex=4, minWidth=250)
    if "doc_id" in display_df.columns:
        gb.configure_column("doc_id", hide=True)

    grid_options = gb.build()

    ag_grid = AgGrid(
        display_df,
        gridOptions=grid_options,
        update_mode="SELECTION_CHANGED",
        data_return_mode="AS_INPUT",
        fit_columns_on_grid_load=True,
        theme="streamlit",
        pagination=True,
        paginationPageSize=items_per_page,
    )

    # Selection details
    selected_rows = ag_grid.get("selected_rows", [])
    if isinstance(selected_rows, list) and len(selected_rows) > 0:
        sel = selected_rows[0]
        sel_doc_id = sel.get("doc_id")

        if sel_doc_id and "doc_id" in page_df.columns:
            match = page_df[page_df["doc_id"] == sel_doc_id]
        else:
            # Fallback: match on title + year
            selected_title = sel.get("title")
            selected_year = sel.get("year")
            match = page_df[
                (page_df.get("title") == selected_title) & (page_df.get("year") == selected_year)
            ]

        if not match.empty:
            selected_full_row = match.iloc[0]

            st.markdown("---")
            st.subheader(f"Details for: {selected_full_row.get('title', '')}")

            col1, col2 = st.columns(2)
            if selected_full_row.get("url"):
                col1.markdown(f"🔗 [Open Paper]({selected_full_row['url']})")
            if selected_full_row.get("open_access_link"):
                col2.markdown(f"📄 [Download PDF]({selected_full_row['open_access_link']})")

            if selected_full_row.get("full_authors"):
                st.markdown(f"**Authors:** {selected_full_row['full_authors']}")

            col1, col2, col3 = st.columns(3)
            if selected_full_row.get("official_country_name"):
                col1.markdown(f"**Country:** {selected_full_row['official_country_name']}")
            if selected_full_row.get("WHO_region"):
                col2.markdown(f"**WHO Region:** {selected_full_row['WHO_region']}")
            if selected_full_row.get("WB_income_group"):
                col3.markdown(f"**Income Group:** {selected_full_row['WB_income_group']}")

            if selected_full_row.get("abstract"):
                st.markdown("**Abstract:**")
                st.markdown(selected_full_row["abstract"])

    # -----------------------------
    # CSV export (filtered dataset)
    # -----------------------------
    download_columns = [
        "externalIds",
        "corpusID",
        "url",
        "title",
        "abstract",
        "year",
        "isOpenAccess",
        "open_access_link",
        "official_country_name",
        "WHO_region",
        "WB_income_group",
        "full_authors",
    ]
    available_download_columns = [c for c in download_columns if c in papers_df.columns]
    csv = papers_df[available_download_columns].to_csv(index=False)
    st.download_button("Download as CSV", csv, "papers_export.csv", "text/csv")
