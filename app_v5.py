import streamlit as st
import pandas as pd
from pymongo import MongoClient
from st_aggrid import AgGrid, GridOptionsBuilder
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("MONGO_DB_NAME")
collection_name = os.getenv("MONGO_COLLECTION_NAME")

# MongoDB Connection
mongo_uri = st.secrets["MONGO"]["URI"]
client = MongoClient(mongo_uri)
client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True, serverSelectionTimeoutMS=5000)
db = client[db_name]
collection = db[collection_name]

# Page config
st.set_page_config(layout="wide")

# Get URL parameters
query_params = st.query_params
page = int(query_params.get("page", ["1"])[0])
scroll_target = query_params.get("scroll", [""])[0]

# Title
st.title('Price and Availability Survey Database')
st.markdown("""
##### Swathi Iyengar 
##### · Version 1.0 - 28 April 2025
###### · Built with Python, Semantic Scholar, MongoDB, and Streamlit
""", unsafe_allow_html=True)


# Add header with usage instructions
st.markdown("---")
st.markdown("""
**Instructions:**
- Use the sidebar filters to narrow down your search results
- Click any column header to sort the table or search within that column
- Click on a record to view more details about the paper
- Additional details will be displayed below the table
- Use the download button to export the filtered results as a CSV file
""")

# Sidebar filters
st.sidebar.header('Search Filters')
search_term = st.sidebar.text_input("Search Title or Abstract")

try:
    years = sorted([y for y in collection.distinct("year") if y is not None], reverse=True)
    year_filter = st.sidebar.multiselect("Select Year(s)", years)
    
    open_access_filter = st.sidebar.multiselect("Open Access", ["Yes", "No"])
    countries = sorted([c for c in collection.distinct("official_country_name") if c is not None])
    country_filter = st.sidebar.multiselect("Select Country/Countries", countries)
    
    regions = sorted([r for r in collection.distinct("WHO_region") if r is not None])
    region_filter = st.sidebar.multiselect("Select WHO Region(s)", regions)
except Exception as e:
    st.error(f"Error loading filters: {e}")
    st.stop()

# Column selection
available_columns = [
    "title", "authors", "year", "isOpenAccess",
    "official_country_name", "WHO_region", "WB_income_group"
]
default_columns = ["title", "authors", "year", "isOpenAccess"]

selected_columns = st.sidebar.multiselect(
    "Select columns to display in table:",
    available_columns,
    default=default_columns
)

# Build MongoDB query
query = {}
if search_term:
    query["$or"] = [
        {"title": {"$regex": search_term, "$options": "i"}},
        {"abstract": {"$regex": search_term, "$options": "i"}},
    ]
if year_filter:
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

# Query the database
papers_cursor = collection.find(query)
papers_df = pd.DataFrame(list(papers_cursor))

# Handle scroll after page change
if scroll_target == "grid":
    st.query_params.update({"page": str(page)})
    st.markdown("<script>window.scrollTo(0, 500);</script>", unsafe_allow_html=True)

# Reset page if sidebar search triggered
if any([search_term, year_filter, open_access_filter, country_filter, region_filter, selected_columns != default_columns]) and page != 1:
    st.query_params.update({"page": "1"})
    page = 1

# If papers exist
if not papers_df.empty:
    st.success(f"Found {papers_df.shape[0]} papers matching your criteria.")
    
    # Preprocessing
    def format_authors(authors_list):
        if not authors_list or not isinstance(authors_list, list):
            return ""
        if len(authors_list) == 1:
            return authors_list[0].get("name", "")
        authors = authors_list[0].get("name", "") if authors_list else ""
        return f"{authors} +{len(authors_list)-1}" if authors else ""

    if "authors" in papers_df.columns:
        papers_df["authors"] = papers_df["authors"].apply(format_authors)
        papers_df["full_authors"] = papers_df["authors"].apply(lambda x: ", ".join([a.get("name", "") for a in x]) if isinstance(x, list) else "")

    if "year" in papers_df.columns:
        papers_df["year"] = pd.to_numeric(papers_df["year"], errors="coerce").fillna(0).astype(int)

    def extract_openaccess_url(row):
        if row.get("isOpenAccess") and row.get("openAccessPdf") and row["openAccessPdf"].get("url"):
            return row["openAccessPdf"]["url"]
        return None

    if "openAccessPdf" in papers_df.columns:
        papers_df["open_access_link"] = papers_df.apply(extract_openaccess_url, axis=1)
    else:
        papers_df["open_access_link"] = None

    if "isOpenAccess" in papers_df.columns:
        papers_df["isOpenAccess"] = papers_df["isOpenAccess"].map({True: "Yes", False: "No"})


    # Pagination setup
    items_per_page = 300
    total_pages = max(1, (len(papers_df) + items_per_page - 1) // items_per_page)
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(papers_df))
    page_df = papers_df.iloc[start_idx:end_idx]

    # AgGrid setup
    display_df = page_df[selected_columns]
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_default_column(filter=True)  # Enable column search
    if "title" in selected_columns:
        gb.configure_column("title", wrapText=True, autoHeight=True, flex=4, minWidth=250)
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


    # Show selected row details
    selected_rows = ag_grid["selected_rows"]
    if selected_rows is not None and not selected_rows.empty:
        selected_row_info = selected_rows.iloc[0]

        selected_title = selected_row_info["title"]
        selected_year = selected_row_info["year"]

        selected_full_row = page_df[
            (page_df["title"] == selected_title) & (page_df["year"] == selected_year)
        ].iloc[0]

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

    # Download CSV
    # Define exactly the columns you want in the download
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
        "full_authors"
    ]

    # Check if each download column exists in papers_df
    available_download_columns = [col for col in download_columns if col in papers_df.columns]

    csv = papers_df[available_download_columns].to_csv(index=False)
    st.download_button("Download as CSV", csv, "papers_export.csv", "text/csv")

else:
    st.info("No papers found for the selected filters.")
