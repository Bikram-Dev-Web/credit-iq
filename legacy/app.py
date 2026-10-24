"""CreditIQ — AI Credit Risk & Lending Intelligence Platform.

Launch with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src import config, state, ui  # noqa: E402

st.set_page_config(
    page_title="CreditIQ — AI Credit Risk & Lending Intelligence",
    page_icon=str(ROOT / "assets" / "icon.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

config.ensure_dirs()
state.init_state()
ui.inject_css()

try:
    st.logo(str(ROOT / "assets" / "logo.svg"), icon_image=str(ROOT / "assets" / "icon.svg"), size="large")
except TypeError:  # older Streamlit without the size argument
    st.logo(str(ROOT / "assets" / "logo.svg"), icon_image=str(ROOT / "assets" / "icon.svg"))

PAGES = {
    "Overview": [
        st.Page("app_pages/1_Overview.py", title="Executive Overview", icon=":material/space_dashboard:", default=True),
    ],
    "Data": [
        st.Page("app_pages/2_Data_Center.py", title="Data Center", icon=":material/database:"),
        st.Page("app_pages/3_EDA.py", title="Exploratory Analysis", icon=":material/query_stats:"),
    ],
    "Model": [
        st.Page("app_pages/4_Model_Lab.py", title="AI Model Lab", icon=":material/model_training:"),
        st.Page("app_pages/5_Borrower_Simulator.py", title="Borrower Risk Simulator", icon=":material/person_search:"),
        st.Page("app_pages/8_Explainability_Fairness.py", title="Explainability & Fairness", icon=":material/balance:"),
    ],
    "Finance": [
        st.Page("app_pages/6_Portfolio_Intelligence.py", title="Portfolio Intelligence", icon=":material/pie_chart:"),
        st.Page("app_pages/7_Financial_Scenarios.py", title="Financial Scenario Lab", icon=":material/calculate:"),
    ],
    "Governance": [
        st.Page("app_pages/9_Model_Monitoring.py", title="Model Monitoring", icon=":material/monitor_heart:"),
        st.Page("app_pages/10_Reports_Documentation.py", title="Reports & Documentation", icon=":material/description:"),
    ],
}

page = st.navigation(PAGES)
ui.sidebar_status()
ui.synthetic_banner()
ui.show_flash()
page.run()
ui.footer()
