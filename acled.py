import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    st.set_page_config(
        page_title="ACLED",
        page_icon="⚔️",
    )

    st.title("ACLED")


if __name__ == "__main__":
    main()
