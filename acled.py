import requests
import streamlit as st
from bs4 import BeautifulSoup
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

url = "https://acleddata.com/curated-data-files"


@st.cache_resource(ttl="10 min")
def start_session() -> requests.Session:
    session = requests.Session()

    response = session.get(url)

    if response.ok:
        return session

    from seleniumbase import SB

    def verify_success(sb):
        for _ in range(8):
            if "Curated Data" in sb.get_page_title():
                break
            sb.sleep(1)
        sb.assert_title_contains("Curated Data")

    with SB(uc_cdp=True, headless=True, guest_mode=True) as sb:
        sb.open(url)
        if "Curated Data" not in sb.get_page_title():
            if sb.is_element_visible('input[value*="Verify"]'):
                sb.click('input[value*="Verify"]')
            elif sb.is_element_visible('iframe[title*="challenge"]'):
                sb.switch_to_frame('iframe[title*="challenge"]')
                sb.click("span.mark", timeout=15)
            else:
                raise Exception("Detected!")
            try:
                verify_success(sb)
            except Exception:
                raise Exception("Detected!")

        session.headers.update({"User-Agent": sb.get_user_agent()})
        for cookie in sb.driver.get_cookies():
            session.cookies.set(cookie["name"], cookie["value"])

    response = session.get(url)
    response.raise_for_status()
    return session

def main():
    st.set_page_config(
        page_title="ACLED",
        page_icon="⚔️",
    )

    st.title("ACLED")

    session = start_session()

    response = session.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    datasets = [
        {
            "name": download_box_content.find("h1").get_text(strip=True),
            "url": download_box_content.find("a")["href"],
        }
        for download_box_content in soup.find_all(
            "div", attrs={"class": "download-box-content"}
        )
    ]

    st.header("Datasets")
    st.dataframe(datasets)


if __name__ == "__main__":
    main()
