import shutil
from datetime import date, timedelta
from email.message import Message
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

url = "https://acleddata.com/curated-data-files"

tmp_path = Path("/tmp")
data_path = Path("data")
data_path.mkdir(exist_ok=True)


@st.cache_resource(ttl=timedelta(minutes=10))
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


@st.cache_data(ttl=timedelta(days=1))
def load_datasets() -> list[dict]:
    response = start_session().get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    return [
        {
            "name": download_box_content.find("h1").get_text(strip=True),
            "url": download_box_content.find("a")["href"],
        }
        for download_box_content in soup.find_all(
            "div", attrs={"class": "download-box-content"}
        )
    ]


def human_file_size(file: Path, suffix="B"):
    # From: https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    size = file.stat().st_size
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(size) < 1024.0:
            return f"{size:3.1f}{unit}{suffix}"
        size /= 1024.0
    return f"{size:.1f}Yi{suffix}"


def download_dataset(dataset):
    name = dataset["name"]

    r = start_session().get(dataset["url"], stream=True)
    r.raise_for_status()

    m = Message()
    for key, value in r.headers.items():
        m[key] = value
    filename = m.get_param(
        "filename", failobj=[f"{name}.xlsx"], header="content-disposition"
    )[-1]

    data_file = data_path / filename

    if not data_file.exists():
        LOGGER.info(f"Downloading dataset to file: {filename} ({dataset})")

        temp_file = tmp_path / filename
        with temp_file.open("wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        shutil.move(temp_file, data_path / filename)

        LOGGER.info(
            f"Downloaded dataset to file: {filename} ({human_file_size(data_file)})"
        )

    return filename


def convert_to_hdf5(filename: str):
    """Convert dataset file to HDF5 for faster load in Pandas"""
    h5_filename = str(Path(filename).with_suffix(".h5"))
    data_file = data_path / h5_filename

    if not data_file.exists():
        LOGGER.info(f"Converting file to HDF5: {filename}...")

        temp_file = tmp_path / h5_filename
        LOGGER.info(f"Reading file: {filename}...")
        df = pd.read_excel(data_path / filename)

        LOGGER.info(f"Writing file: {h5_filename}...")
        df.to_hdf(temp_file, key="df", mode="w")

        shutil.move(temp_file, data_file)

        LOGGER.info(
            f"Converted file to HDF5: {h5_filename} ({human_file_size(data_file)})"
        )

    return h5_filename


@st.cache_data
def load_dataset(dataset, start_year, end_year) -> pd.DataFrame:
    dataset_file = download_dataset(dataset)
    h5_file = convert_to_hdf5(dataset_file)
    df = pd.read_hdf(data_path / h5_file)
    return df[df.YEAR.between(start_year, end_year)]


def main():
    st.set_page_config(
        page_title="ACLED",
        page_icon="⚔️",
    )

    st.title("ACLED")

    start_date, end_date = st.slider(
        "Date range",
        value=(date.today() - relativedelta(years=1), date.today()),
        min_value=date(2020, 1, 1),
        max_value=date.today(),
    )

    datasets = load_datasets()

    selected_datasets = st.multiselect(
        "Datasets", datasets, format_func=lambda d: d["name"]
    )

    if selected_datasets:

        df = pd.concat(
            map(
                lambda d: load_dataset(d, start_date.year, end_date.year),
                selected_datasets,
            ),
            keys=[d["name"] for d in selected_datasets],
        )

        df = df[df.EVENT_DATE.dt.date.between(start_date, end_date + timedelta(days=1))]

        item_count = st.slider("Items", 10, len(df), value=100)

        st.header("Recent")
        st.dataframe(df.sort_values("EVENT_DATE", ascending=False).head(item_count))

        st.header("Most fatal")
        st.dataframe(df.sort_values("FATALITIES", ascending=False).head(item_count))

        fatalities_by_date_and_country = (
            df.groupby(["EVENT_DATE", "COUNTRY"])["FATALITIES"]
            .sum()
            .reset_index()
            .pivot(index="EVENT_DATE", columns="COUNTRY")
            .FATALITIES.fillna(0)
        )

        st.header("Fatalities by date and country")
        st.plotly_chart(px.line(fatalities_by_date_and_country))

        st.header("Cumulative fatalities by date and country")
        st.plotly_chart(px.line(fatalities_by_date_and_country.cumsum()))


if __name__ == "__main__":
    main()
