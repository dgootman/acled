import csv
import shutil
from datetime import date
from email.message import Message
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from bs4 import BeautifulSoup
from streamlit.logger import get_logger
from xlsx2csv import Xlsx2csv

LOGGER = get_logger(__name__)

url = "https://acleddata.com/curated-data-files"

tmp_path = Path("/tmp")
data_path = Path("data")
data_path.mkdir(exist_ok=True)


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


@st.cache_data(ttl="1 day")
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

    if not (data_path / filename).exists():
        LOGGER.info(f"Downloading dataset to file {filename}: {dataset}")

        temp_file = tmp_path / filename
        with temp_file.open("wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        shutil.move(temp_file, data_path / filename)

        LOGGER.info(f"Downloaded dataset to file {filename}: {dataset}")

    return filename


def convert_to_csv(filename: str):
    """Convert dataset file to CSV for faster load in Pandas"""
    csv_filename = str(Path(filename).with_suffix(".csv"))

    if not (data_path / csv_filename).exists():
        LOGGER.info(f"Converting file to CSV: {filename}")

        temp_file = tmp_path / csv_filename
        Xlsx2csv(data_path / filename, dateformat="%Y-%m-%dT%H:%M:%S").convert(
            str(temp_file)
        )
        shutil.move(temp_file, data_path / csv_filename)

        LOGGER.info(f"Converted file to CSV: {filename}")

    return csv_filename


class FilteringCsvReader:
    """CSV file reader with a condition function to load only applicable rows"""

    def __init__(self, filename: str, condition: Callable[[dict], bool]) -> None:
        self.file = open(filename, "rt")
        self.condition = condition
        self.iterator = self.__iter__()

    def read(self, count=-1):
        return next(self.iterator, "")

    def __iter__(self):
        line = next(self.file)
        headers = next(csv.reader([line]))
        yield line

        try:
            while (line := next(self.file)) != None:
                data = dict(zip(headers, next(csv.reader([line]))))
                while not self.condition(data):
                    line = next(self.file)
                    data = dict(zip(headers, next(csv.reader([line]))))
                yield line
        except StopIteration:
            return ""


@st.cache_data
def load_dataset(dataset, start_year, end_year):
    dataset_file = download_dataset(dataset)
    csv_file = convert_to_csv(dataset_file)
    return pd.read_csv(
        FilteringCsvReader(
            data_path / csv_file,
            lambda d: start_year <= int(d["YEAR"]) <= end_year,
        )
    )


def main():
    st.set_page_config(
        page_title="ACLED",
        page_icon="⚔️",
    )

    st.title("ACLED")

    datasets = load_datasets()

    start_year, end_year = st.select_slider(
        "Date range",
        options=range(2000, date.today().year + 1),
        value=(date.today().year - 1, date.today().year),
    )

    selected_datasets = st.multiselect(
        "Datasets", datasets, format_func=lambda d: d["name"]
    )

    if selected_datasets:

        df = pd.concat(
            map(
                lambda d: load_dataset(d, start_year, end_year),
                selected_datasets,
            ),
            keys=[d["name"] for d in selected_datasets],
        )

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
