import colorsys
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

tmp_path = Path("/tmp")
data_path = Path("static")
data_path.mkdir(exist_ok=True)


def human_file_size(file: Path, suffix="B"):
    # From: https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    size = file.stat().st_size
    for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
        if abs(size) < 1024.0:
            return f"{size:3.1f}{unit}{suffix}"
        size /= 1024.0
    return f"{size:.1f}Y{suffix}"


def download_dataset(year):
    filename = f"{year}.csv"

    data_file = data_path / filename

    if not data_file.exists():
        r = requests.get(
            "https://api.acleddata.com/acled/read.exportcsv",
            params={
                "email": st.secrets["acled_email"],
                "key": st.secrets["acled_key"],
                "year": year,
                "limit": 0,
            },
            stream=True,
        )

        LOGGER.info(
            f"Download dataset response: {dict(status_code=r.status_code, headers=r.headers)}"
        )

        r.raise_for_status()

        LOGGER.info(f"Downloading dataset to file: {filename}")

        temp_file = tmp_path / filename
        with temp_file.open("wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        shutil.move(temp_file, data_path / filename)

        LOGGER.info(
            f"Downloaded dataset to file: {filename} ({human_file_size(data_file)})"
        )

    return filename


def store_dataset(raw_file: str):
    """Convert dataset file to Parquet for faster load in Pandas"""
    stored_file = str(Path(raw_file).with_suffix(".parquet"))
    stored_file_path = data_path / stored_file

    if not stored_file_path.exists():
        LOGGER.info(f"Converting file to Parquet: {raw_file}...")

        temp_file = tmp_path / stored_file
        LOGGER.info(f"Reading file: {raw_file}...")
        df = pd.read_csv(data_path / raw_file, parse_dates=["event_date"])

        LOGGER.info(f"Writing file: {stored_file}...")
        df.to_parquet(temp_file)

        shutil.move(temp_file, stored_file_path)

        LOGGER.info(
            f"Converted file to Parquet: {stored_file} ({human_file_size(stored_file_path)})"
        )

    return stored_file


@st.cache_data
def load_dataset(year) -> pd.DataFrame:
    raw_file = download_dataset(year)
    stored_file = store_dataset(raw_file)
    df = pd.read_parquet(data_path / stored_file)
    return df


def main():
    st.set_page_config(
        page_title="ACLED Dashboard",
        page_icon="https://acleddata.com/acleddatanew/wp-content/uploads/2019/10/acled-favi-high-res.png",
    )

    st.title("ACLED | Armed Conflict Location & Events Data")

    with st.form("date_range"):
        start_date, end_date = st.slider(
            "##### Date range",
            value=(date.today() - relativedelta(years=1), date.today()),
            min_value=date(2020, 1, 1),
            max_value=date.today(),
        )

        st.form_submit_button()

    df = pd.concat(map(load_dataset, range(start_date.year, end_date.year + 1)))

    df = df[df.event_date.dt.date.between(start_date, end_date + timedelta(days=1))]

    with st.expander("Filters"):
        filter_columns = [
            ("region", "Regions"),
            ("country", "Countries"),
            ("event_type", "Event Types"),
            ("sub_event_type", "Sub-event Types"),
            ("source_scale", "Source Type"),
        ]

        st_columns = st.columns(2)

        for index, (column, name) in enumerate(filter_columns):
            values = st_columns[index % 2].multiselect(
                name, options=sorted(df[column].unique())
            )
            if values:
                df = df[df[column].isin(values)]

        fatalities = st.checkbox("Only events with fatalities")
        if fatalities:
            df = df[df.fatalities > 0]

    item_count = st.slider("Items", 10, len(df), value=100)

    def render_map(df: pd.DataFrame):
        df["date_rank"] = df["event_date"].rank(pct=True, method="dense")
        df["fatality_rank"] = np.log10(np.clip(df["fatalities"], 1, 500)) / np.log10(
            500
        )
        df["color"] = df.apply(
            lambda row: tuple(
                int(255 * x)
                for x in colorsys.hsv_to_rgb(
                    (
                        # green for no fatalities
                        120 / 360
                        if row["fatalities"] == 0
                        # yellow/red gradient based on number of fatalities
                        else 60 / 360 * (1 - row["fatality_rank"])
                    ),
                    1.0,
                    1.0,
                )
            )
            # opacity based on date rank
            + ((0.5 + 0.5 * row["date_rank"]),),
            axis=1,
        )
        df["size"] = df["fatalities"].rank(method="dense") * 10

        st.map(df.sort_values("event_date"), color="color", size="size")

    st.header("Recent")

    recent = df.sort_values("event_date", ascending=False).head(item_count)
    st.dataframe(recent)
    render_map(recent)

    st.header("Most fatal")
    most_fatal = df.sort_values("fatalities", ascending=False).head(item_count)
    st.dataframe(most_fatal)
    render_map(most_fatal)

    if not df.empty:
        fatalities_by_date_and_country = (
            df.groupby(["event_date", "country"])["fatalities"]
            .sum()
            .reset_index()
            .pivot(index="event_date", columns="country")
            .fatalities.fillna(0)
        )

        st.header("Fatalities by date and country")
        st.plotly_chart(px.line(fatalities_by_date_and_country))

        st.header("Cumulative fatalities by date and country")
        st.plotly_chart(px.line(fatalities_by_date_and_country.cumsum()))

    with st.expander("Cached files"):
        for file in sorted(Path("static").iterdir()):
            col1, col2, col3 = st.columns([1, 2, 10])
            with col1:
                if st.button("🗑️", key=str(file)):
                    file.unlink()
                    st.rerun()
            with col2:
                st.write(human_file_size(file))
            with col3:
                st.markdown(f"[{file.name}](app/{file})")


if __name__ == "__main__":
    main()
