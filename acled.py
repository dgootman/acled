import colorsys
from datetime import date, timedelta
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta


@st.cache_resource(ttl=86400)
def access_token() -> str:
    r = requests.post(
        "https://acleddata.com/oauth/token",
        data={
            "username": st.secrets["acled_email"],
            "password": st.secrets["acled_password"],
            "grant_type": "password",
            "client_id": "acled",
        },
    )
    r.raise_for_status()
    return r.json()["access_token"]


@st.cache_data(ttl=86400)
def load_page(url: str) -> str:
    r = requests.get(url, headers={"Authorization": "Bearer " + access_token()})
    r.raise_for_status()
    return r.text


def load_soup(url: str) -> BeautifulSoup:
    return BeautifulSoup(load_page(url), features="html.parser")


@st.cache_data(ttl=86400)
def load_excel(url: str) -> pd.DataFrame:
    df = pd.read_excel(
        url, storage_options={"Authorization": "Bearer " + access_token()}
    )
    df.columns = [c.lower() for c in df.columns]

    return df


@st.cache_data(ttl=86400)
def load_dataset() -> pd.DataFrame:
    data_files_page = load_soup(
        "https://acleddata.com/conflict-data/download-data-files"
    )

    df = pd.concat(
        load_excel(
            urljoin(
                "https://acleddata.com",
                load_soup(urljoin("https://acleddata.com", a["href"])).select_one(
                    "a.o-button--file"
                )["href"],
            )
        )
        for a in data_files_page.select('a[href^="/aggregated/"]')
        if a.get_text(strip=True).startswith("Aggregated data on ")
    )

    return df


def main():
    st.set_page_config(
        page_title="ACLED Dashboard",
        page_icon="https://acleddata.com/acleddatanew/wp-content/uploads/2019/10/acled-favi-high-res.png",
    )

    st.title("ACLED | Armed Conflict Location & Events Data")

    df = load_dataset()
    df["event_date"] = df["week"]

    with st.form("date_range"):
        start_date, end_date = st.slider(
            "##### Date range",
            value=(date.today() - relativedelta(years=1), date.today()),
            min_value=df["event_date"].min().date(),
            max_value=date.today(),
        )

        st.form_submit_button()

    df = df[df["event_date"].dt.date.between(start_date, end_date + timedelta(days=1))]

    with st.expander("Filters"):
        filter_columns = [
            ("region", "Regions"),
            ("country", "Countries"),
            ("event_type", "Event Types"),
            ("sub_event_type", "Sub-event Types"),
            # ("source_scale", "Source Type"),
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

        st.map(
            df.sort_values("event_date"),
            color="color",
            size="size",
            latitude="centroid_latitude",
            longitude="centroid_longitude",
        )

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


if __name__ == "__main__":
    main()
