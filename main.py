import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ---------------------------------------
# 기본 설정
# ---------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 바탕으로 회귀선을 만들어 "
    "연도별 예상 평균기온을 확인합니다."
)


# ---------------------------------------
# 데이터 불러오기
# ---------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 필요한 데이터가 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"])

    # 연도 만들기
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# ---------------------------------------
# 연도별 평균기온 계산
# ---------------------------------------
yearly = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# ---------------------------------------
# 분석에 사용할 연도 선택
# 조건:
# 1. 2025년 이하
# 2. 관측일 300일 이상
# ---------------------------------------
analysis = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()


analysis = analysis.sort_values("연도").reset_index(drop=True)


# ---------------------------------------
# 회귀분석
# 독립변수 = 1908년부터 지난 연수
# ---------------------------------------
analysis["지난연수"] = analysis["연도"] - 1908

x = analysis["지난연수"].to_numpy()
y = analysis["평균기온"].to_numpy()


# 선형회귀
slope, intercept = np.polyfit(x, y, 1)

analysis["회귀기온"] = intercept + slope * analysis["지난연수"]


# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# ---------------------------------------
# 기본 정보
# ---------------------------------------
st.subheader("📊 회귀 분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("회귀에 사용한 연도 수", f"{len(analysis)}년")

with col2:
    st.metric("시작 연도", f"{analysis['연도'].min()}년")

with col3:
    st.metric("끝 연도", f"{analysis['연도'].max()}년")

with col4:
    st.metric("상관계수", f"{correlation:.3f}")


st.caption(
    "분석 조건: 2025년 이하이며, 해당 연도의 관측일수가 300일 이상인 연도만 사용했습니다."
)


# ---------------------------------------
# 산점도 + 회귀선
# ---------------------------------------
st.subheader("📈 서울 연평균기온과 연도의 관계")


fig = go.Figure()


# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=analysis["연도"],
        y=analysis["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        text=analysis["연도"].astype(str) + "년",
        customdata=np.column_stack(
            (analysis["관측일수"],)
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        ),
        marker=dict(size=7)
    )
)


# 회귀선
fig.add_trace(
    go.Scatter(
        x=analysis["연도"],
        y=analysis["회귀기온"],
        mode="lines",
        name="회귀선",
        hovertemplate=(
            "%{x}년<br>"
            "회귀 예상기온: %{y:.2f}℃"
            "<extra></extra>"
        ),
        line=dict(width=3)
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    hovermode="x unified",
    height=550,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------
# 회귀식 표시
# ---------------------------------------
st.subheader("🧮 회귀식")

st.write(
    f"**예상 기온 = {slope:.4f} × (연도 - 1908) + {intercept:.4f}**"
)

st.write(
    f"상관계수: **{correlation:.4f}**"
)

if slope > 0:
    st.info(
        f"회귀선의 기울기는 1년에 약 {slope:.4f}℃입니다. "
        "회귀선 기준으로 시간이 지날수록 연평균기온이 높아지는 방향입니다."
    )
else:
    st.info(
        f"회귀선의 기울기는 1년에 약 {slope:.4f}℃입니다. "
        "회귀선 기준으로 시간이 지날수록 연평균기온이 낮아지는 방향입니다."
    )


# ---------------------------------------
# 연도 선택 → 예상 기온
# ---------------------------------------
st.subheader("🔮 연도를 선택해 예상 기온 확인하기")

selected_year = st.slider(
    "예상 기온을 확인할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 선택한 연도의 회귀 예상값
selected_x = selected_year - 1908
predicted_temp = intercept + slope * selected_x


st.markdown(
    f"""
    <div style="
        background-color: #f5f7fa;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    ">
        <div style="font-size: 22px; color: #555;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="
            font-size: 56px;
            font-weight: bold;
            margin-top: 10px;
        ">
            {predicted_temp:.2f}℃
        </div>
        <div style="font-size: 15px; color: #777;">
            과거 관측자료로 만든 선형 회귀선 기준
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------
# 선택한 연도를 그래프에도 표시
# ---------------------------------------
prediction_years = np.arange(1900, 2101)
prediction_x = prediction_years - 1908
prediction_temps = intercept + slope * prediction_x

fig_prediction = go.Figure()

# 회귀선 전체 구간
fig_prediction.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temps,
        mode="lines",
        name="회귀선",
        line=dict(width=3)
    )
)

# 실제 관측자료
fig_prediction.add_trace(
    go.Scatter(
        x=analysis["연도"],
        y=analysis["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=6),
        hovertemplate=(
            "%{x}년<br>"
            "실제 평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 선택한 연도
fig_prediction.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상",
        marker=dict(
            size=16,
            symbol="star"
        ),
        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            "예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig_prediction.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        range=[1900, 2100],
        tickmode="linear",
        dtick=20
    ),
    height=550,
    hovermode="x unified"
)

st.plotly_chart(fig_prediction, use_container_width=True)


# ---------------------------------------
# 참고: 실제 데이터
# ---------------------------------------
with st.expander("📋 회귀에 사용된 연도별 데이터 보기"):
    display_df = analysis.copy()

    display_df["평균기온"] = display_df["평균기온"].round(2)
    display_df["회귀기온"] = display_df["회귀기온"].round(2)

    display_df = display_df[
        ["연도", "관측일수", "평균기온", "지난연수", "회귀기온"]
    ]

    display_df.columns = [
        "연도",
        "관측일수",
        "실제 평균기온(℃)",
        "1908년부터 지난 연수",
        "회귀 예상기온(℃)"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ---------------------------------------
# 기본 설정
# ---------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균기온 변화를 살펴보고, "
    "과거 자료를 이용해 앞으로의 예상 기온을 확인합니다."
)


# ---------------------------------------
# 데이터 불러오기
# ---------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# ---------------------------------------
# 연도별 평균기온과 관측일수
# ---------------------------------------
yearly = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# ---------------------------------------
# 분석 대상
# 2025년 이하
# 관측일수 300일 이상
# ---------------------------------------
analysis = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

analysis = analysis.sort_values("연도").reset_index(drop=True)


# ---------------------------------------
# 전체 기간 회귀
# 독립변수 = 1908년부터 지난 연수
# ---------------------------------------
analysis["지난연수"] = analysis["연도"] - 1908

x = analysis["지난연수"].to_numpy()
y = analysis["평균기온"].to_numpy()

slope, intercept = np.polyfit(x, y, 1)

analysis["회귀기온"] = (
    intercept + slope * analysis["지난연수"]
)

correlation = np.corrcoef(x, y)[0, 1]


# ---------------------------------------
# 100년당 기온 변화
# ---------------------------------------
slope_100 = slope * 100


# ---------------------------------------
# 최근 20년 회귀
# 2006~2025년
# 관측일수 300일 이상인 해만 사용
# ---------------------------------------
recent = analysis[
    (analysis["연도"] >= 2006) &
    (analysis["연도"] <= 2025)
].copy()

recent_x = recent["연도"].to_numpy() - 1908
recent_y = recent["평균기온"].to_numpy()

recent_slope, recent_intercept = np.polyfit(
    recent_x,
    recent_y,
    1
)

recent_slope_100 = recent_slope * 100


# ---------------------------------------
# 분석 기간 정보
# ---------------------------------------
st.subheader("📊 분석에 사용한 기간")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "전체 회귀 연도 수",
        f"{len(analysis)}년"
    )

with col2:
    st.metric(
        "전체 시작 연도",
        f"{analysis['연도'].min()}년"
    )

with col3:
    st.metric(
        "전체 끝 연도",
        f"{analysis['연도'].max()}년"
    )

with col4:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )

st.caption(
    "2025년 이하이면서 관측일수가 300일 이상인 연도만 분석에 사용했습니다."
)


# ---------------------------------------
# 핵심 결과
# 100년에 몇 도 오르는가?
# ---------------------------------------
st.subheader("🌡️ 100년에 기온이 얼마나 변했을까?")

if slope_100 >= 0:
    direction_text = "오릅니다"
else:
    direction_text = "내려갑니다"

st.markdown(
    f"""
    <div style="
        background-color: #f5f7fa;
        border-radius: 18px;
        padding: 35px;
        text-align: center;
        margin: 15px 0 30px 0;
    ">
        <div style="
            font-size: 22px;
            color: #555;
        ">
            전체 기간 회귀선 기준
        </div>

        <div style="
            font-size: 52px;
            font-weight: bold;
            margin: 12px 0;
        ">
            100년에 {abs(slope_100):.2f}℃
        </div>

        <div style="
            font-size: 20px;
            color: #555;
        ">
            기온이 {direction_text}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------
# 전체 기간 vs 최근 20년 비교
# ---------------------------------------
st.subheader("📈 전체 기간과 최근 20년의 기울기 비교")

compare1, compare2 = st.columns(2)

with compare1:
    st.markdown(
        f"""
        <div style="
            border: 2px solid #ddd;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
        ">
            <div style="
                font-size: 22px;
                font-weight: bold;
            ">
                전체 기간
            </div>

            <div style="
                font-size: 42px;
                font-weight: bold;
                margin: 15px 0;
            ">
                {slope_100:+.2f}℃
            </div>

            <div style="font-size: 16px; color: #666;">
                100년당 변화량
            </div>

            <div style="font-size: 14px; color: #888; margin-top: 8px;">
                {analysis['연도'].min()}~{analysis['연도'].max()}년
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with compare2:
    st.markdown(
        f"""
        <div style="
            border: 2px solid #ddd;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
        ">
            <div style="
                font-size: 22px;
                font-weight: bold;
            ">
                최근 20년
            </div>

            <div style="
                font-size: 42px;
                font-weight: bold;
                margin: 15px 0;
            ">
                {recent_slope_100:+.2f}℃
            </div>

            <div style="font-size: 16px; color: #666;">
                100년당 변화량
            </div>

            <div style="font-size: 14px; color: #888; margin-top: 8px;">
                2006~2025년 · 사용 연도 {len(recent)}년
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.caption(
    "최근 20년 기울기는 2006~2025년 중 관측일수가 300일 이상인 연도만 사용했습니다."
)


# ---------------------------------------
# 산점도 + 전체 기간 회귀선
# ---------------------------------------
st.subheader("📈 서울 연평균기온과 연도의 관계")

fig = go.Figure()


# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=analysis["연도"],
        y=analysis["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        text=analysis["연도"].astype(str) + "년",
        customdata=analysis["관측일수"],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        ),
        marker=dict(size=7)
    )
)


# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=analysis["연도"],
        y=analysis["회귀기온"],
        mode="lines",
        name="전체 기간 회귀선",
        hovertemplate=(
            "%{x}년<br>"
            "회귀 예상기온: %{y:.2f}℃"
            "<extra></extra>"
        ),
        line=dict(width=3)
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    hovermode="x unified",
    height=550,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------------
# 회귀식
# ---------------------------------------
st.subheader("🧮 전체 기간 회귀식")

st.write(
    f"**예상 기온 = {slope:.4f} × (연도 - 1908) + {intercept:.4f}**"
)

st.write(
    f"전체 기간 기울기: **{slope:.4f}℃/년** "
    f"→ **100년에 {slope_100:+.2f}℃**"
)

st.write(
    f"최근 20년 기울기: **{recent_slope:.4f}℃/년** "
    f"→ **100년에 {recent_slope_100:+.2f}℃**"
)


# ---------------------------------------
# 최근 20년 회귀선도 비교해서 보기
# ---------------------------------------
st.subheader("🔎 최근 20년 회귀선 비교")

fig_recent = go.Figure()


# 전체 기간 실제 자료
fig_recent.add_trace(
    go.Scatter(
        x=analysis["연도"],
        y=analysis["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=6),
        hovertemplate=(
            "%{x}년<br>"
            "평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 전체 기간 회귀선
all_years = np.arange(
    analysis["연도"].min(),
    analysis["연도"].max() + 1
)

all_x = all_years - 1908
all_pred = intercept + slope * all_x

fig_recent.add_trace(
    go.Scatter(
        x=all_years,
        y=all_pred,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(width=3)
    )
)


# 최근 20년 회귀선
recent_years = np.arange(2006, 2026)
recent_x_plot = recent_years - 1908

recent_pred = (
    recent_intercept +
    recent_slope * recent_x_plot
)

fig_recent.add_trace(
    go.Scatter(
        x=recent_years,
        y=recent_pred,
        mode="lines",
        name="최근 20년 회귀선",
        line=dict(width=3, dash="dash")
    )
)


fig_recent.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=5
    ),
    height=550,
    hovermode="x unified"
)

st.plotly_chart(
    fig_recent,
    use_container_width=True
)


# ---------------------------------------
# 연도 선택 → 예상 기온
# ---------------------------------------
st.subheader("🔮 연도를 선택해 예상 기온 확인하기")

selected_year = st.slider(
    "예상 기온을 확인할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
    key="prediction_year_slider"
)

selected_x = selected_year - 1908

predicted_temp = intercept + slope * selected_x


st.markdown(
    f"""
    <div style="
        background-color: #f5f7fa;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    ">
        <div style="
            font-size: 22px;
            color: #555;
        ">
            {selected_year}년 예상 연평균기온
        </div>

        <div style="
            font-size: 56px;
            font-weight: bold;
            margin-top: 10px;
        ">
            {predicted_temp:.2f}℃
        </div>

        <div style="
            font-size: 15px;
            color: #777;
        ">
            전체 기간 회귀선 기준
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------
# 1900~2100 예상 회귀선
# ---------------------------------------
prediction_years = np.arange(1900, 2101)
prediction_x = prediction_years - 1908

prediction_temps = (
    intercept +
    slope * prediction_x
)


fig_prediction = go.Figure()


# 전체 회귀선
fig_prediction.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temps,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(width=3)
    )
)


# 실제 관측자료
fig_prediction.add_trace(
    go.Scatter(
        x=analysis["연도"],
        y=analysis["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=6),
        hovertemplate=(
            "%{x}년<br>"
            "실제 평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 선택한 연도
fig_prediction.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상",
        marker=dict(
            size=16,
            symbol="star"
        ),
        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            "예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig_prediction.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        range=[1900, 2100],
        tickmode="linear",
        dtick=20
    ),
    height=550,
    hovermode="x unified"
)

st.plotly_chart(
    fig_prediction,
    use_container_width=True
)


# ---------------------------------------
# 회귀에 사용된 데이터
# ---------------------------------------
with st.expander("📋 회귀에 사용된 연도별 데이터 보기"):

    display_df = analysis.copy()

    display_df["평균기온"] = (
        display_df["평균기온"].round(2)
    )

    display_df["회귀기온"] = (
        display_df["회귀기온"].round(2)
    )

    display_df = display_df[
        [
            "연도",
            "관측일수",
            "평균기온",
            "지난연수",
            "회귀기온"
        ]
    ]

    display_df.columns = [
        "연도",
        "관측일수",
        "실제 평균기온(℃)",
        "1908년부터 지난 연수",
        "회귀 예상기온(℃)"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
