import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균기온 데이터를 이용하여 선형회귀 모델을 만들고, "
    "과거 자료로 학습한 모델이 최근 기온을 얼마나 잘 예측하는지 비교합니다."
)


# =========================================================
# 데이터 불러오기
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 결측값 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    # 연도 만들기
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# =========================================================
# 연도별 평균기온 계산
# =========================================================
yearly = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# =========================================================
# 분석 대상
# 2025년 이하
# 관측일수 300일 이상
# =========================================================
analysis = yearly[
    (yearly["연도"] <= 2025)
    & (yearly["관측일수"] >= 300)
].copy()

analysis = analysis.sort_values(
    "연도"
).reset_index(drop=True)


# 데이터가 충분한지 확인
if len(analysis) == 0:
    st.error("분석할 데이터가 없습니다.")
    st.stop()


# =========================================================
# 전체 기간 회귀
# 독립변수 = 1908년부터 지난 연수
# =========================================================
analysis["지난연수"] = (
    analysis["연도"] - 1908
)

x_all = analysis["지난연수"].to_numpy()
y_all = analysis["평균기온"].to_numpy()


# 선형회귀
slope, intercept = np.polyfit(
    x_all,
    y_all,
    1
)


# 회귀 예상값
analysis["회귀기온"] = (
    intercept
    + slope * analysis["지난연수"]
)


# 상관계수
correlation = np.corrcoef(
    x_all,
    y_all
)[0, 1]


# 100년당 변화량
slope_100 = slope * 100


# =========================================================
# 전체 기간 기본 정보
# =========================================================
st.subheader("📊 전체 데이터 분석")

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric(
        "사용한 연도 수",
        f"{len(analysis)}년"
    )

with info2:
    st.metric(
        "시작 연도",
        f"{analysis['연도'].min()}년"
    )

with info3:
    st.metric(
        "끝 연도",
        f"{analysis['연도'].max()}년"
    )

with info4:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )

st.caption(
    "2025년 이하이면서 관측일수가 300일 이상인 연도만 사용했습니다."
)


# =========================================================
# 전체 기간 산점도 + 회귀선
# =========================================================
st.subheader("📈 전체 기간 연평균기온과 회귀선")

fig_all = go.Figure()


# 실제 연평균기온
fig_all.add_trace(
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
        marker=dict(
            size=7
        )
    )
)


# 전체 기간 회귀선
fig_all.add_trace(
    go.Scatter(
        x=analysis["연도"],
        y=analysis["회귀기온"],
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            width=3
        )
    )
)


fig_all.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    height=550,
    hovermode="x unified"
)

st.plotly_chart(
    fig_all,
    use_container_width=True
)


# =========================================================
# 전체 기간 기울기
# =========================================================
st.subheader("🌡️ 전체 기간 기온 변화")

st.metric(
    "100년에 기온이 변하는 정도",
    f"{slope_100:+.2f}℃"
)

st.write(
    f"전체 기간 회귀선의 기울기는 "
    f"**{slope:.4f}℃/년**입니다."
)

st.write(
    f"이를 100년으로 환산하면 "
    f"**100년에 {slope_100:+.2f}℃**입니다."
)


# =========================================================
# 학습 데이터 / 테스트 데이터
# =========================================================

# 최근 50년 학습
train_50 = analysis[
    (analysis["연도"] >= 1956)
    & (analysis["연도"] <= 2005)
].copy()


# 최근 100년 학습
train_100 = analysis[
    (analysis["연도"] >= 1906)
    & (analysis["연도"] <= 2005)
].copy()


# 최근 20년 테스트
test_data = analysis[
    (analysis["연도"] >= 2006)
    & (analysis["연도"] <= 2025)
].copy()


st.subheader("🧪 학습 데이터와 테스트 데이터")

data1, data2, data3 = st.columns(3)

with data1:
    st.metric(
        "50년 학습",
        f"{len(train_50)}년"
    )
    st.caption("1956~2005년")

with data2:
    st.metric(
        "100년 학습",
        f"{len(train_100)}년"
    )
    st.caption("1906~2005년")

with data3:
    st.metric(
        "공통 테스트",
        f"{len(test_data)}년"
    )
    st.caption("2006~2025년")


# =========================================================
# 선형회귀 모델 만들기
# =========================================================

def train_and_evaluate(
    train_data,
    test_data
):

    # X
    X_train = (
        train_data["연도"] - 1908
    ).to_numpy().reshape(-1, 1)

    X_test = (
        test_data["연도"] - 1908
    ).to_numpy().reshape(-1, 1)

    # y
    y_train = train_data["평균기온"].to_numpy()
    y_test = test_data["평균기온"].to_numpy()

    # 모델
    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    # 예측
    predictions = model.predict(
        X_test
    )

    # 평가
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # 기울기
    model_slope = model.coef_[0]

    # 100년당 기울기
    model_slope_100 = model_slope * 100

    return (
        model,
        predictions,
        mae,
        mse,
        r2,
        model_slope,
        model_slope_100
    )


# 50년 모델
(
    model_50,
    pred_50,
    mae_50,
    mse_50,
    r2_50,
    slope_50,
    slope100_50
) = train_and_evaluate(
    train_50,
    test_data
)


# 100년 모델
(
    model_100,
    pred_100,
    mae_100,
    mse_100,
    r2_100,
    slope_100_model,
    slope100_100
) = train_and_evaluate(
    train_100,
    test_data
)


# =========================================================
# 기울기 비교
# =========================================================
st.subheader("📐 학습 기간에 따른 회귀선 기울기 비교")

slope1, slope2 = st.columns(2)

with slope1:

    st.metric(
        "1956~2005년 학습",
        f"{slope100_50:+.2f}℃ / 100년"
    )

    st.caption(
        f"연간 기울기: {slope_50:.4f}℃"
    )


with slope2:

    st.metric(
        "1906~2005년 학습",
        f"{slope100_100:+.2f}℃ / 100년"
    )

    st.caption(
        f"연간 기울기: {slope_100_model:.4f}℃"
    )


# =========================================================
# 예측 성능 비교
# =========================================================
st.subheader("🎯 최근 20년 예측 성능 비교")

comparison = pd.DataFrame(
    {
        "모델": [
            "최근 50년 학습 (1956~2005)",
            "최근 100년 학습 (1906~2005)"
        ],
        "MAE (℃)": [
            mae_50,
            mae_100
        ],
        "MSE (℃²)": [
            mse_50,
            mse_100
        ],
        "R²": [
            r2_50,
            r2_100
        ],
        "기울기 (℃/100년)": [
            slope100_50,
            slope100_100
        ]
    }
)


comparison["MAE (℃)"] = (
    comparison["MAE (℃)"].round(3)
)

comparison["MSE (℃²)"] = (
    comparison["MSE (℃²)"].round(3)
)

comparison["R²"] = (
    comparison["R²"].round(3)
)

comparison["기울기 (℃/100년)"] = (
    comparison["기울기 (℃/100년)"].round(3)
)


st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


st.caption(
    "MAE와 MSE는 낮을수록 예측 오차가 작고, "
    "R²는 높을수록 테스트 데이터를 잘 설명합니다."
)


# =========================================================
# 실제값 vs 예측값
# =========================================================
st.subheader("🔍 최근 20년 실제 기온과 예측 기온")

fig_test = go.Figure()


# 실제값
fig_test.add_trace(
    go.Scatter(
        x=test_data["연도"],
        y=test_data["평균기온"],
        mode="lines+markers",
        name="실제 연평균기온",
        line=dict(
            width=3
        ),
        marker=dict(
            size=7
        ),
        hovertemplate=(
            "%{x}년<br>"
            "실제 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 50년 학습 모델
fig_test.add_trace(
    go.Scatter(
        x=test_data["연도"],
        y=pred_50,
        mode="lines",
        name="1956~2005 학습",
        line=dict(
            width=3,
            dash="dash"
        ),
        hovertemplate=(
            "%{x}년<br>"
            "50년 학습 예측: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 100년 학습 모델
fig_test.add_trace(
    go.Scatter(
        x=test_data["연도"],
        y=pred_100,
        mode="lines",
        name="1906~2005 학습",
        line=dict(
            width=3,
            dash="dot"
        ),
        hovertemplate=(
            "%{x}년<br>"
            "100년 학습 예측: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig_test.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=2
    ),
    height=550,
    hovermode="x unified"
)

st.plotly_chart(
    fig_test,
    use_container_width=True
)


# =========================================================
# 결과 해석
# =========================================================
st.subheader("💡 모델 비교 결과")

if mae_50 < mae_100:
    mae_result = "최근 50년 학습 모델의 MAE가 더 작습니다."
elif mae_50 > mae_100:
    mae_result = "최근 100년 학습 모델의 MAE가 더 작습니다."
else:
    mae_result = "두 모델의 MAE가 같습니다."


if mse_50 < mse_100:
    mse_result = "최근 50년 학습 모델의 MSE가 더 작습니다."
elif mse_50 > mse_100:
    mse_result = "최근 100년 학습 모델의 MSE가 더 작습니다."
else:
    mse_result = "두 모델의 MSE가 같습니다."


if r2_50 > r2_100:
    r2_result = "최근 50년 학습 모델의 R²가 더 높습니다."
elif r2_50 < r2_100:
    r2_result = "최근 100년 학습 모델의 R²가 더 높습니다."
else:
    r2_result = "두 모델의 R²가 같습니다."


st.write(f"• **MAE:** {mae_result}")
st.write(f"• **MSE:** {mse_result}")
st.write(f"• **R²:** {r2_result}")


slope_difference = abs(
    slope100_50 - slope100_100
)

st.write(
    f"두 모델의 100년당 기울기 차이는 "
    f"**{slope_difference:.2f}℃**입니다."
)


# =========================================================
# 연도 선택 → 예상 기온
# =========================================================
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

predicted_temp = (
    intercept
    + slope * selected_x
)


st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f}℃"
)

st.caption(
    "전체 기간 회귀선 기준"
)


# =========================================================
# 1900~2100년 회귀 예상 그래프
# =========================================================
prediction_years = np.arange(
    1900,
    2101
)

prediction_x = (
    prediction_years - 1908
)

prediction_temps = (
    intercept
    + slope * prediction_x
)


fig_prediction = go.Figure()


# 회귀선
fig_prediction.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temps,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            width=3
        )
    )
)


# 실제 관측값
fig_prediction.add_trace(
    go.Scatter(
        x=analysis["연도"],
        y=analysis["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=6
        ),
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


# =========================================================
# 회귀에 사용된 데이터
# =========================================================
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
