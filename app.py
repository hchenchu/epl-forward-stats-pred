import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="EPL Forwards Explorer", layout="wide")
st.title("EPL Forwards Explorer")
st.caption("Upload the cleaned forwards CSV, explore it with a chart, and predict a stat with a straight line.")

# The numeric columns the app works with, and the columns the file must have.
NUMERIC_COLS = [
    "Appearances", "Wins", "Losses", "Goals", "Goals per match",
    "Headed goals", "Goals with right foot", "Goals with left foot",
    "Penalties scored", "Freekicks scored", "Shooting accuracy %",
    "Assists", "Yellow cards", "Red cards", "Fouls",
]
REQUIRED_COLS = ["Name", "Club", "Position"] + NUMERIC_COLS

# Not every input makes sense for every target.
# This table lists the inputs that are relevant for each target.
# (We leave out headed, right-foot and left-foot goals as inputs for Goals,
# because they simply add up to Goals.)
TARGET_INPUTS = {
    "Goals": ["Appearances", "Wins", "Losses", "Shooting accuracy %", "Assists", "Penalties scored", "Freekicks scored"],
    "Goals per match": ["Appearances", "Shooting accuracy %", "Assists", "Penalties scored", "Freekicks scored"],
    "Assists": ["Appearances", "Wins", "Losses", "Goals", "Goals per match", "Shooting accuracy %"],
    "Wins": ["Appearances", "Goals", "Assists", "Goals per match"],
    "Losses": ["Appearances", "Goals", "Assists", "Yellow cards"],
    "Headed goals": ["Appearances", "Goals", "Wins"],
    "Goals with right foot": ["Appearances", "Goals", "Wins", "Shooting accuracy %"],
    "Goals with left foot": ["Appearances", "Goals", "Wins", "Shooting accuracy %"],
    "Penalties scored": ["Appearances", "Goals", "Wins"],
    "Yellow cards": ["Appearances", "Fouls", "Red cards", "Losses"],
    "Red cards": ["Appearances", "Fouls", "Yellow cards"],
    "Fouls": ["Appearances", "Yellow cards", "Red cards", "Losses"],
}
ALL_INPUTS = [c for c in NUMERIC_COLS if any(c in v for v in TARGET_INPUTS.values())]


def targets_for(a, b):
    # Targets for which both chosen inputs are relevant.
    return [t for t, inputs in TARGET_INPUTS.items() if a in inputs and b in inputs and t not in (a, b)]


def partners_for(a):
    # Second inputs that share at least one target with the first input.
    return [b for b in ALL_INPUTS if b != a and targets_for(a, b)]


# The shape is the same as before.
# The user uploads a file, and everything else sits inside an if.
uploaded_file = st.file_uploader("Upload the cleaned EPL forwards CSV", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("This file could not be read as a CSV.")
        st.stop()

    # Before we use the file, we check it is the right one.
    # Each check shows a message and stops the app.
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error("This file is missing required column(s): " + ", ".join(missing))
        st.stop()

    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = [c for c in NUMERIC_COLS if df[c].isna().any()]
    if bad:
        st.error("These columns have blank or non-numeric values: " + ", ".join(bad))
        st.stop()

    if (df["Position"].astype(str).str.lower() != "forward").any():
        st.error("Every row must be a forward. This looks like the uncleaned dataset.")
        st.stop()

    # Players with 0 appearances only add rows of zeros.
    df = df[df["Appearances"] >= 1].copy()
    if len(df) < 10:
        st.error("Need at least 10 players with one or more appearances.")
        st.stop()

    # First, the user picks a player.
    # Everything below depends on this choice, so until a player is chosen
    # we show a message and stop. The box is searchable: type part of a name.
    df["Label"] = df["Name"] + " (" + df["Club"] + ")"
    player_label = st.selectbox("Select a player", sorted(df["Label"]), index=None,
                                placeholder="Type or choose a forward...")
    if player_label is None:
        st.info("Select a player to continue.")
        st.stop()
    player = df[df["Label"] == player_label].iloc[0]

    # A small card with the player's main numbers.
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Appearances", f"{player['Appearances']:,.0f}")
    p2.metric("Goals", f"{player['Goals']:,.0f}")
    p3.metric("Assists", f"{player['Assists']:,.0f}")
    p4.metric("Goals per match", f"{player['Goals per match']:.2f}")

    # Two tabs: one for charts, one for predictions.
    tab_chart, tab_predict = st.tabs(["Visualize", "Predict"])

    # ---- Tab 1: the chart ----
    with tab_chart:
        col1, col2 = st.columns(2)
        column = col1.selectbox("Column", NUMERIC_COLS, index=NUMERIC_COLS.index("Goals"))
        chart_type = col2.selectbox("Chart type", ["Histogram", "Scatter", "Line"])

        # Where does the chosen player stand on this column?
        player_value = float(player[column])
        rank = int((df[column] > player_value).sum()) + 1
        st.caption(f"{player['Name']}: {column} = {player_value:g}, ranked {rank} of {len(df)} forwards.")

        if chart_type == "Histogram":
            # A histogram only needs one column.
            # A red dashed line marks the chosen player.
            fig = px.histogram(df, x=column, nbins=15, title=f"Distribution of {column}")
            fig.add_vline(x=player_value, line_dash="dash", line_color="red", annotation_text=player["Name"])
        else:
            # Scatter and line need a second column for the x axis.
            x_options = [c for c in NUMERIC_COLS if c != column]
            x_col = st.selectbox("X axis", x_options)
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=column, hover_name="Name",
                                 title=f"{column} versus {x_col}")
                # The chosen player is drawn as a big red dot.
                fig.add_scatter(x=[float(player[x_col])], y=[player_value], mode="markers+text",
                                text=[player["Name"]], textposition="top center",
                                marker=dict(size=14, color="red"), name="Selected player")
            else:
                # For a line we average the column for each x value,
                # so the line runs from left to right without zigzagging.
                line_data = df.groupby(x_col)[column].mean().reset_index().sort_values(x_col)
                fig = px.line(line_data, x=x_col, y=column, markers=True,
                              title=f"Average {column} by {x_col}")
                # The red dot is the chosen player, so you can compare him with the average line.
                fig.add_scatter(x=[float(player[x_col])], y=[player_value], mode="markers+text",
                                text=[player["Name"]], textposition="top center",
                                marker=dict(size=14, color="red"), name="Selected player")
        st.plotly_chart(fig, width="stretch")

    # ---- Tab 2: the prediction ----
    with tab_predict:
        # Pick two inputs. The target list only shows relevant targets.
        c1, c2, c3 = st.columns(3)
        in1 = c1.selectbox("Input 1: what you know", ALL_INPUTS)
        in2 = c2.selectbox("Input 2: what else you know", partners_for(in1))
        target = c3.selectbox("Target: what you want to predict", targets_for(in1, in2))

        # Then the model, with two input columns this time.
        x = df[[in1, in2]]
        y = df[target]
        model = LinearRegression()
        model.fit(x, y)

        # With two inputs we cannot draw one line through the dots.
        # Instead we plot what the model predicted against what really happened.
        # The closer the dots are to the line, the better the model.
        check = pd.DataFrame({"Predicted": model.predict(x), "Actual": y, "Name": df["Name"]})
        fig = px.scatter(check, x="Predicted", y="Actual", hover_name="Name",
                         title=f"Predicted versus actual {target}")
        low_end = min(check["Predicted"].min(), check["Actual"].min())
        high_end = max(check["Predicted"].max(), check["Actual"].max())
        fig.add_scatter(x=[low_end, high_end], y=[low_end, high_end], mode="lines", name="Perfect prediction")
        # The chosen player is the red dot.
        player_inputs = pd.DataFrame({in1: [player[in1]], in2: [player[in2]]})
        fig.add_scatter(x=[model.predict(player_inputs)[0]], y=[float(player[target])], mode="markers+text",
                        text=[player["Name"]], textposition="top center",
                        marker=dict(size=14, color="red"), name="Selected player")
        st.plotly_chart(fig, width="stretch")

        # Three numbers go underneath: one slope per input, and R squared.
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Each extra {in1} adds", f"{model.coef_[0]:,.2f}")
        m2.metric(f"Each extra {in2} adds", f"{model.coef_[1]:,.2f}")
        m3.metric("R2 on this data", f"{model.score(x, y):.2f}")
        if model.score(x, y) < 0.3:
            st.warning("R2 is low: these inputs explain little of the target, so treat predictions as rough.")

        # The last part is the prediction.
        # The boxes start at the chosen player's real numbers.
        # Change them to ask a what-if question.
        # The table we give to predict has the same column names as the training data.
        st.subheader("Try it")
        lo1, hi1 = df[in1].min(), df[in1].max()
        lo2, hi2 = df[in2].min(), df[in2].max()
        v1_col, v2_col = st.columns(2)
        value1 = v1_col.number_input(f"Enter a value for {in1} ({lo1:g} to {hi1:g})",
                                     value=float(player[in1]), key=f"{player_label}|{in1}")
        value2 = v2_col.number_input(f"Enter a value for {in2} ({lo2:g} to {hi2:g})",
                                     value=float(player[in2]), key=f"{player_label}|{in2}")

        # If a value is outside the range of the data, we show an error
        # and give no prediction. A line does not know where its data stopped.
        # Ask it about 1,000 appearances, and it will happily give a confident number.
        errors = []
        if value1 < lo1 or value1 > hi1:
            errors.append(f"{in1} = {value1:g} is outside the range in the data ({lo1:g} to {hi1:g}).")
        if value2 < lo2 or value2 > hi2:
            errors.append(f"{in2} = {value2:g} is outside the range in the data ({lo2:g} to {hi2:g}).")

        if errors:
            for message in errors:
                st.error(message + " Enter a value inside the range.")
        else:
            prediction = model.predict(pd.DataFrame({in1: [value1], in2: [value2]}))[0]
            r1, r2 = st.columns(2)
            r1.metric(f"Predicted {target}", f"{max(prediction, 0):,.2f}")
            r2.metric(f"{player['Name']}'s actual {target}", f"{player[target]:,.2f}")
            if prediction < 0:
                st.caption("The model gave a negative number, so it is shown as 0.")
else:
    st.info("Upload epl_forwards_clean.csv to get started.")
