# sl.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai

df = pd.read_csv("final_delivery_info.csv")


df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce")  # Convert to numeric, set invalid to NaN
df = df.dropna(subset=["Cluster", "tSNE1", "tSNE2"])  # Drop rows with NaN in Cluster, tSNE1, or tSNE2
df["Cluster"] = df["Cluster"].astype(int)  # Ensure Cluster is integer type
print(f"Total properties in DataFrame (after cleaning): {len(df)}")
print("Season breakdown (after cleaning):")
print(df["SeasonOfDelivery"].value_counts())

# Streamlit App UI
st.title("Property Lease-Up Dashboard")


season = st.sidebar.selectbox("Filter by Season", ["All"] + list(df["SeasonOfDelivery"].unique()))
if season != "All":
    df = df[df["SeasonOfDelivery"] == season]


# --- Section: t-SNE Cluster Plot ---
st.subheader("📊 Property Clusters (t-SNE)")
fig, ax = plt.subplots()
scatter = ax.scatter(df["tSNE1"], df["tSNE2"], c=df["Cluster"], cmap='tab10', alpha=0.8)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
st.pyplot(fig)

# --- Section: Data Table ---
st.subheader("📋 Property Table")
st.dataframe(df[["ProjID", "RentAtDelivery", "AgeAtDelivery", "SeasonOfDelivery", "Cluster"]])

# --- Section: GPT-Powered Cluster Insight ---
st.subheader("🔍 GPT-Powered Cluster Summary")

cluster_id = st.selectbox("Choose a Cluster", sorted(df["Cluster"].unique()))
cluster_df = df[df["Cluster"] == cluster_id]

# Show stats
st.write(f"📈 Summary Statistics for Cluster {cluster_id}")
st.dataframe(cluster_df.describe().round(2))

if st.button("Generate GPT Insight"):
        prompt = f"""
        You are a real estate analyst specializing in property lease-up trends. 
        Based on the following summary statistics for a cluster of properties:
        - Age at Delivery (years): Mean = {cluster_df['AgeAtDelivery'].mean():.2f}, Std = {cluster_df['AgeAtDelivery'].std():.2f}
        - Submarket Competition (number of nearby deliveries): Mean = {cluster_df['SubmarketCompetition'].mean():.2f}, Std = {cluster_df['SubmarketCompetition'].std():.2f}
        - Rent at Delivery ($): Mean = {cluster_df['RentAtDelivery'].mean():.2f}, Std = {cluster_df['RentAtDelivery'].std():.2f}
        - Average Occupancy in First 3 Months: Mean = {cluster_df['AvgOcc_3Mo_PostDelivery'].mean():.2f}, Std = {cluster_df['AvgOcc_3Mo_PostDelivery'].std():.2f}
        - Dominant Season of Delivery: {cluster_df['SeasonOfDelivery'].mode()[0] if not cluster_df['SeasonOfDelivery'].mode().empty else 'N/A'}
        Provide a  insight describing the lease-up behavior of this cluster, focusing on how these features influence performance.
        """
        
        with st.spinner("🤖 Generating Insight..."):
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response.choices[0].message.content
            st.success("✅ GPT Insight:")
            st.markdown(summary)

# --- Section: Documentation ---
st.subheader("📖 How It Works")
st.markdown("""
- **Data Source**: The dataset is from Task 1, with features like AgeAtDelivery, SubmarketCompetition, RentAtDelivery, AvgOcc_3Mo_PostDelivery, and SeasonOfDelivery.
- **Clustering**: Properties are clustered using t-SNE and KMeans based on the engineered features, visualized in the scatter plot.
- **GenAI**: GPT-4o-mini provides insights on selected clusters, with a prompt engineered to focus on feature impacts.
- **Interactivity**: Plotly charts allow zooming, hovering, and filtering for dynamic exploration.
""")