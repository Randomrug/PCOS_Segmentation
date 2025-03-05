from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Function to load and preprocess data
def load_and_preprocess_data(file_path, feature_weights):
    # Load the dataset
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Strip any extra spaces in the column names

    # Select and rename columns
    columns_to_extract = {
        "Age (yrs)": "Age", "Weight (Kg)": "Weight", "Height(Cm)": "Height",
        "Blood Group": "Blood Group", "Pulse rate(bpm)": "Pulse Rate",
        "RR (breaths/min)": "Respiratory Rate", "Hb(g/dl)": "Hemoglobin",
        "Cycle length(days)": "Cycle Length", "Marraige Status": "Marriage Status",
        "Pregnant(Y/N)": "Pregnant", "Hip(inch)": "Hip", "Waist(inch)": "Waist",
        "Weight gain(Y/N)": "Weight Gain", "hair growth(Y/N)": "Hair Growth",
        "Skin darkening (Y/N)": "Skin Darkening", "Hair loss(Y/N)": "Hair Loss",
        "Pimples(Y/N)": "Pimples", "Fast food (Y/N)": "Fast Food",
        "Reg.Exercise(Y/N)": "Regular Exercise"
    }
    df = df.rename(columns=columns_to_extract)
    df = df.dropna()  # Drop rows with missing values

    # Calculate BMI and add it to the DataFrame
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2

    # Extract relevant features including BMI and Blood Group
    features = df[["Age", "Weight", "Height", "Pulse Rate", "Respiratory Rate", 
                   "Hemoglobin", "Cycle Length", "Hip", "Waist", "Marriage Status", 
                   "Pregnant", "Weight Gain", "Hair Growth", "Skin Darkening", 
                   "Hair Loss", "Pimples", "Fast Food", "Regular Exercise", "Blood Group", "BMI"]]
    
    # Standardize features FIRST
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply feature weights AFTER standardization
    for feature, weight in feature_weights.items():
        if feature in features.columns:
            scaled_features[:, features.columns.get_loc(feature)] *= weight  # Apply weight to feature

    return df, features, scaled_features, scaler

# Function to train GMM and get clusters
def gmm_clustering(scaled_features):
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    
    # Fit a GMM model with a fixed random seed
    gmm = GaussianMixture(n_components=4, random_state=42)  # Set random_state for stability
    clusters = gmm.fit_predict(pca_features)

    return clusters, gmm, pca_features, pca

# Function to preprocess user input
def preprocess_user_input(user_input, feature_weights, scaler, feature_columns):
    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_input], columns=feature_columns)
    
    # Standardize the user input FIRST
    user_scaled = scaler.transform(user_df)
    
    # Apply feature weights AFTER standardization
    for feature, weight in feature_weights.items():
        if feature in feature_columns:
            user_scaled[:, feature_columns.get_loc(feature)] *= weight  # Apply weight to feature
    
    return user_scaled

# Function to generate cluster plot
def generate_plot(pca_features, clusters, gmm, pca_user_point=None):
    plt.figure(figsize=(8, 6))

    # Scatter plot of the clusters
    sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=clusters, palette="viridis", s=60, alpha=0.7)

    # Plot GMM ellipses for each cluster
    ax = plt.gca()
    for i in range(gmm.n_components):
        mean = gmm.means_[i]
        covar = gmm.covariances_[i]
        
        # Eigenvalue decomposition for plotting covariance ellipse
        eigenvalues, eigenvectors = np.linalg.eigh(covar[:2, :2])  # Use only the first two dimensions
        v = eigenvectors * np.sqrt(eigenvalues) * 2  # scaling factor
        
        ax.add_patch(plt.matplotlib.patches.Ellipse(mean[:2], width=v[0, 0], height=v[1, 1], angle=np.arctan2(v[1, 0], v[0, 0]), 
                                                   color='red', alpha=0.3, lw=2))

    if pca_user_point is not None:
        # Flatten the user input point for plotting
        plt.scatter(pca_user_point[0, 0], pca_user_point[0, 1], color='red', marker='X', s=100, label="User Input Point")

    plt.title('GMM Clustering with PCA Reduction to 2 Components')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title="Cluster")

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode plot to base64 for HTML embedding
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    return plot_url

# Function to get cluster description and advice
def get_cluster_description_and_advice(cluster):
    descriptions = {
        0: {
            "name": "Cluster 0 (Purple Dots)",
            "description": "Represents individuals with diverse features, less possibly those with PCOS symptoms but not strongly associated with weight gain. May have moderate BMI, mixed hormone-related symptoms, and variable cycle lengths.",
            "advice": "For Cluster 0 (Purple - Diverse Symptoms, Moderate PCOS Risk)\n"
                      "Maintain a balanced diet with a focus on anti-inflammatory foods like leafy greens, nuts, and berries.\n"
                      "Regular light exercise (yoga, walking) can help regulate cycles.\n"
                      "Monitor cycle length and hormonal symptoms regularly."
        },
        1: {
            "name": "Cluster 1 (Blue Dots)",
            "description": "Likely consists of individuals with lower BMI, relatively normal cycle lengths, and fewer PCOS symptoms. May have lower weight gain, less hormonal imbalance, and better metabolic health.",
            "advice": "For Cluster 1 (Blue - Minimal Symptoms, Lower PCOS Risk)\n"
                      "Continue with healthy habits such as regular exercise and a balanced diet**.\n"
                      "If any symptoms arise (e.g., cycle irregularities, acne), seek early medical advice.\n"
                      "Maintain a consistent sleep schedule to prevent hormonal imbalances."
        },
        2: {
            "name": "Cluster 2 (Green Dots)",
            "description": "Appears to be a mixed category with individuals showing mild-to-moderate PCOS symptoms. Features like skin darkening, pimples, and irregular cycle lengths may be common, but weight gain is not the dominant factor.",
            "advice": "For Cluster 2 (Green - Mild to Moderate Symptoms, Possible PCOS Indicators)**\n"
                      "Introduce a low-GI diet to stabilize blood sugar levels.\n"
                      "Engage in moderate strength training and cardio to improve insulin sensitivity.\n"
                      "Monitor hormonal symptoms and consider consulting a doctor for preventive care."
        },
        3: {
            "name": "Cluster 3 (Yellow Dots)",
            "description": "Strongly associated with weight gain, high BMI, and metabolic disturbances. More likely to exhibit severe PCOS symptoms such as irregular cycles, hair growth, hair loss, and skin darkening.",
            "advice": "For Cluster 3 (Yellow - Severe Symptoms, High PCOS Risk)\n"
                      "Weight management is crucial—focus on a high-protein, fiber-rich diet to regulate insulin.\n"
                      "Engage in intensive workouts (HIIT, strength training) at least 4-5 times a week.\n"
                      "Consider medical intervention (hormonal therapy, supplements) under guidance.\n"
                      "Avoid processed foods and excessive fast food consumption."
        }
    }
    return descriptions.get(cluster, {"name": "Unknown", "description": "No description available.", "advice": "No advice available."})

# Load data and train the model
file_path = r"Dataset\PCOS_data_cleaned.csv"  # Update the file path
feature_weights = {
    "Age": 0.9,  
    "Weight": 1.2,  
    "Height": 0.7,
    "Pulse Rate": 1.2,
    "Respiratory Rate": 1.3,  
    "Hemoglobin": 0.9,  
    "Cycle Length": 0.95,  
    "Hip": 1.1,
    "Waist": 1.1,
    "Marriage Status": 0.5,
    "Pregnant": 0.5,
    "Weight Gain": 1.1,  
    "Hair Growth": 1.1,  
    "Skin Darkening": 1.1,  
    "Hair Loss": 0.9,
    "Pimples": 0.9,
    "Fast Food": 1.1,  
    "Regular Exercise": 1.1,  
    "Blood Group": 0.5,
    "BMI": 1.4,  
}
df, features, scaled_features, scaler = load_and_preprocess_data(file_path, feature_weights)
clusters, gmm, pca_features, pca = gmm_clustering(scaled_features)

# Flask routes
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input from the form
        user_input = [
            int(request.form["age"]),
            float(request.form["weight"]),
            float(request.form["height"]),
            int(request.form["pulse_rate"]),
            int(request.form["respiratory_rate"]),
            float(request.form["hemoglobin"]),
            int(request.form["cycle_length"]),
            float(request.form["hip"]),
            float(request.form["waist"]),
            int(request.form["marriage_status"]),
            int(request.form["pregnant"]),
            int(request.form["weight_gain"]),
            int(request.form["hair_growth"]),
            int(request.form["skin_darkening"]),
            int(request.form["hair_loss"]),
            int(request.form["pimples"]),
            int(request.form["fast_food"]),
            int(request.form["regular_exercise"]),
            float(request.form["weight"]) / (float(request.form["height"]) / 100) ** 2,  # BMI
            int(request.form["blood_group"])  # Blood group mapping
        ]

        # Predict the cluster for the user input
        user_input_scaled = preprocess_user_input(user_input, feature_weights, scaler, features.columns)
        pca_user_input = pca.transform(user_input_scaled)
        cluster = gmm.predict(pca_user_input)[0]

        # Generate plot
        plot_url = generate_plot(pca_features, clusters, gmm, pca_user_point=pca_user_input)

        # Get cluster description and advice
        cluster_info = get_cluster_description_and_advice(cluster)

        # Return the result to the HTML template
        return render_template("pcos2.html", cluster=cluster, plot_url=plot_url, cluster_info=cluster_info)
    
    # Render the form for GET requests
    return render_template("pcos2.html", cluster=None, plot_url=None, cluster_info=None)

if __name__ == "__main__":
    app.run(debug=True)