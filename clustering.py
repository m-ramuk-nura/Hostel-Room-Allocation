import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import os

class StudentClusterGenerator:
    def __init__(self, input_path='Dataset/final.csv', output_dir='clusters', optimal_k=20):
        self.input_path = input_path
        self.output_dir = output_dir
        self.optimal_k = optimal_k
        self.feature_weights = {
            'How open are you to adjusting if you don’t get roommate of similar choice?': 15.0,
            'Are you comfortable sharing a room with someone with a different food habit?': 12.0,
            'What is your food preference?': 11.0,
            'What is your sleeping pattern?': 10.0,
            'What is your study habit preference?': 10.0,
            'What kind of sleeping environment do you prefer?': 10.0,
            'Do you consider yourself more of an:': 4.0,
        }

    def load_data(self):
        df = pd.read_csv(self.input_path)
        df.columns = df.columns.str.strip()
        return df

    def preprocess_data(self, df):
        feature_columns = list(self.feature_weights.keys())
        df_cluster = df.copy()

        for col in feature_columns:
            if col in df_cluster.columns:
                df_cluster[col] = df_cluster[col].fillna('Unknown')

        label_encoders = {}
        weighted_features = []

        for col in feature_columns:
            if col in df_cluster.columns:
                le = LabelEncoder()
                encoded_col = f"{col}_encoded"
                df_cluster[encoded_col] = le.fit_transform(df_cluster[col].astype(str))
                label_encoders[col] = le

                weight = self.feature_weights[col]
                weighted_col = f"{col}_weighted"
                df_cluster[weighted_col] = df_cluster[encoded_col] * weight
                weighted_features.append(weighted_col)

        return df_cluster, weighted_features

    def perform_clustering(self, df_cluster, weighted_features):
        X_weighted = df_cluster[weighted_features].values
        kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        df_cluster['Cluster'] = kmeans.fit_predict(X_weighted)
        return df_cluster

    def save_clusters(self, df):
        os.makedirs(self.output_dir, exist_ok=True)
        cluster_files = []

        for cluster_id in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster_id]
            output_filename = f"{self.output_dir}/Cluster_{cluster_id:03d}_{len(cluster_data)}_students.csv"
            cluster_data.to_csv(output_filename, index=False)
            cluster_files.append(output_filename)

        return cluster_files

    def run(self):
        df = self.load_data()
        df_cluster, weighted_features = self.preprocess_data(df)
        df_cluster = self.perform_clustering(df_cluster, weighted_features)


        df['Cluster'] = df_cluster['Cluster']

        saved_files = self.save_clusters(df)


        df.to_csv("Dataset/final_with_clusters.csv", index=False)



        return saved_files
