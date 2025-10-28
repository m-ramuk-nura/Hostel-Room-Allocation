import os
import re
import ast
import dotenv
import pandas as pd
import google.generativeai as genai

class HealthConditionRanker:
    def __init__(self, dataset_path='Dataset/final.csv', folder_path='clusters'):
        self.dataset_path = dataset_path
        self.folder_path = folder_path
        self.key = None
        self.ranking = {}

    def load_api_key(self):
        dotenv.load_dotenv()
        self.key = os.getenv('GOOGLE_GEMINI_API_KEY')
        if not self.key:
            raise ValueError("❌ GOOGLE_GEMINI_API_KEY not found in .env file")
        genai.configure(api_key=self.key)
       

    def extract_conditions(self):
        df = pd.read_csv(self.dataset_path)
        df.columns = df.columns.str.strip()

        health_col = "Do you have any health conditions or allergies your roommate should know?  (If no enter 'Nil')"
        if health_col not in df.columns:
            raise ValueError(f"❌ Column '{health_col}' not found in dataset")

        conditions = df[health_col].dropna().unique().tolist()
      
        return conditions

    def get_rankings_from_gemini(self, conditions):
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
        You are ranking the following health conditions to prioritize hostel room allocation
        based on how each condition affects environmental or physical accommodation needs.

        Guidelines:
        1. Output only a valid **Python dictionary** — no markdown, code fences, comments, or explanations.
        2. Smaller numbers indicate higher priority (1 = top priority).
        3. Rank only these conditions: {conditions}.
        4. Base your ranking on a combination of:
        - **Breathing and Air Sensitivity:** Conditions that require clean air, ventilation, or less dust (e.g., respiratory or allergy-related) get the highest priority (Rank 1–2).
        - **Mobility Limitations:** Conditions that make climbing stairs or walking difficult should also be ranked very high (Rank 1–2).
        - **Chronic or Metabolic Disorders:** Long-term but manageable conditions (e.g., needing medication or diet control) get moderate priority (Rank 3–4).
        - **Mild Discomforts or Non-critical Allergies:** Low-impact issues get lower priority (Rank 5–6).
        - **No condition or equivalent terms** (nil, none, no, nill, empty, or missing) → exclude completely.
        5. Normalize all condition names (Title Case).
        6. Example format: {{"Asthma": 1, "Diabetes": 3, "Headache": 5}}
        """

       
        response = model.generate_content(prompt)

        clean_text = re.sub(r"```[a-zA-Z]*", "", response.text).replace("```", "").strip()
        self.ranking = ast.literal_eval(clean_text)
        

    def update_cluster_files(self):
        ranking_dict_normalized = {k.strip().title(): v for k, v in self.ranking.items()}
        health_col = "Do you have any health conditions or allergies your roommate should know?  (If no enter 'Nil')"


        for filename in os.listdir(self.folder_path):
            if not filename.endswith('.csv'):
                continue

            file_path = os.path.join(self.folder_path, filename)
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()

            if health_col not in df.columns:
                continue

            df['__clean_condition__'] = (
                df[health_col]
                .astype(str)
                .str.strip()
                .str.title()
                .replace({"Nil": "", "Nill": "", "No": "", "None": "", "Nan": ""})
            )

            df['Health_Condition_Rank'] = (
                df['__clean_condition__']
                .map(ranking_dict_normalized)
                .fillna(-1)
                .astype(int)
            )

            df.drop(columns=['__clean_condition__'], inplace=True)
            df.to_csv(file_path, index=False)


    def run(self):
        self.load_api_key()
        conditions = self.extract_conditions()
        self.get_rankings_from_gemini(conditions)
        self.update_cluster_files()
