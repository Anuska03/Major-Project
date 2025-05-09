import os
import json
import duckdb
import pandas as pd
from openai import OpenAI
from typing import Dict, Any
import dotenv

class TextToSQLAgent:
    def __init__(self, query : str, user_id: str, session_id: str):
        dotenv.load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API Key.")
        self.client = OpenAI(api_key=self.api_key)
        self.conn = duckdb.connect(database=":memory:")
        self.query = query
        self.user_id = user_id
        self.session_id = session_id

    def load_metadata(self, user_id: str, session_id: str) -> Dict:
        metadata_path = os.path.join("data", user_id, session_id, "all_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Missing metadata at: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_csvs_to_duckdb(self, user_id: str, session_id: str, metadata: Dict):
        base_dir = os.path.join("data", user_id, session_id)
        for table_name in metadata.keys():
            csv_path = os.path.join(base_dir, f"{table_name}.csv")
            df = pd.read_csv(csv_path)
            self.conn.register(table_name, df)

    def generate_sql_prompt(self, user_query: str, metadata: Dict) -> str:
        prompt_blocks = []
        for table_name, meta in metadata.items():
            sample_rows = pd.DataFrame(meta["First 3 rows"]).to_markdown(index=False)
            columns = ", ".join(meta["Column details"].keys())
            prompt_blocks.append(f"""
Table: {table_name}
Columns: {columns}
Sample Data:
{sample_rows}
""")
        return f"""
You are a SQL assistant. Use the tables below to generate a DuckDB-compatible SQL query.

{''.join(prompt_blocks)}

User Question: {user_query}
SQL:
"""

    def generate_sql(self, prompt: str) -> str:
        res = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You generate accurate DuckDB SQL queries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        return res.choices[0].message.content.strip()

    def run_sql(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).df()

    def process(self, user_id: str, session_id: str, query: str) -> Dict[str, Any]:
        try:
            metadata = self.load_metadata(user_id, session_id)
            self.load_csvs_to_duckdb(user_id, session_id, metadata)
            prompt = self.generate_sql_prompt(query, metadata)
            sql = self.generate_sql(prompt)
            result_df = self.run_sql(sql)

            return {
                "query": query,
                "sql": sql,
                "result": result_df.to_dict(orient="records")
            }

        except Exception as e:
            return {
                "query": query,
                "error": str(e)
            }