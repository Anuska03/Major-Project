import os
import json
import pandas as pd
import duckdb
from typing import List, Union
from io import StringIO, BytesIO

class CsvIngestor:
    def __init__(self):
        self.metadata_dict = {}
        self.saved_paths = []
        self.table_index = 1

    def ingest(self, file: Union[BytesIO, StringIO], user_id: str, session_id: str) -> List[str]:
        """
        Ingest a single CSV file:
        - Saves raw CSV
        - Extracts metadata
        - Creates DuckDB-compatible table
        - Returns list of saved file paths
        """
        user_id, session_id = str(user_id), str(session_id)
        base_folder = os.path.join("data", user_id, session_id)
        os.makedirs(base_folder, exist_ok=True)

        try:
            df = pd.read_csv(file)
            table_name = f"df{self.table_index}"

            # Save raw CSV
            csv_path = os.path.join(base_folder, f"{table_name}.csv")
            df.to_csv(csv_path, index=False)
            self.saved_paths.append(csv_path)

            # Save metadata
            metadata = self._generate_metadata(df, table_name)
            self.metadata_dict[table_name] = metadata

            # Store as DuckDB-compatible table (optional disk persistence)
            con = duckdb.connect(database=os.path.join(base_folder, "tables.duckdb"), read_only=False)
            con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM read_csv_auto('{csv_path}')")
            con.close()

            self.table_index += 1

        except Exception as e:
            print(f"❌ Error processing CSV: {e}")

        # Save metadata file
        metadata_path = os.path.join(base_folder, "all_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_dict, f, indent=2)
        self.saved_paths.append(metadata_path)

        return self.saved_paths

    def _generate_metadata(self, df: pd.DataFrame, table_name: str) -> dict:
        return {
            "Table name": table_name,
            "Number of rows": len(df),
            "Column details": {
                col: {
                    "dtype": str(df[col].dtype),
                    "nulls": int(df[col].isnull().sum()),
                    "unique_values": int(df[col].nunique())
                }
                for col in df.columns
            },
            "First 3 rows": df.head(3).to_dict(orient="records")
        }
