import pandas as pd
import os


def validate_and_clean(file_path):
    print(f"--- Starting Validation for {file_path} ---")

    # 1. Load data & Handle phantom rows/spaces
    # skipinitialspace=True removes spaces after commas
    df = pd.read_csv(file_path, skipinitialspace=True)

    # Drop rows that are completely empty (fixes the 'phantom rows' issue)
    df = df.dropna(how='all')

    # 2. Check for Missing Values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values. Fixing now...")
        # Fill missing ratings with the column mean
        if 'rating' in df.columns:
            df['rating'] = df['rating'].fillna(df['rating'].mean())
        # Fill other missing values with a placeholder
        df = df.fillna("Unknown")

    # 3. DE-DUPLICATION (Task 4: Data Transformation)
    # This specifically fixes the 'cannot reshape' error.
    # We look for rows with the same user_id AND item_id.
    duplicate_interactions = df.duplicated(subset=['user_id', 'item_id']).sum()

    if duplicate_interactions > 0:
        print(f"Found {duplicate_interactions} duplicate user-item pairs. Removing...")
        # Keep the 'last' entry as it's often the most recent rating
        df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')

    # 4. Save Cleaned Data
    output_path = 'data/prepared/clean_interactions.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Validation Complete. Clean data saved to: {output_path}")
    print(f"Final Row Count: {len(df)}")
    return df


if __name__ == "__main__":
    validate_and_clean('sample_interactions.csv')
