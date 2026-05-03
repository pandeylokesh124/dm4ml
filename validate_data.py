import os
from datetime import datetime

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


class DataValidator:
    REQUIRED_COLUMNS = ['user_id', 'item_id', 'rating']
    RATING_RANGE = (1, 5)

    def __init__(self, output_dir="data"):
        self.output_dir = output_dir

    def _build_report_text(self, metrics, issues, output_path):
        lines = [
            'Data Quality Report',
            '===================',
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            '',
            'Summary Metrics',
            '---------------',
        ]

        for label, value in metrics.items():
            lines.append(f"{label}: {value}")

        lines.extend(['', 'Issues & Validation Notes', '------------------------'])
        if issues:
            lines.extend([f"- {issue}" for issue in issues])
        else:
            lines.append('- No issues detected.')

        report_text = '\n'.join(lines)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        return report_text

    def _build_report_pdf(self, report_text, output_path):
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter
        margin = 0.75 * inch
        x = margin
        y = height - margin
        line_height = 14

        for line in report_text.splitlines():
            if y < margin:
                c.showPage()
                y = height - margin
            c.drawString(x, y, line)
            y -= line_height

        c.save()

    def validate_and_clean(self, file_path):
        print(f"--- Starting Validation for {file_path} ---")

        df = pd.read_csv(file_path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        df = df.dropna(how='all')

        issues = []
        metrics = {
            'Input file': file_path,
            'Raw row count': len(df),
            'Raw column count': len(df.columns),
        }

        # Column checks
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        extra_columns = [col for col in df.columns if col not in self.REQUIRED_COLUMNS]

        if missing_columns:
            issues.append(f"Missing required columns: {', '.join(missing_columns)}")
        if extra_columns:
            issues.append(f"Extra columns detected: {', '.join(extra_columns)}")

        # Rating validation
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            invalid_rating = df['rating'].isnull().sum()
            out_of_range = df[~df['rating'].between(*self.RATING_RANGE)].shape[0]

            if invalid_rating > 0:
                issues.append(f"Found {invalid_rating} non-numeric or invalid rating values.")
            if out_of_range > 0:
                issues.append(
                    f"Found {out_of_range} ratings outside the allowed range {self.RATING_RANGE[0]}-{self.RATING_RANGE[1]}."
                )

        # Missing values
        missing_by_column = df.isnull().sum().to_dict()
        total_missing = sum(missing_by_column.values())

        if total_missing > 0:
            issues.append(f"Total missing values: {total_missing}")
            for col, count in missing_by_column.items():
                if count > 0:
                    issues.append(f"  - {col}: {count} missing")

        # Duplicates
        duplicate_count = (
            df.duplicated(subset=['user_id', 'item_id']).sum()
            if all(col in df.columns for col in ['user_id', 'item_id'])
            else 0
        )

        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate user-item rows.")
            df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')

        # Handle missing
        if total_missing > 0 and 'rating' in df.columns:
            mean_rating = df['rating'].mean()
            df['rating'] = df['rating'].fillna(mean_rating)

        df = df.fillna('Unknown')

        # Clip ratings
        if 'rating' in df.columns:
            df.loc[~df['rating'].between(*self.RATING_RANGE), 'rating'] = df['rating'].clip(*self.RATING_RANGE)

        # Save cleaned data
        output_data_path = os.path.join(self.output_dir, 'prepared', 'clean_interactions.csv')
        os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
        df.to_csv(output_data_path, index=False)

        # Metrics update
        metrics.update({
            'Cleaned row count': len(df),
            'Cleaned column count': len(df.columns),
            'Duplicate rows removed': duplicate_count,
            'Total missing values handled': total_missing,
        })

        # Reports
        report_base = os.path.join(self.output_dir, 'data_quality_report')
        report_txt = f'{report_base}.txt'
        report_pdf = f'{report_base}.pdf'

        report_text = self._build_report_text(metrics, issues, report_txt)
        self._build_report_pdf(report_text, report_pdf)

        print(f"Validation Complete. Clean data saved to: {output_data_path}")
        print(f"Reports generated: {report_txt}, {report_pdf}")
        print(f"Final Row Count: {len(df)}")

        return df