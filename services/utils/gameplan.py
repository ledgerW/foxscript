import sys
sys.path.append('..')

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

if os.getenv('IS_OFFLINE'):
    LAMBDA_DATA_DIR = '.'
else:
    LAMBDA_DATA_DIR = '/tmp'

STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')


class PDF(FPDF):
    def __init__(self, orientation='L', unit='mm', format='A4'):
        super().__init__(orientation, unit, format)
    
    def header(self):
        self.set_font('Arial', 'B', 20)
        self.cell(0, 10, 'SEO Gameplan', 0, 1, 'C')
        self.ln()
    
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Text color in gray
        self.set_text_color(128)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')
    
    def text_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)
    
    def text_body(self, body, style='', ln=True):
        self.set_font('Arial', style, 12)
        self.cell(0, 10, text=body, align='L')
        if ln:
            self.ln()

    def add_plot(self, image_path, w=250):
        self.image(image_path, w=w)
        self.ln(10)
    
    def add_table(self, table_data):
        self.set_fill_color(r=255, g=212, b=101)
        # Column titles and widths for landscape orientation
        column_titles = [
            "Category",
            "T1 Articles", "T1 Volume",
            "T2 Articles", "T2 Volume",
            "T3 Articles", "T3 Volume",
            "Total Articles", "Total Volume"
        ]
        column_widths = [70, 25]
        self.set_font('Arial', 'B', 10)
        for idx, title in enumerate(column_titles):
            width = column_widths[0] if idx==0 else column_widths[1]
            fill = True if title in ['T2 Volume', 'T3 Volume'] else False
            self.cell(width, 10, title, border=1, fill=fill)
        self.ln()
        for index, row in table_data.iterrows():
            self.set_font('Arial', 'B', 10)
            self.cell(column_widths[0], 10, row['Category_'], border=1)
            self.set_font('Arial', '', 10)
            for tier in range(1, 4):
                if index+1 == table_data.shape[0]:
                    self.set_font('Arial', 'B', 10)
                self.cell(column_widths[1], 10, str(row[f'Total_Keywords_{tier}']), border=1)
                
                fill = False if tier==1 else True
                self.cell(column_widths[1], 10, str(row[f'Total_Search_Volume_{tier}']), border=1, fill=fill)
            self.set_font('Arial', 'B', 10)
            for total in ['Total_Keywords', 'Total_Search_Volume']:
                self.cell(column_widths[1], 10, str(row[total]), border=1)
            self.ln()


def make_cluster_plots(cluster_csv_path):
    # Load the CSV data into a DataFrame
    data = pd.read_csv(cluster_csv_path)

    # Aggregate data for categories
    total_keywords_by_category = data.groupby('Category')['Keyword'].count().reset_index(name='Total Articles')
    total_volume_by_category = data.groupby('Category')['Volume'].sum().reset_index(name='Total Search Volume')

    # Aggregate data for tiers
    total_volume_by_tier = data.groupby('Tier')['Volume'].sum().reset_index(name='Total Search Volume')
    total_keywords_by_tier = data.groupby('Tier')['Keyword'].count().reset_index(name='Total Articles')

    # Visualizations
    # Total Search Volume by Tier
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Tier', y='Total Search Volume', data=total_volume_by_tier, palette="magma")
    plt.title('Total Search Volume by Tier')
    plt.xlabel('Tier')
    plt.ylabel('Total Search Volume')
    plt.tight_layout()
    plt.savefig(os.path.join(LAMBDA_DATA_DIR, 'total_volume_by_tier.png'))
    plt.close()

    # Total Articles by Tier
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Tier', y='Total Articles', data=total_keywords_by_tier, palette="magma")
    plt.title('Total Articles by Tier')
    plt.xlabel('Tier')
    plt.ylabel('Total Articles')
    plt.tight_layout()
    plt.savefig(os.path.join(LAMBDA_DATA_DIR, 'total_articles_by_tier.png'))
    plt.close()

    # Total Search Volume by Category
    plt.figure(figsize=(10, 6))
    total_volume_by_category_sorted = total_volume_by_category[total_volume_by_category['Category'] != 'Grand Total'].sort_values(by='Total Search Volume', ascending=False)
    sns.barplot(x='Total Search Volume', y='Category', data=total_volume_by_category_sorted, palette="magma")
    plt.title('Total Search Volume by Category')
    plt.xlabel('Total Search Volume')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig(os.path.join(LAMBDA_DATA_DIR, 'total_volume_by_category.png'))
    plt.close()

    # Total Keywords by Category
    plt.figure(figsize=(10, 6))
    total_keywords_by_category_sorted = total_keywords_by_category[total_keywords_by_category['Category'] != 'Grand Total'].sort_values(by='Total Articles', ascending=False)
    sns.barplot(x='Total Articles', y='Category', data=total_keywords_by_category_sorted, palette="magma")
    plt.title('Total Articles by Category')
    plt.xlabel('Total Articles')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig(os.path.join(LAMBDA_DATA_DIR, 'total_keywords_by_category.png'))
    plt.close()

    # Paths to the generated plots
    plot_paths = {
        "total_volume_by_tier": os.path.join(LAMBDA_DATA_DIR, 'total_volume_by_tier.png'),
        "total_articles_by_tier": os.path.join(LAMBDA_DATA_DIR, 'total_articles_by_tier.png'),
        "total_volume_by_category": os.path.join(LAMBDA_DATA_DIR, 'total_volume_by_category.png'),
        "total_articles_by_category": os.path.join(LAMBDA_DATA_DIR, 'total_keywords_by_category.png')
    }

    return data, plot_paths


def get_cluster_pivot(data):
    # Aggregate data to calculate total keywords and total search volume for each category and tier
    category_tier_aggregation = data.groupby(['Category', 'Tier']).agg(
        Total_Keywords=('Keyword', 'count'),
        Total_Search_Volume=('Volume', 'sum')
    ).reset_index()

    # Pivot the data for easier table generation in the report
    category_tier_pivot = category_tier_aggregation.pivot_table(
        index='Category',
        columns='Tier',
        values=['Total_Keywords', 'Total_Search_Volume'],
        fill_value=0
    ).reset_index()

    # Flatten the multi-level column hierarchy for easier access and table generation
    category_tier_pivot.columns = ['_'.join(str(v) for v in col).strip() for col in category_tier_pivot.columns.values]

    # Add totals
    category_tier_pivot['Total_Keywords'] = category_tier_pivot['Total_Keywords_1'] + category_tier_pivot['Total_Keywords_2'] + category_tier_pivot['Total_Keywords_3']
    category_tier_pivot['Total_Search_Volume'] = category_tier_pivot['Total_Search_Volume_1'] + category_tier_pivot['Total_Search_Volume_2'] + category_tier_pivot['Total_Search_Volume_3']
    category_tier_pivot = category_tier_pivot.sort_values(by='Total_Search_Volume', ascending=False)

    new_data = {}
    for col in category_tier_pivot.columns:
        if col == 'Category_':
            new_data[col] = category_tier_pivot[col].tolist() + ['Total']
        else:
            new_data[col] = category_tier_pivot[col].tolist() + [category_tier_pivot[col].sum()]
            new_data[col] = [int(v) for v in new_data[col]]

    with_totals_df = pd.DataFrame(new_data)

    return with_totals_df


def make_cluster_report(category_tier_pivot, plot_paths):
    # Create instance of FPDF class with landscape orientation
    pdf = PDF(orientation='L')

    # Add a page
    pdf.add_page()
    pdf.text_title('SEO Content Experiment Design')
    pdf.text_body('Goal', 'B', ln=True)
    pdf.text_body("- Select 300 keywords across six high value Categories with a mix of Tier 2 and Tier 3")
    pdf.text_body("  monthly keyword search volume popularity (med/low search volume and competition)")
    pdf.text_body("- Let run for 30-90 days to measure performance")
    pdf.text_body("- Identify winning Categories generating the most conversions")
    pdf.text_body("- Double down")
    pdf.text_body("")
    
    
    pdf.text_body('KPIs to Track for Success', 'B', ln=True)
    pdf.text_body("- Average Traffic per month per Article by Month 1, Month 2, Month 3, etc...")
    pdf.text_body("- Conversions by Category per Month")
    pdf.text_body("- Conversion Rate by Category")

    # Adding the table of total keywords and total search volume by category and tier
    pdf.add_page()
    pdf.text_title('Total Articles and Search Volume by Category and Tier')
    pdf.add_table(category_tier_pivot)

    # Add plots
    tables_and_plots = [
        ("Total Articles by Tier", None, None, "total_articles_by_tier"),
        ("Total Search Volume by Tier", None, None, "total_volume_by_tier"),
        ("Total Articles by Category", None, None, "total_articles_by_category"),
        ("Total Search by Category", None, None, "total_volume_by_category"),
    ]

    for title, table_data, col_widths, plot_path in tables_and_plots:
        pdf.add_page()
        #pdf.chapter_title(title)
        if table_data is not None:
            pdf.add_table(table_data.values.tolist(), col_widths)
        if plot_path is not None:
            if 'tier' in plot_path:
                pdf.add_plot(plot_paths[plot_path], w=230)
                pdf.text_body("""Tier 1: > 10K search/mo       Tier 2: 1K - 10K search/mo       Tier 3: < 1K search/mo
""", 'B', ln=False)
            else:
                pdf.add_plot(plot_paths[plot_path])

    # Save the PDF to a file in landscape orientation
    report_path = os.path.join(LAMBDA_DATA_DIR, 'seo_keyword_report_category_tier.pdf')
    pdf.output(report_path)

    return report_path