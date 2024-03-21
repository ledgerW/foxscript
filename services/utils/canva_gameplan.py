import sys
sys.path.append('..')

import os
import pandas as pd
import numpy as np
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

if os.getenv('IS_OFFLINE'):
    LAMBDA_DATA_DIR = '.'
else:
    LAMBDA_DATA_DIR = '/tmp'

STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')


def make_canva_plot_csvs(cluster_df: pd.DataFrame, sample_df: pd.DataFrame):
    human_article_sample_cnt = sample_df.query('Tier == 1').shape[0]
    ai_article_sample_cnt = sample_df.query('Tier == [2,3]').shape[0]
    total_volume = sample_df.Volume.sum()
    
    plot_paths = {}

    # Human Articles X Category
    # !!! UPDATE TO PERCENTAGE OF HUMAN ARTICLES
    plot_paths['human_articles_by_category'] = os.path.join(LAMBDA_DATA_DIR, 'human_articles_by_category.csv')
    sample_df\
        .query('Tier == 1')\
        .groupby('Category', as_index=False)\
        .agg(Articles=('Cluster', 'count'))\
        .sort_values(by='Articles', ascending=False)\
        .assign(Articles=lambda df: np.round((df.Articles / human_article_sample_cnt)*100, 2))\
        .to_csv(plot_paths['human_articles_by_category'], index=False)

    # AI Articles X Category
    plot_paths['ai_articles_by_category'] = os.path.join(LAMBDA_DATA_DIR, 'ai_articles_by_category.csv')
    sample_df\
        .query('Tier == [2,3]')\
        .groupby('Category', as_index=False)\
        .agg(Articles=('Cluster', 'count'))\
        .sort_values(by='Articles', ascending=False)\
        .assign(Articles=lambda df: np.round((df.Articles / ai_article_sample_cnt)*100, 2))\
        .to_csv(plot_paths['ai_articles_by_category'], index=False)
    
    # AI Mo Traffic X Category
    plot_paths['ai_traffic_by_category'] = os.path.join(LAMBDA_DATA_DIR, 'ai_traffic_by_category.csv')
    sample_df\
        .query('Tier == [2,3]')\
        .groupby('Category', as_index=False)\
        .agg(Traffic=('Volume', 'sum'))\
        .sort_values(by='Traffic', ascending=False)\
        .assign(Traffic=lambda df: np.round((df.Traffic / total_volume)*100, 2))\
        .to_csv(plot_paths['ai_traffic_by_category'], index=False)
    
    # Article Type by Number of Articles
    plot_paths['num_articles_by_type'] = os.path.join(LAMBDA_DATA_DIR, 'num_articles_by_type.csv')
    human_article_cnt = cluster_df.query('Tier == 1').shape[0]
    ai_article_cnt = cluster_df.query('Tier == [2,3]').shape[0]
    pd.DataFrame({
        'Article Type': ['Human Written', 'AI Written'],
        'Article Count': [human_article_cnt, ai_article_cnt]
    }).to_csv(plot_paths['num_articles_by_type'], index=False)

    
    # Get 1 Year Growth Traffic 
    #sorted_cluster_df = cluster_df.sort_values('Volume', ascending=False)[['Cluster', 'Volume']]
    avg_vol_per_article = cluster_df.Volume.mean()
    growth_rate = [.25, .5, .75, 1]
    month_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    new_articles = [0, 100, 300, 550, 550, 550, 550]
    article_counts = [0, 100, 400, 950, 1500, 2050, 2600]

    batch1_vol = avg_vol_per_article * 100 * 0.01
    batch1 = np.array([0] + [batch1_vol*rate for rate in growth_rate] + [batch1_vol]*8)

    batch2_vol = avg_vol_per_article * 300 * 0.01
    batch2 = np.array([0]*2 + [batch2_vol*rate for rate in growth_rate] + [batch2_vol]*7)

    batch3_vol = avg_vol_per_article * 550 * 0.01
    batch3 = np.array([0]*3 + [batch3_vol*rate for rate in growth_rate] + [batch3_vol]*6)

    batch4_vol = avg_vol_per_article * 550 * 0.01
    batch4 = np.array([0]*4 + [batch4_vol*rate for rate in growth_rate] + [batch4_vol]*5)

    batch5_vol = avg_vol_per_article * 550 * 0.01
    batch5 = np.array([0]*5 + [batch5_vol*rate for rate in growth_rate] + [batch5_vol]*4)

    batch6_vol = avg_vol_per_article * 550 * 0.01
    batch6 = np.array([0]*6 + [batch6_vol*rate for rate in growth_rate] + [batch6_vol]*3)

    new_visits = batch1 + batch2 + batch3 + batch4 + batch5 + batch6
    
    
    # New Organic Search Visits
    plot_paths['organic_search_by_articles'] = os.path.join(LAMBDA_DATA_DIR, 'organic_search_by_articles.csv')

    pd.DataFrame({
        'Month': [str(m) for m in month_number],
        'New Monthly Organic Search Visits': [int(v) for v in new_visits]
    }).to_csv(plot_paths['organic_search_by_articles'], index=False)

    # New Organic Search Leads
    plot_paths['organic_leads_by_articles'] = os.path.join(LAMBDA_DATA_DIR, 'organic_leads_by_articles.csv')

    pd.DataFrame({
        'Month': [str(m) for m in month_number],
        'New Monthly Organic Search Leads': [int(v*0.03) for v in new_visits]
    }).to_csv(plot_paths['organic_leads_by_articles'], index=False)

    return plot_paths


def make_canva_vars_csv(input_vals: dict, cluster_df: pd.DataFrame, cat_dist: pd.Series):
    ## Slide Vars
    canva_vars = {}

    canva_vars['Customer'] = input_vals['customer_name']
    canva_vars['Customer URL'] = input_vals['customer_url']

    ### Total Search Market
    total_search_market = cluster_df.Volume.sum()
    total_search_market = '{:,.0f}'.format(total_search_market)
    canva_vars["Total Search Market"] = total_search_market

    ### Current Mo. Search Traffic (Passed In From SEMRush)
    canva_vars['Current Monthly Traffic'] = '{:,.0f}'.format(input_vals['monthly_traffic'])

    ### Current Total Search Market %
    current_search_market_pct = '{:,.2f}'.format(input_vals['monthly_traffic'] / cluster_df.Volume.sum())
    canva_vars['Current Share of Search Market'] = current_search_market_pct

    ### Current Mo Leads
    current_mo_leads = input_vals['monthly_traffic'] * input_vals['lead_pct']
    current_mo_leads = '{:,.0f}'.format(current_mo_leads)
    canva_vars['Current Monthly Leads'] = current_mo_leads

    ### Total Human Articles
    total_human_articles = cluster_df.query('Tier == 1').shape[0]
    total_human_articles = '{:,.0f}'.format(total_human_articles)
    canva_vars['Total Human Articles'] = total_human_articles

    ### Total AI Articles
    total_ai_articles = cluster_df.query('Tier == [2,3]').shape[0]
    total_ai_articles = '{:,.0f}'.format(total_ai_articles)
    canva_vars['Total AI Articles'] = total_ai_articles

    ### Total Human Article Traffic
    total_human_traffic = cluster_df.query('Tier == 1').Volume.sum()
    total_human_traffic = '{:,.0f}'.format(total_human_traffic)
    canva_vars['Total Human Search Traffic'] = total_human_traffic

    ### Total AI Article Traffic
    total_ai_traffic = cluster_df.query('Tier == [2,3]').Volume.sum()
    total_ai_traffic = '{:,.0f}'.format(total_ai_traffic)
    canva_vars['Total AI Search Traffic'] = total_ai_traffic

    ### Top Customer Category
    top_cust_category = cat_dist.index[0]
    canva_vars['Top Customer Category'] = top_cust_category

    ### Second Customer Category
    second_cust_category = cat_dist.index[1]
    canva_vars['Second Customer Category'] = second_cust_category

    ### Third Customer Category
    third_cust_category = cat_dist.index[2]
    canva_vars['Third Customer Category'] = third_cust_category

    ### Total Potential Search Capture
    total_search_market = cluster_df.Volume.sum()
    total_search_capture = '{:,.0f}'.format(total_search_market * input_vals['capture_pct'])
    canva_vars['Total Potential Search Traffic Capture'] = total_search_capture

    ### Total Potential Search LEADS
    total_search_market = cluster_df.Volume.sum()
    total_search_lead = '{:,.0f}'.format(total_search_market * input_vals['capture_pct'] * input_vals['lead_pct'])
    canva_vars['Total Potential Search Leads'] = total_search_lead

    # Write to CSV
    canva_vars_path = os.path.join(LAMBDA_DATA_DIR, 'canva_vars.csv')
    pd.DataFrame(canva_vars, index=[0]).to_csv(canva_vars_path, index=False)

    return canva_vars_path