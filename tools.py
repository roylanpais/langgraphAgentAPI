import json
import pandas as pd
from sklearn.datasets import fetch_california_housing
from langchain_core.tools import tool

@tool
def get_inflation_factor() -> str:
    """
    Returns a JSON string with the inflation factor.
    """
    result = {"inflation_factor": 1.03}
    return json.dumps(result)

@tool
def get_housing_statistics() -> str:
    """
    Compute summary statistics and filtered data for the California Housing dataset.
    """
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    avg_price = df['MedHouseVal'].mean()
    
    filtered_df_age = df[df['HouseAge'] > 20]
    ave_rooms_stats = filtered_df_age['AveRooms'].agg(['mean', 'std', 'median', 'min', 'max']).to_dict()
    
    q80 = df['AveOccup'].quantile(0.8)
    count_occup = df[df['AveOccup'] >= q80].shape[0]
    
    results = {
        "average_price": avg_price,
        "ave_rooms_stats_house_age_gt_20": ave_rooms_stats,
        "count_ave_occup_ge_4th_quintile": count_occup
    }
    
    return json.dumps(results)
