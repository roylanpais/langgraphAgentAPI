Langgraph agent in python
Load dataset:
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame
Tool 1: API call (inflation factor)
Simulate API call returning inflation factor.
Example output: {"inflation_factor": 1.03}
Tool2 : Summary statistics and fintered data
Compute:
Average house price (MedHouseVal).
Mean, Std, Median, Min, Max for AveRooms where HouseAge > 20.
Count of observations where AveOccup >= 4th quintile.

Edges: User-> Agent-> Tool-> Agent-> Answer

Agent:
Tool 1: API call (inflation factor)
What is the average house price?
Return: Summary_dict["average_price"]
Tool2 : Summary statistics and fintered data
what was the inflation factor?
Return: api_response["inflation_factor"]