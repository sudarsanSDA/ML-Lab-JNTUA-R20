import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian Network
model = BayesianNetwork([
    ('Age', 'HeartDisease'),
    ('ChestPainType', 'HeartDisease'),
    ('BloodPressure', 'HeartDisease'),
    ('Cholesterol', 'HeartDisease'),
    ('MaxHeartRate', 'HeartDisease'),
])

# Define the CPDs (Conditional Probability Distributions)
model.add_cpds(
    TabularCPD(variable='Age', variable_card=4, values=[[0.25], [0.25], [0.25], [0.25]]),
    TabularCPD(variable='ChestPainType', variable_card=3, values=[[0.33], [0.33], [0.33]]),
    TabularCPD(variable='BloodPressure', variable_card=4, values=[[0.25], [0.25], [0.25], [0.25]]),
    TabularCPD(variable='Cholesterol', variable_card=3, values=[[0.33], [0.33], [0.34]]),
    TabularCPD(variable='MaxHeartRate', variable_card=2, values=[[0.5], [0.5]]),

    # Define CPD for HeartDisease given its parents
    TabularCPD(
        variable='HeartDisease', variable_card=2,
        values=[[0.8]*288,  # P(HeartDisease=No) for all 288 combinations
                [0.2]*288], # P(HeartDisease=Yes) for all 288 combinations
        evidence=['Age', 'ChestPainType', 'BloodPressure', 'Cholesterol', 'MaxHeartRate'],
        evidence_card=[4, 3, 4, 3, 2]
    )
)

# Check if the model is valid
assert model.check_model()

# Perform inference using VariableElimination
inference = VariableElimination(model)

# Query to predict heart disease for a patient with known data
# Example: Age = 1, ChestPainType = 1 (these would map to indices based on your encoding for Age and ChestPainType)
predicted = inference.query(variables=['HeartDisease'], evidence={'Age': 1, 'ChestPainType': 1})

print("Predicted Heart Disease status:\n", predicted)
