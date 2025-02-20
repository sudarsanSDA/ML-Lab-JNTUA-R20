import csv

def candidate_elimination_algorithm(file_path):
    # Read training data from CSV file
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    
    # Separate header and instances
    header = data[0]
    instances = data[1:]
    
    # Initialize specific and general hypotheses
    num_attributes = len(header) - 1
    specific_h = ["0"] * num_attributes
    general_h = [["?"] * num_attributes]
    
    # Process training examples
    for instance in instances:
        if instance[-1].strip().lower() == "yes":  # Positive example
            for i in range(num_attributes):
                if specific_h[i] == "0":  # Initialize specific hypothesis
                    specific_h[i] = instance[i]
                elif specific_h[i] != instance[i]:  # Generalize specific hypothesis
                    specific_h[i] = "?"
            
            # Retain general hypotheses that agree with the positive example
            general_h = [gh for gh in general_h if all(
                gh[i] == "?" or gh[i] == instance[i] for i in range(num_attributes)
            )]
        else:  # Negative example
            new_general_h = []
            for gh in general_h:
                for i in range(num_attributes):
                    if gh[i] == "?":  # Specialize general hypothesis
                        for value in set(row[i] for row in instances if row[-1].strip().lower() == "yes"):
                            if value != instance[i]:
                                new_hypothesis = gh[:]
                                new_hypothesis[i] = value
                                new_general_h.append(new_hypothesis)
            general_h = [h for h in new_general_h if any(specific_h[i] == "?" or h[i] == specific_h[i] for i in range(num_attributes))]
    
    return specific_h, general_h

# Path to the CSV file
file_path = "training_data_task2.csv"

# Candidate-Elimination algorithm output
specific_h, general_h = candidate_elimination_algorithm(file_path)
print(f"Most Specific Hypothesis: {specific_h}")
print(f"Most General Hypotheses: {general_h}")
