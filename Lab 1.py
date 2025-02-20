import csv

def find_s_algorithm(file_path):
    # Read training data from CSV file
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    
    # Separate header and instances
    header = data[0]
    instances = data[1:]
    
    # Initialize most specific hypothesis (h) with "?" except target value
    h = ["0"] * (len(header) - 1)
    
    # Process each training example
    for instance in instances:
        if instance[-1].strip().lower() == "yes":  # Target attribute is "Yes"
            for i in range(len(h)):
                if h[i] == "0":  # Initialize h
                    h[i] = instance[i]
                elif h[i] != instance[i]:  # Generalize h if needed
                    h[i] = "?"
    
    return h

# Path to the CSV file
file_path = "training_data_task1.csv"

# Find-S algorithm output
hypothesis = find_s_algorithm(file_path)
print(f"The most specific hypothesis: {hypothesis}")
