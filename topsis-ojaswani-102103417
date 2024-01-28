import sys
import pandas as pd
import numpy as np

def normalize_matrix(matrix):
    normalized_matrix = matrix / np.sqrt(np.sum(matrix**2, axis=0))
    return normalized_matrix

def validate_input_parameters(args):
    if len(args) != 5:
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)

def read_input_data(input_file):
    try:
        data = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    return data

def validate_input_data(data, weights, impacts):
    if len(data.columns) < 3:
        print("Error: Input file must contain three or more columns.")
        sys.exit(1)

    if not all(data.iloc[:, 1:].applymap(np.isreal).all()):
        print("Error: Non-numeric values found in columns from 2nd to last.")
        sys.exit(1)

    if len(weights) != len(data.columns) - 1 or len(impacts) != len(data.columns) - 1:
        print("Error: Number of weights, impacts, and columns must be the same.")
        sys.exit(1)

    if not all(impact in ['+', '-'] for impact in impacts):
        print("Error: Impacts must be either +ve or -ve.")
        sys.exit(1)

def calculate_weighted_normalized_matrix(normalized_matrix, weights):
    weighted_normalized_matrix = normalized_matrix * weights
    return weighted_normalized_matrix

def calculate_ideal_and_anti_ideal(weighted_normalized_matrix, impacts):
    is_maximize = [impact == '+' for impact in impacts]
    ideal_solution = np.max(weighted_normalized_matrix, axis=0) if is_maximize else np.min(weighted_normalized_matrix, axis=0)
    anti_ideal_solution = np.min(weighted_normalized_matrix, axis=0) if is_maximize else np.max(weighted_normalized_matrix, axis=0)

    return ideal_solution, anti_ideal_solution

def calculate_similarity_score(weighted_normalized_matrix, ideal_solution, anti_ideal_solution):
    similarity_score = np.sqrt(np.sum((weighted_normalized_matrix - ideal_solution)**2, axis=1)) / (
            np.sqrt(np.sum((weighted_normalized_matrix - ideal_solution)**2, axis=1)) +
            np.sqrt(np.sum((weighted_normalized_matrix - anti_ideal_solution)**2, axis=1))
    )
    return similarity_score

def save_rankings(result_file, headers, ranked_indices):
    with open(result_file, 'w') as file:
        file.write("Rankings:\n")
        for i, index in enumerate(ranked_indices):
            file.write(f"{i + 1}. {headers[index - 1]}\n")

def topsis(input_file, weights, impacts, result_file):
    validate_input_parameters(sys.argv)
    
    data = read_input_data(input_file)
    validate_input_data(data, weights, impacts)

    matrix = data.values[:, 1:].astype(float)
    headers = data.columns[1:]

    normalized_matrix = normalize_matrix(matrix)
    weighted_normalized_matrix = calculate_weighted_normalized_matrix(normalized_matrix, weights)
    ideal_solution, anti_ideal_solution = calculate_ideal_and_anti_ideal(weighted_normalized_matrix, impacts)
    similarity_score = calculate_similarity_score(weighted_normalized_matrix, ideal_solution, anti_ideal_solution)

    ranked_indices = np.argsort(similarity_score)[::-1] + 1  # Adding 1 to convert 0-based index to 1-based index

    save_rankings(result_file, headers, ranked_indices)

if __name__ == "__main__":
    topsis(sys.argv[1], list(map(float, sys.argv[2].split(','))), sys.argv[3], sys.argv[4])
