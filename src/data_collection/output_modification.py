import os
import re
import pandas as pd

def insert_line_before_last_n_lines(file_path, n, new_line):
    try:
        # Read the current contents of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Calculate the position to insert the new line
        insert_position = max(0, len(lines) - n)

        # Insert the new line at the calculated position
        lines.insert(insert_position, f"{new_line}\n")
        # Write everything back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)

        print(f"Line '{new_line}' inserted successfully before the last {n} lines in {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# file_path = 'test_123.txt'
# lines_before_last = 2
# new_line_to_insert = 'This is the new line1.\nThis is the new line2.'
# insert_line_before_last_n_lines(file_path, lines_before_last, new_line_to_insert)


def min_max_indices(lst):
    max_val = max(lst)
    min_val = min(lst)

    max_indices = [i for i, x in enumerate(lst) if x == max_val]
    min_indices = [i for i, x in enumerate(lst) if x == min_val]

    return (max_val, max_indices), (min_val, min_indices)


def csv_to_dict(filepath):
    # read the DataFrame from csv
    df = pd.read_csv(filepath)
    # convert the DataFrame to dictionary
    out_dict = df.to_dict(orient='list')  # or 'series', 'split', 'records', 'index' depending on your needs
    return out_dict


def csv2n_unitary(filepath):
    #filepath = f'/Users/sobhan/Desktop/EntgClass1/class {class_n}/case{case_n}.csv'  # replace with your file path
    my_dict = csv_to_dict(filepath)
    n_unitary = my_dict['n_unitary']
    # (max_val, max_indices), (min_val, min_indices) = min_max_indices(n_unitary)

    return min_max_indices(n_unitary)


def process_csv_files(directory_path):
    mins = []
    maxs = []
    try:
        # Get a list of all files in the directory
        all_files = os.listdir(directory_path)

        # Filter out only the CSV files
        csv_files = [file for file in all_files if file.endswith('.csv')]
        print(csv_files)
        # Process each CSV file
        for csv_file in csv_files:
            extracted_number = re.search(r'\d+(\.\d+)?', csv_file)
            i = int(extracted_number.group())
            file_path = os.path.join(directory_path, csv_file)
            (max_val, max_indices), (min_val, min_indices) = csv2n_unitary(file_path)
            mins.append((min_val, i, min_indices[0]))
            maxs.append((max_val, i, max_indices[0]))
        print("All CSV files processed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    sorted_mins = sorted(mins, key=lambda x: x[0])
    sorted_maxs = sorted(maxs, key=lambda x: x[0])
    return sorted_mins[0], sorted_maxs[-1]
# Example usage
# directory_path = 'dir_test'
# process_csv_files(directory_path)


def handle_class_n(n, directory_path=None):
    if directory_path is None:
        directory_path = f'/Users/sobhan/Desktop/EntgClass_v3/class {n}'
    x, y = process_csv_files(directory_path)
    print(x,y)
    str_to_add = f"best n_unitary {x[0]}: LC: {x[1]} iso: {x[2]}\nworst n_unitary {y[0]}: LC: {y[1]} iso: {y[2]}"
    file_path = directory_path+'/bests.txt'
    insert_line_before_last_n_lines(file_path, 2, str_to_add)


def delete_non_txt_csv_files(directory_path):
    try:
        # Get a list of all files in the directory
        all_files = os.listdir(directory_path)

        # Filter out only the .txt and .csv files
        txt_csv_files = [file for file in all_files if file.endswith(('.txt', '.csv'))]

        # Delete non .txt and .csv files
        for file in all_files:
            file_path = os.path.join(directory_path, file)
            if file not in txt_csv_files and os.path.isfile(file_path):
                os.remove(file_path)

    except Exception as e:
        print(f"An error occurred: {e}")


def find_and_delete_last_line(directory_path, file_extension='.txt'):
    try:
        # Get a list of all files in the directory
        all_files = os.listdir(directory_path)

        # Find the .txt file
        txt_files = [file for file in all_files if file.endswith(file_extension)]

        if not txt_files:
            print("No .txt file found in the directory.")
            return

        # Take the first .txt file (you can modify this logic if there are multiple)
        txt_file_path = os.path.join(directory_path, txt_files[0])

        # Read the contents of the file
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()

        # Remove the last line
        if lines:
            lines = lines[:-1]

        # Write the modified content back to the file
        with open(txt_file_path, 'w') as file:
            file.writelines(lines)

    except Exception as e:
        print(f"An error occurred: {e}")