import csv
import re

def process_data_to_csv(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file:
        data = input_file.read()
    
    # Initialize CSV file with headers
    with open(output_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["curr_attack", "mtd_freq", "compromised_num", "interrupted", "mtd"])

        # Split the data into blocks for each STATS section
        blocks = data.split("STATS BEFORE MTD OPERATION")[1:]
        
        for block in blocks:
            lines = block.strip().split("\n")
            curr_attack = re.search(r"Current Attack: (.+)", lines[0]).group(1)
            mtd_freq = re.search(r"MTD Frequency: (.+)", lines[1]).group(1)
            compromised_num = re.search(r"Compromised Number: (.+)", lines[2]).group(1)
            interrupted = "Interrupted" in block
            mtd = re.search(r"MTD: (\w+)", lines[3]).group(1)
            
            # Write the extracted data to the CSV
            csv_writer.writerow([curr_attack, mtd_freq, compromised_num, interrupted, mtd])

# Example usage
input_file_path = 'raw_data.txt'
output_file_path = 'output.csv'
process_data_to_csv(input_file_path, output_file_path)
