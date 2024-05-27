location = "Dataset//"

with open(location + "Whookid_Africa.txt", "r", encoding="utf-8") as infile:
    lines = infile.readlines()

with open(location+"pre_process//output_Whookid_Africa.txt", "w", encoding="utf-8") as outfile:
    for line in lines:
        # Find the index of the first occurrence of "-"
        dash_index = line.find("-")
        if dash_index != -1:
            # Remove all text from 0 index to the found index (excluding the dash itself)
            modified_line = line[dash_index + 1:].strip()
            outfile.write(modified_line + "\n")
        else:
            # If no dash is found, write the entire line to the output file
            outfile.write(line)
