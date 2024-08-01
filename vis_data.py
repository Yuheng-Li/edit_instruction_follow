import json
import os 
import random
# Load the JSON data
image_root = "evalsample_real_img"
json_path = '/mnt/localssd/edit_instruction_follow_data/evalsample_real_img.json'
save_path = '/mnt/localssd/edit_instruction_follow_data/image_transformation_results.html'

with open(json_path, 'r') as file:
    data = json.load(file)
random.shuffle(data)
data = data[0:20]

# Create the HTML file
html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Transformation Results</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        img {
            max-width: 300px;  /* Set the desired maximum width here */
            height: auto;
        }
    </style>
</head>
<body>
    <h1>Image Transformation Results</h1>
    <table>
        <tr>
            <th>Instruction</th>
            <th>Before Image</th>
            <th>After Image</th>
            <th>Yes/No</th>
        </tr>
'''

# Populate the table rows with data from JSON
for item in data:
    instruction = item["conversations"][0]["value"].split("\n")[0].split(": ")[1].strip()
    before_image = os.path.join(image_root, item["image"][0])
    after_image = os.path.join(image_root, item["image"][1])
    yes_no = item["conversations"][1]["value"]

    html_content += f'''
        <tr>
            <td>{instruction}</td>
            <td><img src="{before_image}" alt="Before Image"></td>
            <td><img src="{after_image}" alt="After Image"></td>
            <td>{yes_no}</td>
        </tr>
    '''

# Close the HTML tags
html_content += '''
    </table>
</body>
</html>
'''

# Write the HTML content to a file
with open(save_path, 'w') as file:
    file.write(html_content)

print("HTML file has been created successfully.")
