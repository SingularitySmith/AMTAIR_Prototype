# [AMTAIR Prototype Demonstration (Public Colab Notebook)](https://colab.research.google.com/github/SingularitySmith/AMTAIR_Prototype/blob/main/version_history/AMTAIR_Prototype_0_1.3.ipynb#scrollTo=lt8-AnebGUXr)

## Instructions --- How to use this notebook:


1. **Import Libraries & Install Packages**: Run Section 0.1 to set up the necessary dependencies for data processing and visualization.

2. **Connect to GitHub Repository & Load Data files**: Run Section 0.2 to establish connections to the data repository and load example datasets. This step retrieves sample ArgDown files and extracted data for demonstration.

3. **Process Source Documents to ArgDown**: Sections 1.0-1.8 demonstrate the extraction of argument structures from source documents (such as PDFs) into ArgDown format, a markdown-like notation for structured arguments.

4. **Convert ArgDown to BayesDown**: Sections 2.0-2.3 handle the transformation of ArgDown files into BayesDown format, which incorporates probabilistic information into the argument structure.

5. **Extract Data into Structured Format**: Section 3.0 processes BayesDown format into structured database entries (CSV) that can be used for analysis.

6. **Create and Analyze Bayesian Networks**: Section 4.0 demonstrates how to build Bayesian networks from the extracted data and provides tools for analyzing risk pathways.

7. **Save and Export Results**: Sections 5.0-6.0 provide methods for archiving results and exporting visualizations.



>[AMTAIR Prototype Demonstration (Public Colab Notebook)](#scrollTo=lt8-AnebGUXr)

>>[Instructions --- How to use this notebook:](#scrollTo=22NBzTxxsnfQ)

>>[Key Concepts:](#scrollTo=NovjnOw6bzLi)

>>[Example Workflow:](#scrollTo=NovjnOw6bzLi)

>>[Troubleshooting:](#scrollTo=NovjnOw6bzLi)

>[0.1 Import Libraries & Packages](#scrollTo=GtVFO-s74vI_)

>>[0.2 Connect to GitHub Repository](#scrollTo=2a3VR0fLhJow)

>>[0.3 File Import](#scrollTo=y-ix4Rp5fE9m)

>[1.0 Sources (PDF's of Papers) to ArgDown (.md file)](#scrollTo=52XyPlte5HrU)

>>[1.1 Specify Source Document (e.g. PDF)](#scrollTo=ESKnZ_4f_a6y)

>>[1.2 Generate ArgDown Extraction Prompt](#scrollTo=6ToQFra3_nl9)

>>[1.3 Prepare LLM API Call](#scrollTo=pGv2KcZU_9Bn)

>>[1.4 Make ArgDown Extraction LLM API Call](#scrollTo=i5xsDYnsAWC4)

>>[1.5 Save ArgDown Extraction Response](#scrollTo=Lc2nMp8nAfeU)

>>[1.6 Review and Check ArgDown.md File](#scrollTo=5HcCfqE4A0ht)

>>[1.6.2 Check the Graph Structure with the ArgDown Sandbox Online](#scrollTo=gSpkvLbCC_PI)

>>[1.7 Extract ArgDown Graph Information as DataFrame](#scrollTo=MAm0UKpeBvyr)

>>[1.8 Store ArgDown Information as 'ArgDown.csv' file](#scrollTo=iFC6oiyICREn)

>[2.0 Probability Extractions: ArgDown (.csv) to BayesDown (.md + plugin JSON syntax)](#scrollTo=7SGB0XMp5VFq)

>>[2.1 Generate and Extract "Prior-, Conditional- and Posterior Probability Questions" from ArgDown.csv](#scrollTo=V66ZHih3BTC0)

>>[2.2 Generate BayesDown Extraction Prompt](#scrollTo=5649brU2BTMh)

>>[2.3 Repeat Steps from 1.3 to 1.8 but for BayesDown / Probability Extraction](#scrollTo=uBCTJNNeDwuT)

>>[2.3 Converting ArgDown to BayesDown with Probability Extraction](#scrollTo=wF4W8y_C4ytX)

>>>[2.3.1 BayesDown Format Specification](#scrollTo=ivcnd2ml41Nv)

>[3.0 Data Extraction: BayesDown (.md) to Database (.csv)](#scrollTo=SJ9OIyEv5qqb)

>>>[3.1 ExtractBayesDown-Data_v1](#scrollTo=AFnu_1Ludahi)

>>[3.1.2 Test BayesDown Extraction](#scrollTo=eUBJh8Qp4yd4)

>>[3.1.2.2 Check the Graph Structure with the ArgDown Sandbox Online](#scrollTo=z4Hgs0ICDQyW)

>>>[3.1.2.B Test with 'Example_file_combined_withBayesDown_Crossgenerational.md'](#scrollTo=oSDF6M_h3h6O)

>>[3.3 Extraction](#scrollTo=mv8f4c4D3yJj)

>>>[3.3 Data-Post-Processing](#scrollTo=UcXf3fZ8dahj)

>>>[3.4 Download and save finished data frame as .csv file](#scrollTo=xTwPO_J-dahj)

>[4.0 Analysis & Inference: Practical Software Tools ()](#scrollTo=LHQm7ydMmPhN)

>>[Phase 1: Dependencies/Functions](#scrollTo=LSeSAPvtgIgU)

>>[Phase 2: Node Classification and Styling Module](#scrollTo=byAExfek5yFU)

>>[Phase 3: HTML Content Generation Module](#scrollTo=gnS3jFGU52OZ)

>>[Phase 4: Main Visualization Function](#scrollTo=d2uyG0Pi571f)

>[Quickly check HTML Outputs](#scrollTo=bFtxTKmLElSF)

>[5.0 Archive_version_histories](#scrollTo=0M9gFpK6ioHk)

>>>>>[Heading](#scrollTo=ulwM2lfJcY6g)

>[6.0 Save Outputs](#scrollTo=kjbIj19epbrF)

>>[Convert ipynb to HTML in Colab](#scrollTo=0QqlN6dYpm4s)

>>[Convert .ipynb Notebook to MarkDown](#scrollTo=pS6AhdiSCLw4)




## Key Concepts:

- **ArgDown**: A structured format for representing arguments, with hierarchical relationships between statements.
- **BayesDown**: An extension of ArgDown that incorporates probabilistic information, allowing for Bayesian network construction.
- **Extraction Pipeline**: The process of converting unstructured text to structured argument representations.
- **Bayesian Networks**: Probabilistic graphical models that represent variables and their conditional dependencies.

## Example Workflow:

1. Load a sample ArgDown file from the repository
2. Extract the hierarchical structure and relationships
3. Add probabilistic information to create a BayesDown representation
4. Generate a Bayesian network visualization
5. Analyze conditional probabilities and risk pathways

## Troubleshooting:

- If connectivity issues occur, ensure you have access to the GitHub repository
- For visualization errors, check that all required libraries are properly installed
- When processing custom files, ensure they follow the expected format conventions

# 0.1 Import Libraries & Packages



```python
!pip install pyvis
!pip install --upgrade gspread pandas google-auth google-colab

!pip install pgmpy
!pip install nbconvert
```


```python

import requests      # For making HTTP requests
import io           # For working with in-memory file-like objects

import pandas as pd   # For data manipulation
import numpy as np
import json
import re
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from IPython.display import Markdown, display

import networkx as nx
```


```python
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pyvis.network import Network
```

## 0.2 Connect to GitHub Repository

The Public GitHub Repo Url in use:

https://raw.githubusercontent.com/SingularitySmith/AMTAIR_Prototype/main/

Note:
When encountering errors, accessing the data, try using "RAW" Urls.


```python
# Specify the base repository URL
repo_url = "https://raw.githubusercontent.com/SingularitySmith/AMTAIR_Prototype/main/data/example_1/"

def load_file_from_repo(relative_path):
  """Loads a file from the specified GitHub repository using a relative path."""
  file_url = repo_url + relative_path
  response = requests.get(file_url)

  # Check for bad status codes and print more helpful error messages
  if response.status_code == 404:
    raise HTTPError(f"File not found at URL: {file_url}. Check the file path/name and ensure the file is publicly accessible.", response=response)
  else:
    response.raise_for_status() # Raise for other error codes

  file_object = io.StringIO(response.text)

  if relative_path.endswith(".csv"):
    return pd.read_csv(file_object)
  elif relative_path.endswith(".json"):
    return pd.read_json(file_object)
  elif relative_path.endswith(".md"):
    return file_object.read()  # Return the raw content for .md files
  else:
    raise ValueError("Unsupported file type. Add Support in GitHub Connection in the Second Section of this Python Notebook")

# Load files using relative paths

df = load_file_from_repo("extracted_data.csv") # Update if the file path is incorrect

md_content = load_file_from_repo("ArgDown_TestText.md")

# print(df.head()) # To see the output, run the code.

print(md_content) # To see the output, run the code.
```


```python
print(df.head()) # To see the output, run the code.
```

## 0.3 File Import


```python
md_content
```

# 1.0 Sources (PDF's of Papers) to ArgDown (.md file)

## 1.1 Specify Source Document (e.g. PDF)

Review the source document, ensure it is suitable for API call and upload to / store it in the correct location.

## 1.2 Generate ArgDown Extraction Prompt

Generate Extraction Prompt

## 1.3 Prepare LLM API Call

Combine Systemprompt + API Specifications + ArgDown Instructions + Prompt + Source PDF for API Call

## 1.4 Make ArgDown Extraction LLM API Call


```python

```

## 1.5 Save ArgDown Extraction Response

1. Save and log API return

2. Save ArgDown.md file for further Proecessing


```python

```

## 1.6 Review and Check ArgDown.md File


```python
display(Markdown(md_content))


```

## 1.6.2 Check the Graph Structure with the ArgDown Sandbox Online
Copy and paste the BayesDown formatted ... in the ArgDown Sandbox below to quickly verify that the network renders correctly.


```python
from IPython.display import IFrame

IFrame(src="https://argdown.org/sandbox/map/", width="100%", height="600px")
```

## 1.7 Extract ArgDown Graph Information as DataFrame

Extract:


*   Nodes (Variable_Title)
*   Edges (Parents)
*   Instantiations
*   Description

Implementation nodes:
- One function for ArgDown and BayesDown extraction, but:
- IF YOU ONLY WANT ARGDOWN EXTRACTION: USE ARGUMENT IN FUNCTION CALL "parse_markdown_hierarchy(markdown_text, ArgDown = True)"
- so if you set ArgDown = True, it gives you only instantiations, no probabilities.




```python
def parse_markdown_hierarchy(markdown_text, ArgDown = False):
    """Main function to parse markdown hierarchy into a DataFrame"""

    # Remove comments
    clean_text = remove_comments(markdown_text)

    # Extract all titles with their descriptions and indentation levels
    titles_info = extract_titles_info(clean_text)

    # Establish parent-child relationships
    titles_with_relations = establish_relationships(titles_info, clean_text)

    # Convert to DataFrame
    df = convert_to_dataframe(titles_with_relations, ArgDown)

    # Add No_Parent and No_Children columns
    df = add_no_parent_no_child_columns_to_df(df)

    # Add Parents instantiation columns
    df = add_parents_instantiation_columns_to_df(df)

    return df

def remove_comments(markdown_text):
    """Remove comment blocks from markdown text"""
    return re.sub(r'/\*.*?\*/', '', markdown_text, flags=re.DOTALL)

def extract_titles_info(text):
    """Extract titles with their descriptions and indentation levels"""
    lines = text.split('\n')
    titles_info = {}

    for line in lines:
        if not line.strip():
            continue

        title_match = re.search(r'[<\[](.+?)[>\]]', line)
        if not title_match:
            continue

        title = title_match.group(1)

        # Extract description and metadata
        title_pattern_in_line = r'[<\[]' + re.escape(title) + r'[>\]]:'
        description_match = re.search(title_pattern_in_line + r'\s*(.*)', line)

        if description_match:
            full_text = description_match.group(1).strip()

            # Check if description contains a "{" to not include metadata in description
            if "{" in full_text:
                # Split at the first "{"
                split_index = full_text.find("{")
                description = full_text[:split_index].strip()
                metadata = full_text[split_index:].strip()
            else:
                # Keep the entire description and no metadata
                description = full_text
                metadata = ''
        else:
            description = ''
            metadata = ''  # Ensure metadata is initialized as empty string

        indentation = 0
        if '+' in line:
            symbol_index = line.find('+')
            # Count spaces before the '+' symbol
            i = symbol_index - 1
            while i >= 0 and line[i] == ' ':
                indentation += 1
                i -= 1
        elif '-' in line:
            symbol_index = line.find('-')
            # Count spaces before the '-' symbol
            i = symbol_index - 1
            while i >= 0 and line[i] == ' ':
                indentation += 1
                i -= 1

        # If neither symbol exists, indentation remains 0

        if title in titles_info:
            # Only update description if it's currently empty and we found a new one
            if not titles_info[title]['description'] and description:
                titles_info[title]['description'] = description

            # Store all indentation levels for this title
            titles_info[title]['indentation_levels'].append(indentation)

            # Keep max indentation for backward compatibility
            if indentation > titles_info[title]['indentation']:
                titles_info[title]['indentation'] = indentation

            # Do NOT update metadata here - keep the original metadata
        else:
            # First time seeing this title, create a new entry
            titles_info[title] = {
                'description': description,
                'indentation': indentation,
                'indentation_levels': [indentation],  # Initialize with first indentation level
                'parents': [],
                'children': [],
                'line': None,
                'line_numbers': [],  # Initialize an empty list for all occurrences
                'metadata': metadata  # Set metadata explicitly from what we found
            }

    return titles_info

def establish_relationships(titles_info, text):
    """Establish parent-child relationships between titles using the BayesDown indentation rules"""
    lines = text.split('\n')

    # Dictionary to store line numbers for each title occurrence
    title_occurrences = {}

    # Record line number for each title (including multiple occurrences)
    line_number = 0
    for line in lines:
        if not line.strip():
            line_number += 1
            continue

        title_match = re.search(r'[<\[](.+?)[>\]]', line)
        if not title_match:
            line_number += 1
            continue

        title = title_match.group(1)

        # Store all occurrences of each title with their line numbers
        if title not in title_occurrences:
            title_occurrences[title] = []
        title_occurrences[title].append(line_number)

        # Store all line numbers where this title appears
        if 'line_numbers' not in titles_info[title]:
            titles_info[title]['line_numbers'] = []
        titles_info[title]['line_numbers'].append(line_number)

        # For backward compatibility, keep the first occurrence in 'line'
        if titles_info[title]['line'] is None:
            titles_info[title]['line'] = line_number

        line_number += 1

    # Create an ordered list of all title occurrences with their line numbers
    all_occurrences = []
    for title, occurrences in title_occurrences.items():
        for line_num in occurrences:
            all_occurrences.append((title, line_num))

    # Sort occurrences by line number
    all_occurrences.sort(key=lambda x: x[1])

    # Get indentation for each occurrence
    occurrence_indents = {}
    for title, line_num in all_occurrences:
        for line in lines[line_num:line_num+1]:  # Only check the current line
            indent = 0
            if '+' in line:
                symbol_index = line.find('+')
                # Count spaces before the '+' symbol
                j = symbol_index - 1
                while j >= 0 and line[j] == ' ':
                    indent += 1
                    j -= 1
            elif '-' in line:
                symbol_index = line.find('-')
                # Count spaces before the '-' symbol
                j = symbol_index - 1
                while j >= 0 and line[j] == ' ':
                    indent += 1
                    j -= 1
            occurrence_indents[(title, line_num)] = indent

    # Process for finding parents (looking forward)
    for i, (title, line_num) in enumerate(all_occurrences):
        current_indent = occurrence_indents[(title, line_num)]

        # Look ahead for potential parents that are exactly one indentation level higher
        j = i + 1
        while j < len(all_occurrences):
            next_title, next_line = all_occurrences[j]
            next_indent = occurrence_indents[(next_title, next_line)]

            # If we find a title with same or less indentation, stop looking in this section
            if next_indent <= current_indent:
                break

            # If this is a direct parent (exactly one more indentation) and not the same title
            if next_indent == current_indent + 1 and next_title != title:
                # More indented node is parent of less indented node
                if next_title not in titles_info[title]['parents']:
                    titles_info[title]['parents'].append(next_title)
                if title not in titles_info[next_title]['children']:
                    titles_info[next_title]['children'].append(title)

            j += 1

    # Process for finding children (looking backward)
    for i, (title, line_num) in enumerate(all_occurrences):
        current_indent = occurrence_indents[(title, line_num)]

        # Skip titles with indentation 0 (they don't have children by looking backward)
        if current_indent == 0:
            continue

        # Look for the immediately preceding title with one less indentation (immediate child)
        j = i - 1
        found_child = False

        while j >= 0 and not found_child:
            prev_title, prev_line = all_occurrences[j]
            prev_indent = occurrence_indents[(prev_title, prev_line)]

            # If the previous title has exactly one less indentation and is not the same title
            if prev_indent == current_indent - 1 and prev_title != title:
                # Current title is parent of previous title
                if title not in titles_info[prev_title]['parents']:
                    titles_info[prev_title]['parents'].append(title)
                if prev_title not in titles_info[title]['children']:
                    titles_info[title]['children'].append(prev_title)
                found_child = True  # Only find one immediate child

            # If we encounter a title with even less indentation, stop looking
            if prev_indent < current_indent - 1:
                break

            j -= 1

    return titles_info

def convert_to_dataframe(titles_info, ArgDown):
    """Convert the titles information dictionary to a pandas DataFrame"""
    if ArgDown == True:
        df = pd.DataFrame(columns=['Title', 'Description', 'line', 'line_numbers', 'indentation',
                               'indentation_levels', 'Parents', 'Children', 'instantiations'])
    else:
        df = pd.DataFrame(columns=['Title', 'Description', 'line', 'line_numbers', 'indentation',
                               'indentation_levels', 'Parents', 'Children', 'instantiations',
                               'priors', 'posteriors'])

    for title, info in titles_info.items():
        # Parse the metadata JSON string into a Python dictionary
        if 'metadata' in info and info['metadata']:
            try:
                # Only try to parse if metadata is not empty
                if info['metadata'].strip():
                    jsonMetadata = json.loads(info['metadata'])
                    if ArgDown == True:
                        # Create the row dictionary with instantitions as metadata only, no probabilites yet
                        row = {
                            'Title': title,
                            'Description': info.get('description', ''),
                            'line': info.get('line',''),
                            'line_numbers': info.get('line_numbers', []),
                            'indentation': info.get('indentation',''),
                            'indentation_levels': info.get('indentation_levels', []),
                            'Parents': info.get('parents', []),
                            'Children': info.get('children', []),
                            # Extract specific metadata fields, defaulting to empty if not present
                            'instantiations': jsonMetadata.get('instantiations', []),
                        }


                    else:
                        # create dict with probabilites
                        row = {
                            'Title': title,
                            'Description': info.get('description', ''),
                            'line': info.get('line',''),
                            'line_numbers': info.get('line_numbers', []),
                            'indentation': info.get('indentation',''),
                            'indentation_levels': info.get('indentation_levels', []),
                            'Parents': info.get('parents', []),
                            'Children': info.get('children', []),
                            # Extract specific metadata fields, defaulting to empty if not present
                            'instantiations': jsonMetadata.get('instantiations', []),
                            'priors': jsonMetadata.get('priors', {}),
                            'posteriors': jsonMetadata.get('posteriors', {})
                        }
                else:
                    # Empty metadata case
                    row = {
                        'Title': title,
                        'Description': info.get('description', ''),
                        'line': info.get('line',''),
                        'line_numbers': info.get('line_numbers', []),
                        'indentation': info.get('indentation',''),
                        'indentation_levels': info.get('indentation_levels', []),
                        'Parents': info.get('parents', []),
                        'Children': info.get('children', []),
                        'instantiations': [],
                        'priors': {},
                        'posteriors': {}
                    }
            except json.JSONDecodeError:
                # Handle case where metadata isn't valid JSON
                row = {
                    'Title': title,
                    'Description': info.get('description', ''),
                    'line': info.get('line',''),
                    'line_numbers': info.get('line_numbers', []),
                    'indentation': info.get('indentation',''),
                    'indentation_levels': info.get('indentation_levels', []),
                    'Parents': info.get('parents', []),
                    'Children': info.get('children', []),
                    'instantiations': [],
                    'priors': {},
                    'posteriors': {}
                }
        else:
            # Handle case where metadata field doesn't exist or is empty
            row = {
                'Title': title,
                'Description': info.get('description', ''),
                'line': info.get('line',''),
                'line_numbers': info.get('line_numbers', []),
                'indentation': info.get('indentation',''),
                'indentation_levels': info.get('indentation_levels', []),
                'Parents': info.get('parents', []),
                'Children': info.get('children', []),
                'instantiations': [],
                'priors': {},
                'posteriors': {}
            }

        # Add the row to the DataFrame
        df.loc[len(df)] = row

    return df

def add_no_parent_no_child_columns_to_df(dataframe):
    """Add No_Parent and No_Children boolean columns to the DataFrame"""
    no_parent = []
    no_children = []

    for _, row in dataframe.iterrows():
        no_parent.append(not row['Parents'])
        no_children.append(not row['Children'])

    dataframe['No_Parent'] = no_parent
    dataframe['No_Children'] = no_children

    return dataframe

def add_parents_instantiation_columns_to_df(dataframe):
    """Add all possible instantiations of all parents as list with lists column to the DataFrame"""
    # Create a new column to store parent instantiations
    parent_instantiations = []

    # Iterate through each row in the dataframe
    for _, row in dataframe.iterrows():
        parents = row['Parents']
        parent_insts = []

        # For each parent, find its instantiations and add to the list
        for parent in parents:
            # Find the row where Title matches the parent
            parent_row = dataframe[dataframe['Title'] == parent]

            # If parent found in the dataframe
            if not parent_row.empty:
                # Get the instantiations of this parent
                parent_instantiation = parent_row['instantiations'].iloc[0]
                parent_insts.append(parent_instantiation)

        # Add the list of parent instantiations to our new column
        parent_instantiations.append(parent_insts)

    # Add the new column to the dataframe
    dataframe['parent_instantiations'] = parent_instantiations

    return dataframe


```


```python
# example use case:
ex_csv = parse_markdown_hierarchy(md_content, ArgDown = True)
ex_csv
```

## 1.8 Store ArgDown Information as 'ArgDown.csv' file


```python
# Assuming 'md_content' holds the markdown text
# Store the results of running the function parse_markdown_hierarchy(md_content, ArgDown = True) as the file 'ArgDown.csv'
result_df = parse_markdown_hierarchy(md_content, ArgDown = True)

# Save to CSV
result_df.to_csv('ArgDown.csv', index=False)
```


```python
# Test if 'ArgDown.csv' has been saved correctly with the correct information
# Load the data from the CSV file
argdown_df = pd.read_csv('ArgDown.csv')

# Display the DataFrame
print(argdown_df)
```

# 2.0 Probability Extractions: ArgDown (.csv) to BayesDown (.md + plugin JSON syntax)


```python

```

## 2.1 Generate and Extract "Prior-, Conditional- and Posterior Probability Questions" from 'ArgDown.csv' to '"ArgDown_WithQuestions.csv"'


```python
import pandas as pd
import re
import json
import itertools
from IPython.display import Markdown, display



def parse_instantiations(instantiations_str):
    """
    Parse instantiations from string or list format.
    Handles various input formats flexibly.
    """
    if pd.isna(instantiations_str) or instantiations_str == '':
        return []

    if isinstance(instantiations_str, list):
        return instantiations_str

    try:
        # Try to parse as JSON
        return json.loads(instantiations_str)
    except:
        # Try to parse as string list
        if isinstance(instantiations_str, str):
            # Remove brackets and split by comma
            clean_str = instantiations_str.strip('[]"\'')
            if not clean_str:
                return []
            return [s.strip(' "\'') for s in clean_str.split(',') if s.strip()]

    return []

def parse_parents(parents_str):
    """
    Parse parents from string or list format.
    Handles various input formats flexibly.
    """
    if pd.isna(parents_str) or parents_str == '':
        return []

    if isinstance(parents_str, list):
        return parents_str

    try:
        # Try to parse as JSON
        return json.loads(parents_str)
    except:
        # Try to parse as string list
        if isinstance(parents_str, str):
            # Remove brackets and split by comma
            clean_str = parents_str.strip('[]"\'')
            if not clean_str:
                return []
            return [s.strip(' "\'') for s in clean_str.split(',') if s.strip()]

    return []

def get_parent_instantiations(parent, df):
    """
    Get the instantiations for a parent node from the DataFrame.
    Returns default instantiations if not found.
    """
    parent_row = df[df['Title'] == parent]
    if parent_row.empty:
        return [f"{parent}_TRUE", f"{parent}_FALSE"]

    instantiations = parse_instantiations(parent_row.iloc[0]['instantiations'])
    if not instantiations:
        return [f"{parent}_TRUE", f"{parent}_FALSE"]

    return instantiations

def generate_instantiation_questions(title, instantiation, parents, df):
    """
    Generate questions for a specific instantiation of a node.

    Args:
        title (str): The title of the node
        instantiation (str): The specific instantiation (e.g., "title_TRUE")
        parents (list): List of parent nodes
        df (DataFrame): The full DataFrame for looking up parent instantiations

    Returns:
        dict: Dictionary mapping question types to questions
    """
    questions = {}

    # If no parents, just generate a prior probability question
    if not parents:
        prior_question = f"What is the probability for {title}={instantiation}?"
        questions['prior'] = prior_question
        return questions

    # For nodes with parents, generate conditional probability questions
    # Get all combinations of parent instantiations
    parent_instantiations = []
    for parent in parents:
        parent_insts = get_parent_instantiations(parent, df)
        parent_instantiations.append([(parent, inst) for inst in parent_insts])

    # Generate all combinations
    all_combinations = list(itertools.product(*parent_instantiations))

    # Create conditional probability questions for each combination
    for i, combination in enumerate(all_combinations):
        condition_str = ", ".join([f"{parent}={inst}" for parent, inst in combination])
        question = f"What is the probability for {title}={instantiation} if {condition_str}?"
        questions[f'conditional_{i}'] = question

    return questions

def generate_argdown_with_questions(argdown_csv_path, output_csv_path):
    """
    Generate probability questions based on the ArgDown CSV file and save to a new CSV file.

    Args:
        argdown_csv_path (str): Path to the input ArgDown CSV file
        output_csv_path (str): Path to save the output CSV file with questions
    """
    print(f"Loading ArgDown CSV from {argdown_csv_path}...")

    # Load the ArgDown CSV file
    try:
        df = pd.read_csv(argdown_csv_path)
        print(f"Successfully loaded CSV with {len(df)} rows.")
    except Exception as e:
        raise Exception(f"Error loading ArgDown CSV: {e}")

    # Validate required columns
    required_columns = ['Title', 'Parents', 'instantiations']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise Exception(f"Missing required columns: {', '.join(missing_columns)}")

    # Initialize columns for questions
    df['Generate_Positive_Instantiation_Questions'] = None
    df['Generate_Negative_Instantiation_Questions'] = None

    print("Generating probability questions for each node...")

    # Process each row to generate questions
    for idx, row in df.iterrows():
        title = row['Title']
        instantiations = parse_instantiations(row['instantiations'])
        parents = parse_parents(row['Parents'])

        if len(instantiations) < 2:
            # Default instantiations if not provided
            instantiations = [f"{title}_TRUE", f"{title}_FALSE"]

        # Generate positive instantiation questions
        positive_questions = generate_instantiation_questions(title, instantiations[0], parents, df)

        # Generate negative instantiation questions
        negative_questions = generate_instantiation_questions(title, instantiations[1], parents, df)

        # Update the DataFrame
        df.at[idx, 'Generate_Positive_Instantiation_Questions'] = json.dumps(positive_questions)
        df.at[idx, 'Generate_Negative_Instantiation_Questions'] = json.dumps(negative_questions)

    # Save the enhanced DataFrame
    df.to_csv(output_csv_path, index=False)
    print(f"Generated questions saved to {output_csv_path}")

    return df

# Example usage:
df_with_questions = generate_argdown_with_questions("ArgDown.csv", "ArgDown_WithQuestions.csv")
df_with_questions
```


```python
# Load the data from the ArgDown_WithQuestions CSV file
argdown_with_questions_df = pd.read_csv('ArgDown_WithQuestions.csv')

# Display the DataFrame
print(argdown_with_questions_df)
```


```python
# example use case:
ArgDown_WithQuestions_csv = demonstrate_question_generation("ArgDown.csv")
ArgDown_WithQuestions_csv
```

## 2.2 Save BayesDown Extraction Questions as 'BayesDownQuestions.md'


```python
import pandas as pd
import json
from IPython.display import Markdown, display


def generate_bayesdown_questions_md(argdown_with_questions_path, output_md_path, QuestionsMinimal=False):
    """
    Generate comprehensive BayesDown questions based on the enhanced CSV file.

    Args:
        argdown_with_questions_path (str): Path to the CSV file with probability questions
        output_md_path (str): Path to save the output markdown file
        QuestionsMinimal (bool, optional): If True, only return the questions generated for each node,
                                           excluding terminology explanations. Defaults to False.
    """
    print(f"Loading enhanced CSV from {argdown_with_questions_path}...")

    # Load the enhanced CSV file
    try:
        df = pd.read_csv(argdown_with_questions_path)
        print(f"Successfully loaded CSV with {len(df)} rows.")
    except Exception as e:
        raise Exception(f"Error loading CSV: {e}")

    # Validate required columns
    required_columns = ['Title', 'Generate_Positive_Instantiation_Questions', 'Generate_Negative_Instantiation_Questions']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise Exception(f"Missing required columns: {', '.join(missing_columns)}")

    print("Generating comprehensive BayesDown questions...")

    # Start building the markdown content
    md_content = ""  # Initialize as empty string

    if not QuestionsMinimal:
        md_content += "# BayesDown Probability Questions\n\n"
        md_content += "This document contains questions for extracting probability estimates for BayesDown models.\n\n"

        # Add comprehensive terminology explanation
        md_content += "## Probability Terminology\n\n"


        md_content += "### Types of Probabilities\n\n"
        md_content += "- **Prior Probability**: The unconditional probability of a variable having a specific value before considering any evidence or parent variable states. For example, P(X=TRUE) represents the probability that X is TRUE without any additional information.\n\n"
        md_content += "- **Conditional Probability**: The probability of a variable having a specific value given the values of its parent variables. For example, P(X=TRUE|Y=TRUE, Z=FALSE) represents the probability that X is TRUE when we know that Y is TRUE and Z is FALSE.\n\n"
        md_content += "- **Posterior Probability**: The updated probability of a hypothesis after considering new evidence, calculated using Bayes' theorem. This represents a revised belief based on additional information.\n\n"
        md_content += "- **Joint Probability**: The probability of multiple events occurring together. For example, P(X=TRUE, Y=FALSE) represents the probability that X is TRUE and Y is FALSE simultaneously.\n\n"
        md_content += "- **Marginal Probability**: The probability of an event across all possible states of another variable. It can be calculated by summing the joint probability over all possible values of the other variables.\n\n"

        md_content += "### Source of Probability Estimates\n\n"
        md_content += "For each probability estimate, please identify the source using one of the following categories:\n\n"
        md_content += "- **Direct Statement**: The probability is explicitly stated in the text.\n"
        md_content += "- **Derived Estimate**: The probability is calculated or inferred from other probabilities mentioned in the text.\n"
        md_content += "- **Context-Based Estimate**: The probability is inferred from the general context, tone, or strength of assertions in the text.\n"
        md_content += "- **Expert Judgment**: The probability is based on domain expertise, not directly stated in the text.\n"
        md_content += "- **Default Assignment**: The probability is assigned a reasonable default value due to lack of information.\n\n"

        md_content += "### Certainty of Estimates\n\n"
        md_content += "For each probability estimate, please assess your certainty using one of the following approaches:\n\n"
        md_content += "- **Confidence Interval**: Provide a range that likely contains the true probability (e.g., \"80% confidence interval: 0.3-0.5\").\n"
        md_content += "- **Confidence Level**: Rate your confidence in the estimate on a scale (e.g., \"High confidence: 85%\").\n"
        md_content += "- **Error Margin**: Specify how much the estimate might vary (e.g., \"0.7 ± 0.1\").\n"
        md_content += "- **Qualitative Assessment**: Describe your certainty qualitatively (e.g., \"Very certain\", \"Moderately certain\", \"Highly uncertain\").\n\n"


    # Generate questions for each node
    for idx, row in df.iterrows():
        title = row['Title']
        description = row['Description'] if 'Description' in df.columns and not pd.isna(row['Description']) else ""

        md_content += f"## {title}\n\n"  # Still include title even in minimal mode

        if description:
            md_content += f"{description}\n\n"

        # Process positive instantiation questions
        try:
            positive_questions = json.loads(row['Generate_Positive_Instantiation_Questions'])

            md_content += "### Positive Instantiation Questions\n\n"

            for q_type, question in positive_questions.items():
                md_content += f"1. **{question}**\n"

                # Add source question with appropriate terminology based on question type
                if q_type == 'prior':
                    md_content += f"   - **Source**: What is the source for this prior probability estimate? (Direct statement, derived estimate, context-based, expert judgment, default)\n"
                else:
                    md_content += f"   - **Source**: What is the source for this conditional probability estimate? (Direct statement, derived estimate, context-based, expert judgment, default)\n"

                # Add certainty question
                md_content += f"   - **Certainty**: How certain are you about this probability estimate? (Provide a confidence interval, confidence level, error margin, or qualitative assessment)\n\n"
        except Exception as e:
            md_content += f"No positive instantiation questions available. Error: {e}\n\n"

        # Process negative instantiation questions
        try:
            negative_questions = json.loads(row['Generate_Negative_Instantiation_Questions'])

            md_content += "### Negative Instantiation Questions\n\n"

            for q_type, question in negative_questions.items():
                md_content += f"1. **{question}**\n"

                # Add source question with appropriate terminology based on question type
                if q_type == 'prior':
                    md_content += f"   - **Source**: What is the source for this prior probability estimate? (Direct statement, derived estimate, context-based, expert judgment, default)\n"
                else:
                    md_content += f"   - **Source**: What is the source for this conditional probability estimate? (Direct statement, derived estimate, context-based, expert judgment, default)\n"

                # Add certainty question
                md_content += f"   - **Certainty**: How certain are you about this probability estimate? (Provide a confidence interval, confidence level, error margin, or qualitative assessment)\n\n"
        except Exception as e:
            md_content += f"No negative instantiation questions available. Error: {e}\n\n"

    # Save the markdown content
    with open(output_md_path, 'w') as f:
        f.write(md_content)

    print(f"BayesDown questions saved to {output_md_path}")
    return md_content

# Example usage:
md_content = generate_bayesdown_questions_md("ArgDown_WithQuestions.csv", "BayesDownQuestions.md", QuestionsMinimal=True)
# To get only the questions, set QuestionsMinimal=True

print(md_content)  # Print the returned content to see the questions
```


```python
# Explicitly set the value of QuestionsMinimal
QuestionsMinimal = True  # or False, depending on your needs

# Get the markdown content
md_content = generate_bayesdown_questions_md("ArgDown_WithQuestions.csv", "BayesDownQuestions.md", QuestionsMinimal=QuestionsMinimal)

# Determine the output file path based on QuestionsMinimal
if QuestionsMinimal:
    output_file_path = "BayesDownQuestions.md"
else:
    output_file_path = "FULL_BayesDownQuestions.md"

# Save the markdown content to the appropriate file
with open(output_file_path, 'w') as f:
    f.write(md_content)

print(f"Markdown content saved to {output_file_path}")
```


```python
# Load and print the content of the 'BayesDownQuestions.md' file
with open("BayesDownQuestions.md", "r") as f:
    file_content = f.read()
    print(file_content)
```

## 2.2 Generate BayesDown Extraction Prompt

Generate 2nd Extraction Prompt for Probabilities based on the questions generated from the 'ArgDown.csv' extraction


```python

```

## 2.3 Repeat Steps from 1.3 to 1.8 but for BayesDown / Probability Extraction

## 2.3 Converting ArgDown to BayesDown with Probability Extraction

BayesDown extends the ArgDown format by incorporating probabilistic information about arguments and their relationships. This section demonstrates how to transform an ArgDown representation into BayesDown by:

1. **Extracting probability statements** from the text
2. **Formalizing conditional relationships** between variables
3. **Quantifying uncertainty** in argument strength and variable states



### 2.3.1 BayesDown Format Specification

BayesDown augments ArgDown with probability data in a structured JSON format:

```json
{
  "instantiations": ["state_TRUE", "state_FALSE"],
  "priors": {
    "p(state_TRUE)": "0.7",
    "p(state_FALSE)": "0.3"
  },
  "posteriors": {
    "p(state_TRUE|condition1_TRUE,condition2_FALSE)": "0.9",
    "p(state_TRUE|condition1_FALSE,condition2_TRUE)": "0.4"
  }
}

2.3.2 Probability Extraction Process
The probability extraction pipeline follows these steps:


Identify variables and their possible states
Extract prior probability statements
Identify conditional relationships
Extract conditional probability statements
Format the data in BayesDown syntax

2.3.3 Implementation Steps
To extract probabilities and create BayesDown format:

Run the extract_probabilities function on ArgDown text
Process the results into a structured format
Validate the probability distributions (ensure they sum to 1)
Generate the enhanced BayesDown representation

2.3.4 Validation and Quality Control
The probability extraction process includes validation steps:

Ensuring coherent probability distributions
Checking for logical consistency in conditional relationships
Verifying that all required probability statements are present
Handling missing data with appropriate default values

# 3.0 Data Extraction: BayesDown (.md) to Database (.csv)


### 3.1 ExtractBayesDown-Data_v1
Build data frame with extractable information from BayesDown


```python
# read sprinkler example -- Occam Colab Online
file_path_ex_rain = "https://raw.githubusercontent.com/SingularitySmith/AMTAIR_Prototype/main/data/example_1/BayesDown_Example.md"

# Use requests.get to fetch content from URL
response = requests.get(file_path_ex_rain)
response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

# Read content from the response
md_content_ex_rain = response.text

md_content_ex_rain
```

## 3.1.2 Test BayesDown Extraction





```python
display(Markdown(md_content_ex_rain)) # view BayesDown file formatted as MarkDown
```

## 3.1.2.2 Check the Graph Structure with the ArgDown Sandbox Online
Copy and paste the BayesDown formatted ... in the ArgDown Sandbox below to quickly verify that the network renders correctly.

### 3.1.2.B Test with 'Example_file_combined_withBayesDown_Crossgenerational.md'


```python
# read basic ArgDown example With BayesDown syntax added and corss generational added
import requests  # Import the requests library

# **Corrected URL with /main/**
file_path_easy_ex_B_CG = "https://raw.githubusercontent.com/SingularitySmith/AMTAIR_Prototype/main/Example_file_combined_withBayesDown_Crossgenerational.md"

# Use requests.get to fetch content from URL
response = requests.get(file_path_easy_ex_B_CG)
response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

# Read content from the response
md_content_easy_ex_B_CG = response.text

md_content_easy_ex_B_CG
```

## 3.3 Extraction
BayesDown Extraction Code already part of ArgDown extraction code, therefore just use same function "parse_markdown_hierarchy(markdown_data)" and ignore the extra argument ("ArgDown") because it is automatically set to false amd will by default extract BayesDown.


```python
result_df = parse_markdown_hierarchy(md_content_ex_rain)
result_df
```

### 3.3 Data-Post-Processing
Add rows to data frame that can be calculated from the extracted rows


```python
# here we add all the rows that we have to calculate (joint probability..., maybe in several rounds (e.g. first add conditional proability, then use this column to calc joint probability...)
```

### 3.4 Download and save finished data frame as .csv file


```python
result_df.to_csv('extracted_data.csv', index=False) # save dataframe in environment as .csv file
# Attention: if the new or updated .csv file is required later, it needs to be pushed to the GitRepository!
```

# 4.0 Analysis & Inference: Practical Software Tools ()

## Phase 1: Dependencies/Functions


```python
from pyvis.network import Network
import networkx as nx
from IPython.display import HTML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import colorsys
import json

def create_bayesian_network_with_probabilities(df):
    """
    Create an interactive Bayesian network visualization with enhanced probability visualization
    and node classification based on network structure.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with proper attributes
    for idx, row in df.iterrows():
        title = row['Title']
        description = row['Description']

        # Process probability information
        priors = get_priors(row)
        instantiations = get_instantiations(row)

        # Add node with base information
        G.add_node(
            title,
            description=description,
            priors=priors,
            instantiations=instantiations,
            posteriors=get_posteriors(row)
        )

    # Add edges
    for idx, row in df.iterrows():
        child = row['Title']
        parents = get_parents(row)

        # Add edges from each parent to this child
        for parent in parents:
            if parent in G.nodes():
                G.add_edge(parent, child)

    # Classify nodes based on network structure
    classify_nodes(G)

    # Create network visualization
    net = Network(notebook=True, directed=True, cdn_resources="in_line", height="600px", width="100%")

    # Configure physics for better layout
    net.force_atlas_2based(gravity=-50, spring_length=100, spring_strength=0.02)
    net.show_buttons(filter_=['physics'])

    # Add the graph to the network
    net.from_nx(G)

    # Enhance node appearance with probability information and classification
    for node in net.nodes:
        node_id = node['id']
        node_data = G.nodes[node_id]

        # Get node type and set border color
        node_type = node_data.get('node_type', 'unknown')
        border_color = get_border_color(node_type)

        # Get probability information
        priors = node_data.get('priors', {})
        true_prob = priors.get('true_prob', 0.5) if priors else 0.5

        # Get proper state names
        instantiations = node_data.get('instantiations', ["TRUE", "FALSE"])
        true_state = instantiations[0] if len(instantiations) > 0 else "TRUE"
        false_state = instantiations[1] if len(instantiations) > 1 else "FALSE"

        # Create background color based on probability
        background_color = get_probability_color(priors)

        # Create tooltip with probability information
        tooltip = create_tooltip(node_id, node_data)

        # Create a simpler node label with probability
        simple_label = f"{node_id}\np={true_prob:.2f}"

        # Store expanded content as a node attribute for use in click handler
        node_data['expanded_content'] = create_expanded_content(node_id, node_data)

        # Set node attributes
        node['title'] = tooltip  # Tooltip HTML
        node['label'] = simple_label  # Simple text label
        node['shape'] = 'box'
        node['color'] = {
            'background': background_color,
            'border': border_color,
            'highlight': {
                'background': background_color,
                'border': border_color
            }
        }

    # Set up the click handler with proper data
    setup_data = {
        'nodes_data': {node_id: {
            'expanded_content': json.dumps(G.nodes[node_id].get('expanded_content', '')),
            'description': G.nodes[node_id].get('description', ''),
            'priors': G.nodes[node_id].get('priors', {}),
            'posteriors': G.nodes[node_id].get('posteriors', {})
        } for node_id in G.nodes()}
    }

    # Add custom click handling JavaScript
    click_js = """
    // Store node data for click handling
    var nodesData = %s;

    // Add event listener for node clicks
    network.on("click", function(params) {
        if (params.nodes.length > 0) {
            var nodeId = params.nodes[0];
            var nodeInfo = nodesData[nodeId];

            if (nodeInfo) {
                // Create a modal popup for expanded content
                var modal = document.createElement('div');
                modal.style.position = 'fixed';
                modal.style.left = '50%%';
                modal.style.top = '50%%';
                modal.style.transform = 'translate(-50%%, -50%%)';
                modal.style.backgroundColor = 'white';
                modal.style.padding = '20px';
                modal.style.borderRadius = '5px';
                modal.style.boxShadow = '0 0 10px rgba(0,0,0,0.5)';
                modal.style.zIndex = '1000';
                modal.style.maxWidth = '80%%';
                modal.style.maxHeight = '80%%';
                modal.style.overflow = 'auto';

                // Add expanded content
                modal.innerHTML = nodeInfo.expanded_content || 'No detailed information available';

                // Add close button
                var closeBtn = document.createElement('button');
                closeBtn.innerHTML = 'Close';
                closeBtn.style.marginTop = '10px';
                closeBtn.style.padding = '5px 10px';
                closeBtn.style.cursor = 'pointer';
                closeBtn.onclick = function() {
                    document.body.removeChild(modal);
                };
                modal.appendChild(closeBtn);

                // Add modal to body
                document.body.appendChild(modal);
            }
        }
    });
    """ % json.dumps(setup_data['nodes_data'])

    # Save the graph to HTML
    html_file = "bayesian_network.html"
    net.save_graph(html_file)

    # Inject custom click handling into HTML
    try:
        with open(html_file, "r") as f:
            html_content = f.read()

        # Insert click handling script before the closing body tag
        html_content = html_content.replace('</body>', f'<script>{click_js}</script></body>')

        # Write back the modified HTML
        with open(html_file, "w") as f:
            f.write(html_content)

        return HTML(html_content)
    except Exception as e:
        return HTML(f"<p>Error rendering HTML: {str(e)}</p><p>The network visualization has been saved to '{html_file}'</p>")

def classify_nodes(G):
    """
    Classify nodes as parent, child, or leaf based on network structure
    """
    for node in G.nodes():
        predecessors = list(G.predecessors(node))
        successors = list(G.successors(node))

        if not predecessors:  # No parents
            if successors:  # Has children
                G.nodes[node]['node_type'] = 'parent'
            else:  # No children either
                G.nodes[node]['node_type'] = 'isolated'
        else:  # Has parents
            if not successors:  # No children
                G.nodes[node]['node_type'] = 'leaf'
            else:  # Has both parents and children
                G.nodes[node]['node_type'] = 'child'

def get_border_color(node_type):
    """
    Return border color based on node type
    """
    if node_type == 'parent':
        return '#0000FF'  # Blue
    elif node_type == 'child':
        return '#800080'  # Purple
    elif node_type == 'leaf':
        return '#FF00FF'  # Magenta
    else:
        return '#000000'  # Default black

def get_probability_color(priors):
    """
    Create background color based on probability (red to green gradient)
    """
    # Default to neutral color if no probability
    if not priors or 'true_prob' not in priors:
        return '#F8F8F8'  # Light grey

    # Get probability value
    prob = priors['true_prob']

    # Create color gradient from red (0.0) to green (1.0)
    hue = 120 * prob  # 0 = red, 120 = green (in HSL color space)
    saturation = 0.75
    lightness = 0.8  # Lighter color for better text visibility

    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(hue/360, lightness, saturation)

    # Convert to hex format
    hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

    return hex_color

def create_tooltip(node_id, node_data):
    """
    Create rich HTML tooltip with probability information
    Uses simplified HTML that works well in tooltips
    """
    description = node_data.get('description', '')
    priors = node_data.get('priors', {})
    instantiations = node_data.get('instantiations', ["TRUE", "FALSE"])

    # Start building the HTML tooltip
    html = f"""
    <div style='max-width:350px; padding:10px; background-color:#f8f9fa; border-radius:5px; font-family:Arial, sans-serif;'>
        <h3 style='margin-top:0; color:#202124;'>{node_id}</h3>
        <p style='font-style:italic;'>{description}</p>
    """

    # Add probability information if available
    if priors and 'true_prob' in priors:
        true_prob = priors['true_prob']
        false_prob = 1.0 - true_prob

        # Get proper state names
        true_state = instantiations[0] if len(instantiations) > 0 else "TRUE"
        false_state = instantiations[1] if len(instantiations) > 1 else "FALSE"

        html += f"""
        <div style='margin-top:10px; background-color:#fff; padding:8px; border-radius:4px; border:1px solid #ddd;'>
            <h4 style='margin-top:0; font-size:14px;'>Probabilities:</h4>
            <div>{true_state}: <b>{true_prob:.3f}</b></div>
            <div>{false_state}: <b>{false_prob:.3f}</b></div>
            <div style='width:100%; height:20px; margin-top:5px; border:1px solid #ccc;'>
                <div style='float:left; width:{true_prob*100}%; height:100%; background-color:rgba(0,200,0,0.5); border-right:2px solid green;'></div>
                <div style='float:left; width:{false_prob*100}%; height:100%; background-color:rgba(255,0,0,0.5);'></div>
            </div>
        </div>
        """

    # Add click instruction
    html += """
    <div style='margin-top:10px; font-size:12px; text-align:center; color:#666;'>
        Click for detailed information
    </div>
    """

    # Close the main div
    html += "</div>"

    return html

def create_expanded_content(node_id, node_data):
    """
    Create expanded content shown when a node is clicked
    This is stored as a string and converted to HTML in the click handler
    """
    description = node_data.get('description', '')
    priors = node_data.get('priors', {})
    posteriors = node_data.get('posteriors', {})
    instantiations = node_data.get('instantiations', ["TRUE", "FALSE"])

    # Get probability values
    true_prob = priors.get('true_prob', 0.5) if priors else 0.5
    false_prob = 1.0 - true_prob

    # Get proper state names
    true_state = instantiations[0] if len(instantiations) > 0 else "TRUE"
    false_state = instantiations[1] if len(instantiations) > 1 else "FALSE"

    # Start building HTML content
    html = f"""
    <div style="max-width:600px; padding:20px;">
        <h2 style="margin-top:0;">{node_id}</h2>
        <p style="font-style:italic;">{description}</p>

        <div style="margin-top:20px;">
            <h3>Prior Probabilities</h3>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="background-color:#f0f0f0;">
                    <th style="padding:8px; border:1px solid #ddd; text-align:left;">State</th>
                    <th style="padding:8px; border:1px solid #ddd; text-align:right;">Probability</th>
                    <th style="padding:8px; border:1px solid #ddd;">Visualization</th>
                </tr>
                <tr>
                    <td style="padding:8px; border:1px solid #ddd;">{true_state}</td>
                    <td style="padding:8px; border:1px solid #ddd; text-align:right;">{true_prob:.3f}</td>
                    <td style="padding:8px; border:1px solid #ddd;">
                        <div style="width:100%; height:20px; background-color:#f0f0f0;">
                            <div style="width:{true_prob*100}%; height:100%; background-color:rgba(0,200,0,0.5);"></div>
                        </div>
                    </td>
                </tr>
                <tr>
                    <td style="padding:8px; border:1px solid #ddd;">{false_state}</td>
                    <td style="padding:8px; border:1px solid #ddd; text-align:right;">{false_prob:.3f}</td>
                    <td style="padding:8px; border:1px solid #ddd;">
                        <div style="width:100%; height:20px; background-color:#f0f0f0;">
                            <div style="width:{false_prob*100}%; height:100%; background-color:rgba(255,0,0,0.5);"></div>
                        </div>
                    </td>
                </tr>
            </table>
        </div>
    """

    # Add conditional probabilities if available
    if posteriors and len(posteriors) > 0:
        html += """
        <div style="margin-top:20px;">
            <h3>Conditional Probabilities</h3>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="background-color:#f0f0f0;">
                    <th style="padding:8px; border:1px solid #ddd; text-align:left;">Condition</th>
                    <th style="padding:8px; border:1px solid #ddd; text-align:right;">Value</th>
                </tr>
        """

        # Add each conditional probability
        for key, value in posteriors.items():
            html += f"""
            <tr>
                <td style="padding:8px; border:1px solid #ddd;">{key}</td>
                <td style="padding:8px; border:1px solid #ddd; text-align:right;">{value}</td>
            </tr>
            """

        html += """
            </table>
        </div>
        """

    # Close the main container
    html += """
    </div>
    """

    return html
```

## Phase 2: Node Classification and Styling Module


```python
def classify_nodes(G):
    """
    Classify nodes as parent, child, or leaf based on network structure
    """
    for node in G.nodes():
        predecessors = list(G.predecessors(node))
        successors = list(G.successors(node))

        if not predecessors:  # No parents
            if successors:  # Has children
                G.nodes[node]['node_type'] = 'parent'
            else:  # No children either
                G.nodes[node]['node_type'] = 'isolated'
        else:  # Has parents
            if not successors:  # No children
                G.nodes[node]['node_type'] = 'leaf'
            else:  # Has both parents and children
                G.nodes[node]['node_type'] = 'child'

def get_border_color(node_type):
    """
    Return border color based on node type
    """
    if node_type == 'parent':
        return '#0000FF'  # Blue
    elif node_type == 'child':
        return '#800080'  # Purple
    elif node_type == 'leaf':
        return '#FF00FF'  # Magenta
    else:
        return '#000000'  # Default black

def get_probability_color(priors):
    """
    Create background color based on probability (red to green gradient)
    """
    # Default to neutral color if no probability
    if not priors or 'true_prob' not in priors:
        return '#F8F8F8'  # Light grey

    # Get probability value
    prob = priors['true_prob']

    # Create color gradient from red (0.0) to green (1.0)
    hue = 120 * prob  # 0 = red, 120 = green (in HSL color space)
    saturation = 0.75
    lightness = 0.8  # Lighter color for better text visibility

    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(hue/360, lightness, saturation)

    # Convert to hex format
    hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

    return hex_color

def get_parents(row):
    """
    Extract parent nodes from row data, with safe handling for different data types
    """
    if 'Parents' not in row:
        return []

    parents_data = row['Parents']

    # Handle NaN, None, or empty list
    if isinstance(parents_data, float) and pd.isna(parents_data):
        return []

    if parents_data is None:
        return []

    # Handle different data types
    if isinstance(parents_data, list):
        # Return a list with NaN and empty strings removed
        return [p for p in parents_data if not (isinstance(p, float) and pd.isna(p)) and p != '']

    if isinstance(parents_data, str):
        if not parents_data.strip():
            return []

        # Remove brackets and split by comma, removing empty strings and NaN
        cleaned = parents_data.strip('[]"\'')
        if not cleaned:
            return []

        return [p.strip(' "\'') for p in cleaned.split(',') if p.strip()]

    # Default: empty list
    return []

def get_instantiations(row):
    """
    Extract instantiations with safe handling for different data types
    """
    if 'instantiations' not in row:
        return ["TRUE", "FALSE"]

    inst_data = row['instantiations']

    # Handle NaN or None
    if isinstance(inst_data, float) and pd.isna(inst_data):
        return ["TRUE", "FALSE"]

    if inst_data is None:
        return ["TRUE", "FALSE"]

    # Handle different data types
    if isinstance(inst_data, list):
        return inst_data if inst_data else ["TRUE", "FALSE"]

    if isinstance(inst_data, str):
        if not inst_data.strip():
            return ["TRUE", "FALSE"]

        # Remove brackets and split by comma
        cleaned = inst_data.strip('[]"\'')
        if not cleaned:
            return ["TRUE", "FALSE"]

        return [i.strip(' "\'') for i in cleaned.split(',') if i.strip()]

    # Default
    return ["TRUE", "FALSE"]

def get_priors(row):
    """
    Extract prior probabilities with safe handling for different data types
    """
    if 'priors' not in row:
        return {}

    priors_data = row['priors']

    # Handle NaN or None
    if isinstance(priors_data, float) and pd.isna(priors_data):
        return {}

    if priors_data is None:
        return {}

    result = {}

    # Handle dictionary
    if isinstance(priors_data, dict):
        result = priors_data
    # Handle string representation of dictionary
    elif isinstance(priors_data, str):
        if not priors_data.strip() or priors_data == '{}':
            return {}

        try:
            # Try to evaluate as Python literal
            import ast
            result = ast.literal_eval(priors_data)
        except:
            # Simple parsing for items like {'p(TRUE)': '0.2', 'p(FALSE)': '0.8'}
            if '{' in priors_data and '}' in priors_data:
                content = priors_data[priors_data.find('{')+1:priors_data.rfind('}')]
                items = [item.strip() for item in content.split(',')]

                for item in items:
                    if ':' in item:
                        key, value = item.split(':', 1)
                        key = key.strip(' \'\"')
                        value = value.strip(' \'\"')
                        result[key] = value

    # Extract main probability for TRUE state
    instantiations = get_instantiations(row)
    true_state = instantiations[0] if instantiations else "TRUE"
    true_key = f"p({true_state})"

    if true_key in result:
        try:
            result['true_prob'] = float(result[true_key])
        except:
            pass

    return result

def get_posteriors(row):
    """
    Extract posterior probabilities with safe handling for different data types
    """
    if 'posteriors' not in row:
        return {}

    posteriors_data = row['posteriors']

    # Handle NaN or None
    if isinstance(posteriors_data, float) and pd.isna(posteriors_data):
        return {}

    if posteriors_data is None:
        return {}

    result = {}

    # Handle dictionary
    if isinstance(posteriors_data, dict):
        result = posteriors_data
    # Handle string representation of dictionary
    elif isinstance(posteriors_data, str):
        if not posteriors_data.strip() or posteriors_data == '{}':
            return {}

        try:
            # Try to evaluate as Python literal
            import ast
            result = ast.literal_eval(posteriors_data)
        except:
            # Simple parsing
            if '{' in posteriors_data and '}' in posteriors_data:
                content = posteriors_data[posteriors_data.find('{')+1:posteriors_data.rfind('}')]
                items = [item.strip() for item in content.split(',')]

                for item in items:
                    if ':' in item:
                        key, value = item.split(':', 1)
                        key = key.strip(' \'\"')
                        value = value.strip(' \'\"')
                        result[key] = value

    return result
```

## Phase 3: HTML Content Generation Module


```python
def create_probability_bar(true_prob, false_prob, height="15px", show_values=True, value_prefix=""):
    """
    Creates a reusable HTML component to visualize probability distribution
    """
    true_label = f"{value_prefix}{true_prob:.3f}" if show_values else ""
    false_label = f"{value_prefix}{false_prob:.3f}" if show_values else ""

    html = f"""
    <div style="width:100%; height:{height}; display:flex; border:1px solid #ccc; overflow:hidden; border-radius:3px; margin-top:3px; margin-bottom:3px;">
        <div style="flex-basis:{true_prob*100}%; background:linear-gradient(to bottom, rgba(0,180,0,0.9), rgba(0,140,0,0.7)); border-right:2px solid #008800; display:flex; align-items:center; justify-content:center; overflow:hidden; min-width:{2 if true_prob > 0 else 0}px;">
            <span style="font-size:10px; color:white; text-shadow:0px 0px 2px #000;">{true_label}</span>
        </div>
        <div style="flex-basis:{false_prob*100}%; background:linear-gradient(to bottom, rgba(220,0,0,0.9), rgba(180,0,0,0.7)); border-left:2px solid #880000; display:flex; align-items:center; justify-content:center; overflow:hidden; min-width:{2 if false_prob > 0 else 0}px;">
            <span style="font-size:10px; color:white; text-shadow:0px 0px 2px #000;">{false_label}</span>
        </div>
    </div>
    """
    return html

def create_tooltip(node_id, node_data):
    """
    Create rich HTML tooltip with probability information
    """
    description = node_data.get('description', '')
    priors = node_data.get('priors', {})
    instantiations = node_data.get('instantiations', ["TRUE", "FALSE"])

    # Start building the HTML tooltip
    html = f"""
    <div style="max-width:350px; padding:10px; background-color:#f8f9fa; border-radius:5px; font-family:Arial, sans-serif;">
        <h3 style="margin-top:0; color:#202124;">{node_id}</h3>
        <p style="font-style:italic;">{description}</p>
    """

    # Add prior probabilities section
    if priors and 'true_prob' in priors:
        true_prob = priors['true_prob']
        false_prob = 1.0 - true_prob

        # Get proper state names
        true_state = instantiations[0] if len(instantiations) > 0 else "TRUE"
        false_state = instantiations[1] if len(instantiations) > 1 else "FALSE"

        html += f"""
        <div style="margin-top:10px; background-color:#fff; padding:8px; border-radius:4px; border:1px solid #ddd;">
            <h4 style="margin-top:0; font-size:14px;">Prior Probabilities:</h4>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <div style="font-size:12px;">{true_state}: {true_prob:.3f}</div>
                <div style="font-size:12px;">{false_state}: {false_prob:.3f}</div>
            </div>
            {create_probability_bar(true_prob, false_prob, "20px", True)}
        </div>
        """

    # Add click instruction
    html += """
    <div style="margin-top:8px; font-size:12px; color:#666; text-align:center;">
        Click node to see full probability details
    </div>
    </div>
    """

    return html

def create_expanded_content(node_id, node_data):
    """
    Create expanded content shown when a node is clicked
    """
    description = node_data.get('description', '')
    priors = node_data.get('priors', {})
    posteriors = node_data.get('posteriors', {})
    instantiations = node_data.get('instantiations', ["TRUE", "FALSE"])

    # Get proper state names
    true_state = instantiations[0] if len(instantiations) > 0 else "TRUE"
    false_state = instantiations[1] if len(instantiations) > 1 else "FALSE"

    # Extract probabilities
    true_prob = priors.get('true_prob', 0.5)
    false_prob = 1.0 - true_prob

    # Start building the expanded content
    html = f"""
    <div style="max-width:500px; padding:15px; font-family:Arial, sans-serif;">
        <h2 style="margin-top:0; color:#333;">{node_id}</h2>
        <p style="font-style:italic; margin-bottom:15px;">{description}</p>

        <div style="margin-bottom:20px; padding:12px; border:1px solid #ddd; background-color:#f9f9f9; border-radius:5px;">
            <h3 style="margin-top:0; color:#333;">Prior Probabilities</h3>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <div><strong>{true_state}:</strong> {true_prob:.3f}</div>
                <div><strong>{false_state}:</strong> {false_prob:.3f}</div>
            </div>
            {create_probability_bar(true_prob, false_prob, "25px", True)}
        </div>
    """

    # Add conditional probability table if available
    if posteriors:
        html += """
        <div style="padding:12px; border:1px solid #ddd; background-color:#f9f9f9; border-radius:5px;">
            <h3 style="margin-top:0; color:#333;">Conditional Probabilities</h3>
            <table style="width:100%; border-collapse:collapse; font-size:13px;">
                <tr style="background-color:#eee;">
                    <th style="padding:8px; text-align:left; border:1px solid #ddd;">Condition</th>
                    <th style="padding:8px; text-align:center; border:1px solid #ddd; width:80px;">Value</th>
                    <th style="padding:8px; text-align:center; border:1px solid #ddd;">Visualization</th>
                </tr>
        """

        # Sort posteriors to group by similar conditions
        posterior_items = list(posteriors.items())
        posterior_items.sort(key=lambda x: x[0])

        # Add rows for conditional probabilities
        for key, value in posterior_items:
            try:
                # Try to parse probability value
                prob_value = float(value)
                inv_prob = 1.0 - prob_value

                # Add row with probability visualization
                html += f"""
                <tr>
                    <td style="padding:8px; border:1px solid #ddd;">{key}</td>
                    <td style="padding:8px; text-align:center; border:1px solid #ddd;">{prob_value:.3f}</td>
                    <td style="padding:8px; border:1px solid #ddd;">
                        {create_probability_bar(prob_value, inv_prob, "20px", False)}
                    </td>
                </tr>
                """
            except:
                # Fallback for non-numeric values
                html += f"""
                <tr>
                    <td style="padding:8px; border:1px solid #ddd;">{key}</td>
                    <td style="padding:8px; text-align:center; border:1px solid #ddd;" colspan="2">{value}</td>
                </tr>
                """

        html += """
            </table>
        </div>
        """

    html += "</div>"

    return html
```

## Phase 4: Main Visualization Function


```python
def create_bayesian_network_with_probabilities(df):
    """
    Create an interactive Bayesian network visualization with enhanced probability visualization
    and node classification based on network structure.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with proper attributes
    for idx, row in df.iterrows():
        title = row['Title']
        description = row['Description']

        # Process probability information
        priors = get_priors(row)
        instantiations = get_instantiations(row)

        # Add node with base information
        G.add_node(
            title,
            description=description,
            priors=priors,
            instantiations=instantiations,
            posteriors=get_posteriors(row)
        )

    # Add edges
    for idx, row in df.iterrows():
        child = row['Title']
        parents = get_parents(row)

        # Add edges from each parent to this child
        for parent in parents:
            if parent in G.nodes():
                G.add_edge(parent, child)

    # Classify nodes based on network structure
    classify_nodes(G)

    # Create network visualization
    net = Network(notebook=True, directed=True, cdn_resources="in_line", height="600px", width="100%")

    # Configure physics for better layout
    net.force_atlas_2based(gravity=-50, spring_length=100, spring_strength=0.02)
    net.show_buttons(filter_=['physics'])

    # Add the graph to the network
    net.from_nx(G)

    # Enhance node appearance with probability information and classification
    for node in net.nodes:
        node_id = node['id']
        node_data = G.nodes[node_id]

        # Get node type and set border color
        node_type = node_data.get('node_type', 'unknown')
        border_color = get_border_color(node_type)

        # Get probability information
        priors = node_data.get('priors', {})
        true_prob = priors.get('true_prob', 0.5) if priors else 0.5

        # Get proper state names
        instantiations = node_data.get('instantiations', ["TRUE", "FALSE"])
        true_state = instantiations[0] if len(instantiations) > 0 else "TRUE"
        false_state = instantiations[1] if len(instantiations) > 1 else "FALSE"

        # Create background color based on probability
        background_color = get_probability_color(priors)

        # Create tooltip with probability information
        tooltip = create_tooltip(node_id, node_data)

        # Create a simpler node label with probability
        simple_label = f"{node_id}\np={true_prob:.2f}"

        # Store expanded content as a node attribute for use in click handler
        node_data['expanded_content'] = create_expanded_content(node_id, node_data)

        # Set node attributes
        node['title'] = tooltip  # Tooltip HTML
        node['label'] = simple_label  # Simple text label
        node['shape'] = 'box'
        node['color'] = {
            'background': background_color,
            'border': border_color,
            'highlight': {
                'background': background_color,
                'border': border_color
            }
        }

    # Set up the click handler with proper data
    setup_data = {
        'nodes_data': {node_id: {
            'expanded_content': json.dumps(G.nodes[node_id].get('expanded_content', '')),
            'description': G.nodes[node_id].get('description', ''),
            'priors': G.nodes[node_id].get('priors', {}),
            'posteriors': G.nodes[node_id].get('posteriors', {})
        } for node_id in G.nodes()}
    }

    # Add custom click handling JavaScript
    click_js = """
    // Store node data for click handling
    var nodesData = %s;

    // Add event listener for node clicks
    network.on("click", function(params) {
        if (params.nodes.length > 0) {
            var nodeId = params.nodes[0];
            var nodeInfo = nodesData[nodeId];

            if (nodeInfo) {
                // Create a modal popup for expanded content
                var modal = document.createElement('div');
                modal.style.position = 'fixed';
                modal.style.left = '50%%';
                modal.style.top = '50%%';
                modal.style.transform = 'translate(-50%%, -50%%)';
                modal.style.backgroundColor = 'white';
                modal.style.padding = '20px';
                modal.style.borderRadius = '5px';
                modal.style.boxShadow = '0 0 10px rgba(0,0,0,0.5)';
                modal.style.zIndex = '1000';
                modal.style.maxWidth = '80%%';
                modal.style.maxHeight = '80%%';
                modal.style.overflow = 'auto';

                // Parse the JSON string back to HTML content
                try {
                    var expandedContent = JSON.parse(nodeInfo.expanded_content);
                    modal.innerHTML = expandedContent;
                } catch (e) {
                    modal.innerHTML = 'Error displaying content: ' + e.message;
                }

                // Add close button
                var closeBtn = document.createElement('button');
                closeBtn.innerHTML = 'Close';
                closeBtn.style.marginTop = '10px';
                closeBtn.style.padding = '5px 10px';
                closeBtn.style.cursor = 'pointer';
                closeBtn.onclick = function() {
                    document.body.removeChild(modal);
                };
                modal.appendChild(closeBtn);

                // Add modal to body
                document.body.appendChild(modal);
            }
        }
    });
    """ % json.dumps(setup_data['nodes_data'])

    # Save the graph to HTML
    html_file = "bayesian_network.html"
    net.save_graph(html_file)

    # Inject custom click handling into HTML
    try:
        with open(html_file, "r") as f:
            html_content = f.read()

        # Insert click handling script before the closing body tag
        html_content = html_content.replace('</body>', f'<script>{click_js}</script></body>')

        # Write back the modified HTML
        with open(html_file, "w") as f:
            f.write(html_content)

        return HTML(html_content)
    except Exception as e:
        return HTML(f"<p>Error rendering HTML: {str(e)}</p><p>The network visualization has been saved to '{html_file}'</p>")
```

# Quickly check HTML Outputs


```python
create_bayesian_network_with_probabilities(result_df)
```


```python
# Use the function to create and display the visualization

print(result_df)
```

# 5.0 Archive_version_histories



1.   Import Libraries & Install Packages: [Run Section 0.1](https://colab.research.google.com/github/SingularitySmith/AMTAIR_Prototype/blob/main/Public_AMTAIR_Prototype.ipynb#scrollTo=0_1_Import_Libraries_Packages)
2.   Connect to GitHub Repository & Load Data files: Run Section 0.2
3.   ...
4. [Link Text](#cell-id)
      Requires:
<a name="cell-id"></a>

4. [Test](#Preview-MD-Content)


##### Heading
This is the cell I'm linking to




```python
# notebook_name = "NoHTML_AMTAIR_Prototype"
# repo_url = "https://raw.githubusercontent.com/SingularitySmith/AMTAIR_Prototype/main/data/example_1/"


# !wget {repo_url}{notebook_name}.ipynb
# !jupyter nbconvert --to markdown {notebook_name}.ipynb --output {notebook_name}.md --no-input
```


```python
# Convert ipynb to HTML in Colab
# Upload ipynb
# from google.colab import files
# f = files.upload()

# Convert ipynb to html
# import subprocess
# file0 = list(f.keys())[0]
# _ = subprocess.run(["pip", "install", "nbconvert"])
# _ = subprocess.run(["jupyter", "nbconvert", file0, "--to", "html"])

# download the html
# files.download(file0[:-5]+"html")

```

# 6.0 Save Outputs


## Convert ipynb to HTML in Colab

Instruction:

Download the ipynb, which you want to convert, on your local computer.
Run the code below to upload the ipynb.

The html version will be downloaded automatically on your local machine.
Enjoy it!


```python
#@title Convert ipynb to HTML in Colab
import nbformat
from nbconvert import HTMLExporter
import os

repo_url = "https://raw.githubusercontent.com/SingularitySmith/AMTAIR_Prototype/main/data/example_1/"
notebook_name = "AMTAIR_Prototype_example1"  #Change Notebook name and path when working on different examples

# Download the notebook file
!wget {repo_url}{notebook_name}.ipynb -O {notebook_name}.ipynb  # Corrected line

# Load the notebook
# add error handling for file not found
try:
  with open(f"{notebook_name}.ipynb") as f:
    nb = nbformat.read(f, as_version=4)
except FileNotFoundError:
  print(f"Error: File '{notebook_name}.ipynb' not found. Please check if it was downloaded correctly.")

# Initialize the HTML exporter
exporter = HTMLExporter()

# Convert the notebook to HTML
(body, resources) = exporter.from_notebook_node(nb)

# Save the HTML to a file
with open(f"{notebook_name}IPYNB.html", "w") as f:
    f.write(body)
```

## Convert .ipynb Notebook to MarkDown


```python
import nbformat
from nbconvert import MarkdownExporter
import os

repo_url = "https://raw.githubusercontent.com/SingularitySmith/AMTAIR_Prototype/main/data/example_1/"
notebook_name = "AMTAIR_Prototype_example1"  #Change Notebook name and path when working on different examples

# Download the notebook file
!wget {repo_url}{notebook_name}.ipynb -O {notebook_name}.ipynb  # Corrected line

# Load the notebook
# add error handling for file not found
try:
  with open(f"{notebook_name}.ipynb") as f:
    nb = nbformat.read(f, as_version=4)
except FileNotFoundError:
  print(f"Error: File '{notebook_name}.ipynb' not found. Please check if it was downloaded correctly.")


# Initialize the Markdown exporter
exporter = MarkdownExporter(exclude_output=True)  # Correct initialization

# Convert the notebook to Markdown
(body, resources) = exporter.from_notebook_node(nb)

# Save the Markdown to a file
with open(f"{notebook_name}IPYNB.md", "w") as f:
    f.write(body)
```
