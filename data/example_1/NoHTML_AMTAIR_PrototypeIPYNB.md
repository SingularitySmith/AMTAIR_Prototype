# [AMTAIR Prototype Demonstration (Public Colab Notebook)](https://colab.research.google.com/github/SingularitySmith/AMTAIR_Prototype/blob/main/version_history/AMTAIR_Prototype_0_1.3.ipynb#scrollTo=lt8-AnebGUXr)

## Instructions --- How to use this notebook:

1.   Import Libraries & Install Packages: [Run Section 0.1](https://colab.research.google.com/github/SingularitySmith/AMTAIR_Prototype/blob/main/Public_AMTAIR_Prototype.ipynb#scrollTo=0_1_Import_Libraries_Packages)
2.   Connect to GitHub Repository & Load Data files: Run Section 0.2
3.   ...
4. [Link Text](#cell-id)
      Requires:
<a name="cell-id"></a>
# Heading
This is the cell I'm linking to



1. **Import Libraries & Install Packages**: Run Section 0.1 to set up the necessary dependencies for data processing and visualization.

2. **Connect to GitHub Repository & Load Data files**: Run Section 0.2 to establish connections to the data repository and load example datasets. This step retrieves sample ArgDown files and extracted data for demonstration.

3. **Process Source Documents to ArgDown**: Sections 1.0-1.8 demonstrate the extraction of argument structures from source documents (such as PDFs) into ArgDown format, a markdown-like notation for structured arguments.

4. **Convert ArgDown to BayesDown**: Sections 2.0-2.3 handle the transformation of ArgDown files into BayesDown format, which incorporates probabilistic information into the argument structure.

5. **Extract Data into Structured Format**: Section 3.0 processes BayesDown format into structured database entries (CSV) that can be used for analysis.

6. **Create and Analyze Bayesian Networks**: Section 4.0 demonstrates how to build Bayesian networks from the extracted data and provides tools for analyzing risk pathways.

7. **Save and Export Results**: Sections 5.0-6.0 provide methods for archiving results and exporting visualizations.

### Key Concepts:

- **ArgDown**: A structured format for representing arguments, with hierarchical relationships between statements.
- **BayesDown**: An extension of ArgDown that incorporates probabilistic information, allowing for Bayesian network construction.
- **Extraction Pipeline**: The process of converting unstructured text to structured argument representations.
- **Bayesian Networks**: Probabilistic graphical models that represent variables and their conditional dependencies.

### Example Workflow:

1. Load a sample ArgDown file from the repository
2. Extract the hierarchical structure and relationships
3. Add probabilistic information to create a BayesDown representation
4. Generate a Bayesian network visualization
5. Analyze conditional probabilities and risk pathways

### Troubleshooting:

- If connectivity issues occur, ensure you have access to the GitHub repository
- For visualization errors, check that all required libraries are properly installed
- When processing custom files, ensure they follow the expected format conventions


# 0.1 Import Libraries & Packages



```python
!pip install pyvis
!pip install --upgrade gspread pandas google-auth google-colab

!pip install pgmpy

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

# 0.3 File Import


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

md_content_ex_rain
```

## 1.6.2 Check the Graph Structure with the ArgDown Sandbox Online
Copy and paste the BayesDown formatted ... in the ArgDown Sandbox below to quickly verify that the network renders correctly.


```python
from IPython.display import IFrame

IFrame(src="https://argdown.org/sandbox/map/", width="100%", height="600px")
```

## 1.7 Extract ArgDown Graph Information as DataFrame


```python

```

## 1.8 Store ArgDown Information as 'ArgDown.csv' file


```python

```

# 2.0 Probability Extractions: ArgDown (.csv) to BayesDown (.md + plugin JSON syntax)


```python

```

## 2.1 Gernerate and Extract "Prior-, Conditional- and Posterior Probability Questions" from ArgDown.csv


```python

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
# read basic ArgDown example With BayesDown syntax added and corss generational added -- Ella
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

## 3.3 Extraction Algorithm and Code


```python
def parse_markdown_hierarchy(markdown_text):
    """Main function to parse markdown hierarchy into a DataFrame"""

    # Remove comments
    clean_text = remove_comments(markdown_text)

    # Extract all titles with their descriptions and indentation levels
    titles_info = extract_titles_info(clean_text)

    # Establish parent-child relationships
    titles_with_relations = establish_relationships(titles_info, clean_text)

    # Convert to DataFrame
    df = convert_to_dataframe(titles_with_relations)

    # Add No_Parent and No_Children columns
    df = add_no_parent_no_child_columns_to_df(df)

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

    return titles_info

def convert_to_dataframe(titles_info):
    """Convert the titles information dictionary to a pandas DataFrame"""
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

                    # Create the row dictionary with basic fields
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
```


```python
result_df = parse_markdown_hierarchy(md_content_ex_rain)
result_df
```


```python
result_df_CG = parse_markdown_hierarchy(md_content_easy_ex_B_CG)
result_df_CG
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

### 4.1.1 Core Functions with Imports


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

### 4.1.2. Node Classification and Color Functions


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
    # Using HSL for better visual gradient
    hue = 120 * prob  # 0 = red, 120 = green (in HSL color space)
    saturation = 0.75
    lightness = 0.8  # Lighter color for better text visibility

    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(hue/360, lightness, saturation)

    # Convert to hex format
    hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

    return hex_color
```

### 4.1.3. Probability Visualization Components


```python
def create_node_label(node_id, node_data):
    """
    Create HTML label content for node with probability visualization
    """
    priors = node_data.get('priors', {})
    instantiations = node_data.get('instantiations', ["TRUE", "FALSE"])

    # If no probability information, just return the node ID
    if not priors or 'true_prob' not in priors:
        return node_id

    # Get probabilities
    true_prob = priors.get('true_prob', 0.5)
    false_prob = 1.0 - true_prob

    # Get proper state names
    true_state = instantiations[0] if len(instantiations) > 0 else "TRUE"
    false_state = instantiations[1] if len(instantiations) > 1 else "FALSE"

    # Create HTML for the node label with probability box
    html = f"""
    <div style="text-align:center; font-weight:bold;">{node_id}</div>
    <div style="margin-top:4px; display:flex; flex-direction:column; align-items:center;">
        <div style="font-size:10px;">p({true_state})={true_prob:.3f} | p({false_state})={false_prob:.3f}</div>
        <div style="width:100%; height:10px; display:flex; margin-top:2px; border:1px solid #ccc;">
            <div style="flex-basis:{true_prob*100}%; background-color:rgba(0,200,0,0.5); border-right:2px solid green;"></div>
            <div style="flex-basis:{false_prob*100}%; background-color:rgba(255,0,0,0.5); border-left:2px solid red;"></div>
        </div>
    </div>
    """

    # Create expanded content with detailed probability information (shown on click)
    expanded_html = create_expanded_content(node_id, node_data)

    # Store both versions
    node_data['collapsed_label'] = html
    node_data['expanded_label'] = expanded_html

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

    # Basic info section
    html = f"""
    <div style="max-width:300px; padding:8px; text-align:left;">
        <div style="font-weight:bold; font-size:14px;">{node_id}</div>
        <div style="font-style:italic; margin:4px 0; font-size:12px;">{description}</div>

        <div style="margin-top:6px; font-size:12px;">
            <div>Prior Probabilities:</div>
            <div style="display:flex; align-items:center; margin-top:3px;">
                <div style="width:120px;">p({true_state})={true_prob:.3f}</div>
                <div style="flex-grow:1; height:14px; display:flex; border:1px solid #ccc;">
                    <div style="flex-basis:{true_prob*100}%; background-color:rgba(0,200,0,0.5); border-right:2px solid green;"></div>
                    <div style="flex-basis:{false_prob*100}%; background-color:rgba(255,0,0,0.5); border-left:2px solid red;"></div>
                </div>
            </div>
        </div>
    """

    # Add conditional probability table if available
    if posteriors:
        html += """
        <div style="margin-top:8px; font-size:12px;">
            <div>Conditional Probabilities:</div>
            <table style="width:100%; border-collapse:collapse; margin-top:3px; font-size:10px;">
                <tr style="background-color:#f0f0f0;">
                    <th style="padding:2px; text-align:left; border:1px solid #ccc;">Condition</th>
                    <th style="padding:2px; text-align:center; border:1px solid #ccc;">Value</th>
                    <th style="padding:2px; text-align:center; border:1px solid #ccc;">Visualization</th>
                </tr>
        """

        # Add rows for conditional probabilities
        for i, (key, value) in enumerate(posteriors.items()):
            try:
                prob_value = float(value)
                # Create visualization for this probability
                html += f"""
                <tr>
                    <td style="padding:2px; border:1px solid #ccc;">{key}</td>
                    <td style="padding:2px; text-align:center; border:1px solid #ccc;">{value}</td>
                    <td style="padding:2px; border:1px solid #ccc;">
                        <div style="width:100%; height:12px; display:flex;">
                            <div style="flex-basis:{prob_value*100}%; background-color:rgba(0,200,0,0.5); border-right:2px solid green;"></div>
                            <div style="flex-basis:{(1.0-prob_value)*100}%; background-color:rgba(255,0,0,0.5); border-left:2px solid red;"></div>
                        </div>
                    </td>
                </tr>
                """
            except:
                # Fallback for non-numeric values
                html += f"""
                <tr>
                    <td style="padding:2px; border:1px solid #ccc;">{key}</td>
                    <td style="padding:2px; text-align:center; border:1px solid #ccc;" colspan="2">{value}</td>
                </tr>
                """

        html += """
            </table>
        </div>
        """

    # Add click instruction
    html += """
    <div style="margin-top:8px; font-size:10px; color:#666; text-align:center;">
        Click again to collapse
    </div>
    """

    html += "</div>"

    return html
```

### 4.1.4. Enhanced Tooltip Generation






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

def create_node_label(node_id, node_data):
    """
    Create HTML label content for node with probability visualization
    """
    priors = node_data.get('priors', {})
    instantiations = node_data.get('instantiations', ["TRUE", "FALSE"])

    # If no probability information, just return the node ID
    if not priors or 'true_prob' not in priors:
        return f'<div style="text-align:center; font-weight:bold;">{node_id}</div>'

    # Get probabilities
    true_prob = priors.get('true_prob', 0.5)
    false_prob = 1.0 - true_prob

    # Create HTML for the node label with probability box
    html = f"""
    <div style="text-align:center; font-weight:bold;">{node_id}</div>
    <div style="margin-top:4px; font-size:10px; text-align:center;">p={true_prob:.2f}</div>
    <div style="width:100%; height:10px; display:flex; margin:2px 0; border:1px solid #ccc;">
        <div style="width:{true_prob*100}%; background-color:rgba(0,200,0,0.9); border-right:2px solid green;"></div>
        <div style="width:{false_prob*100}%; background-color:rgba(255,0,0,0.9); border-left:2px solid red;"></div>
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

    # Build a cleaner tooltip that will render correctly
    html = f"""
    <div style="width:300px; padding:10px; background-color:#f8f9fa; border-radius:5px; font-family:Arial, sans-serif;">
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
            <div style="width:100%; height:20px; display:flex; border:1px solid #ccc;">
                <div style="width:{true_prob*100}%; background-color:rgba(0,200,0,0.9); border-right:2px solid green;"></div>
                <div style="width:{false_prob*100}%; background-color:rgba(255,0,0,0.9); border-left:2px solid red;"></div>
            </div>
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
            <div style="width:100%; height:25px; display:flex; border:1px solid #ccc; border-radius:3px;">
                <div style="width:{true_prob*100}%; background:linear-gradient(to bottom, rgba(0,180,0,0.9), rgba(0,140,0,0.7)); border-right:2px solid #008800; display:flex; align-items:center; justify-content:center;">
                    <span style="font-size:12px; color:white; text-shadow:0px 0px 2px #000;">{true_prob:.3f}</span>
                </div>
                <div style="width:{false_prob*100}%; background:linear-gradient(to bottom, rgba(220,0,0,0.9), rgba(180,0,0,0.7)); border-left:2px solid #880000; display:flex; align-items:center; justify-content:center;">
                    <span style="font-size:12px; color:white; text-shadow:0px 0px 2px #000;">{false_prob:.3f}</span>
                </div>
            </div>
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
                        <div style="width:100%; height:20px; display:flex; border:1px solid #ccc; border-radius:3px;">
                            <div style="width:{prob_value*100}%; background:linear-gradient(to bottom, rgba(0,180,0,0.9), rgba(0,140,0,0.7)); display:flex; align-items:center; justify-content:center; min-width:{2 if prob_value > 0 else 0}px;">
                            </div>
                            <div style="width:{inv_prob*100}%; background:linear-gradient(to bottom, rgba(220,0,0,0.9), rgba(180,0,0,0.7)); display:flex; align-items:center; justify-content:center; min-width:{2 if inv_prob > 0 else 0}px;">
                            </div>
                        </div>
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
    # Using HSL for better visual gradient
    hue = 120 * prob  # 0 = red, 120 = green (in HSL color space)
    saturation = 0.95  # Increased saturation for more vibrant colors
    lightness = 0.75  # Slightly lighter for better text visibility

    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(hue/360, lightness, saturation)

    # Convert to hex format
    hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

    return hex_color
```

### 4.1.5. Existing Helper Functions (Modified for Robustness)


```python
def get_parents(row):
    """Extract parent nodes from row data, with safe handling for different data types"""
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
    """Extract instantiations with safe handling for different data types"""
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
    """Extract prior probabilities with safe handling for different data types"""
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
                        key = key.strip(' \'"')
                        value = value.strip(' \'"')
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
    """Extract posterior probabilities with safe handling for different data types"""
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
                        key = key.strip(' \'"')
                        value = value.strip(' \'"')
                        result[key] = value

    return result
```

### 4.1.6. Usage and Integration


```python
# Use the function to create and display the visualization

print(result_df)



# This would typically be called with the result_df from the data extraction

create_bayesian_network_with_probabilities(result_df)
```

### 4.1.B Integrated: Generating Simple Network from Scratch


```python

from pyvis.network import Network
import networkx as nx
from IPython.display import HTML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def create_bayesian_network_with_probabilities(df):
    """
    Create an interactive Bayesian network visualization with probability information
    displayed in tooltips and through visual encoding
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

    # Create network visualization
    net = Network(notebook=True, directed=True, cdn_resources="in_line", height="600px", width="100%")

    # Configure physics for better layout
    net.force_atlas_2based(gravity=-50, spring_length=100, spring_strength=0.02)
    net.show_buttons(filter_=['physics'])

    # Add the graph to the network
    net.from_nx(G)

    # Enhance node appearance with probability information
    for node in net.nodes:
        node_id = node['id']
        node_data = G.nodes[node_id]

        # Create tooltip with probability information
        tooltip = create_tooltip(node_id, node_data)

        # Determine node color based on node type
        if list(G.predecessors(node_id)):  # Has parents
            if list(G.successors(node_id)):  # Also has children
                color = "#FBBC05"  # Orange for intermediate nodes
            else:
                color = "#34A853"  # Green for leaf nodes
        else:
            color = "#4285F4"  # Blue for root nodes

        # Set node attributes
        node['title'] = tooltip
        node['shape'] = 'box'
        node['color'] = color

        # Add marginal probability to label if available
        priors = node_data.get('priors', {})
        if priors and 'true_prob' in priors:
            prob = priors['true_prob']
            node['label'] = f"{node_id}\np={prob:.2f}"
        else:
            node['label'] = node_id

    # Save and read the HTML content
    html_file = "bayesian_network.html"
    net.save_graph(html_file)

    try:
        with open(html_file, "r") as f:
            html_content = f.read()
        return HTML(html_content)
    except Exception as e:
        return HTML(f"<p>Error rendering HTML: {str(e)}</p><p>The network visualization has been saved to '{html_file}'</p>")

def get_parents(row):
    """Extract parent nodes from row data, with safe handling for different data types"""
    if 'Parents' not in row:
        return []

    parents_data = row['Parents']

    # Handle NaN, None, or empty list (More robust)
    # Using any() or all() to resolve ambiguity
    if pd.isna(parents_data).any() or (isinstance(parents_data, (list, np.ndarray, str)) and (len(parents_data) == 0 or (isinstance(parents_data, str) and parents_data.strip() == ''))):
        return []

    # Handle different data types
    if isinstance(parents_data, list):
        # Return a list with NaN and empty strings removed
        return [p for p in parents_data if not pd.isna(p) and p != '']

    if isinstance(parents_data, str):
        if not parents_data.strip():
            return []

        # Remove brackets and split by comma, removing empty strings and NaN
        cleaned = parents_data.strip('[]"\'')
        if not cleaned:
            return []

        return [p.strip(' "\'') for p in cleaned.split(',') if p.strip() and not pd.isna(p)]

    # Default: empty list
    return []

def get_instantiations(row):
    """Extract instantiations with safe handling for different data types"""
    if 'instantiations' not in row:
        return ["TRUE", "FALSE"]

    inst_data = row['instantiations']

    # Handle NaN or None (Corrected condition)
    if pd.isna(inst_data).any():  # or pd.isnull(inst_data).any()
        return ["TRUE", "FALSE"]

    # ... (Rest of the function remains the same)

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
    """Extract prior probabilities with safe handling for different data types"""
    if 'priors' not in row:
        return {}

    priors_data = row['priors']

    # Handle NaN or None
    if pd.isna(priors_data):
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
            result = eval(priors_data)
        except:
            # Simple parsing for items like {'p(TRUE)': '0.2', 'p(FALSE)': '0.8'}
            if '{' in priors_data and '}' in priors_data:
                content = priors_data[priors_data.find('{')+1:priors_data.rfind('}')]
                items = [item.strip() for item in content.split(',')]

                for item in items:
                    if ':' in item:
                        key, value = item.split(':', 1)
                        key = key.strip(' \'"')
                        value = value.strip(' \'"')
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
    """Extract posterior probabilities with safe handling for different data types"""
    if 'posteriors' not in row:
        return {}

    posteriors_data = row['posteriors']

    # Handle NaN or None
    if pd.isna(posteriors_data):
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
            result = eval(posteriors_data)
        except:
            # Simple parsing
            if '{' in posteriors_data and '}' in posteriors_data:
                content = posteriors_data[posteriors_data.find('{')+1:posteriors_data.rfind('}')]
                items = [item.strip() for item in content.split(',')]

                for item in items:
                    if ':' in item:
                        key, value = item.split(':', 1)
                        key = key.strip(' \'"')
                        value = value.strip(' \'"')
                        result[key] = value

    return result

def create_tooltip(node_id, node_data):
    """
    Create rich HTML tooltip with probability information
    """
    description = node_data.get('description', '')
    priors = node_data.get('priors', {})
    posteriors = node_data.get('posteriors', {})
    instantiations = node_data.get('instantiations', ["TRUE", "FALSE"])

    # Start building the HTML tooltip
    html = f"""
    <div style='max-width:350px; padding:10px; background-color:#f8f9fa; border-radius:5px; font-family:Arial, sans-serif;'>
        <h3 style='margin-top:0; color:#202124;'>{node_id}</h3>
        <p style='font-style:italic;'>{description}</p>
    """

    # Add chart if possible
    chart_html = create_simple_chart(node_id, priors, instantiations)
    if chart_html:
        html += f"""
        <div style='margin:10px 0; text-align:center;'>
            {chart_html}
        </div>
        """

    # Add prior probabilities section
    if priors:
        html += """
        <div style='margin-top:10px; background-color:#fff; padding:8px; border-radius:4px; border:1px solid #ddd;'>
            <h4 style='margin-top:0; font-size:14px;'>Prior Probabilities:</h4>
            <table style='width:100%; border-collapse:collapse;'>
                <tr style='background-color:#f1f3f4;'>
                    <th style='padding:4px; text-align:left; border:1px solid #ddd;'>State</th>
                    <th style='padding:4px; text-align:right; border:1px solid #ddd;'>Probability</th>
                </tr>
        """

        for i, state in enumerate(instantiations):
            key = f"p({state})"
            if key in priors:
                value = priors[key]
                try:
                    prob = float(value)
                    # Add visualization bar
                    html += f"""
                    <tr>
                        <td style='padding:4px; border:1px solid #ddd;'>{state}</td>
                        <td style='padding:4px; text-align:right; border:1px solid #ddd;'>
                            <div style='background:#e8f0fe; width:100%; height:18px; position:relative;'>
                                <div style='position:absolute; height:100%; width:{prob*100}%; background:#4285f4;'></div>
                                <span style='position:relative; z-index:1;'>{value}</span>
                            </div>
                        </td>
                    </tr>
                    """
                except:
                    html += f"""
                    <tr>
                        <td style='padding:4px; border:1px solid #ddd;'>{state}</td>
                        <td style='padding:4px; text-align:right; border:1px solid #ddd;'>{value}</td>
                    </tr>
                    """

        html += """
            </table>
        </div>
        """

    # Add conditional probabilities section (simplified)
    if posteriors:
        html += """
        <div style='margin-top:10px; background-color:#fff; padding:8px; border-radius:4px; border:1px solid #ddd;'>
            <h4 style='margin-top:0; font-size:14px;'>Conditional Probabilities:</h4>
            <div style='max-height:150px; overflow-y:auto;'>
                <table style='width:100%; border-collapse:collapse;'>
                    <tr style='background-color:#f1f3f4;'>
                        <th style='padding:4px; text-align:left; border:1px solid #ddd;'>Condition</th>
                        <th style='padding:4px; text-align:right; border:1px solid #ddd;'>Value</th>
                    </tr>
        """

        # Add first few conditional probabilities
        for i, (key, value) in enumerate(list(posteriors.items())[:8]):
            html += f"""
            <tr>
                <td style='padding:4px; border:1px solid #ddd;'>{key}</td>
                <td style='padding:4px; text-align:right; border:1px solid #ddd;'>{value}</td>
            </tr>
            """

        # Show if there are more
        if len(posteriors) > 8:
            html += f"""
            <tr>
                <td colspan='2' style='padding:4px; text-align:center; border:1px solid #ddd;'>
                    (+{len(posteriors) - 8} more)
                </td>
            </tr>
            """

        html += """
                </table>
            </div>
        </div>
        """

    # Close the main div
    html += "</div>"

    return html

def create_simple_chart(node_id, priors, instantiations):
    """
    Create a simple probability chart as an embedded image
    """
    # Check if we have necessary information
    if 'true_prob' not in priors:
        return ""

    try:
        # Set up the values for visualization
        labels = instantiations[:2]  # Use first two states
        values = []

        # Get probability for first state
        true_prob = priors['true_prob']
        values.append(true_prob)

        # Calculate second state probability
        if len(labels) > 1:
            values.append(1.0 - true_prob)

        # Create the chart
        plt.figure(figsize=(3, 2))
        bars = plt.bar(labels, values, color=['#4285F4', '#34A853'])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        plt.ylim(0, 1.1)
        plt.title(f"Probability Distribution", fontsize=10)
        plt.tight_layout()

        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=80)
        plt.close()

        # Convert to base64 for embedding
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')

        return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;">'
    except Exception as e:
        # Return empty string if there's any error
        return ""

# Use the function to create and display the visualization
# create_bayesian_network_with_probabilities(result_df)

```

# 5.0 Archive_version_histories

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

# 6.0 Save Outputs


## Convert ipynb to HTML in Colab

Instruction:

Download the ipynb, which you want to convert, on your local computer.
Run the code below to upload the ipynb.

The html version will be downloaded automatically on your local machine.
Enjoy it!


```python
#@title Convert ipynb to HTML in Colab
# Upload ipynb
from google.colab import files
f = files.upload()

# Convert ipynb to html
import subprocess
file0 = list(f.keys())[0]
_ = subprocess.run(["pip", "install", "nbconvert"])
_ = subprocess.run(["jupyter", "nbconvert", file0, "--to", "html"])

# download the html
files.download(file0[:-5]+"html")

```

# Quickly check HTML Outputs


```python
# Use the function to create and display the visualization

print(result_df)
```


```python
create_bayesian_network_with_probabilities(result_df)
```
