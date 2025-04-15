# BayesDown Syntax Specification

## 1. Introduction to BayesDown

BayesDown stands at the intersection of structured argumentation and probabilistic modeling—a bridge between the qualitative reasoning of ArgDown and the quantitative precision of Bayesian networks. It enables us to capture not just the skeleton of an argument but the uncertainties that pulse through its veins.

This document serves as both a technical reference and a practical guide for extracting structured representations from source materials, whether you're an LLM following carefully crafted prompts or a human researcher verifying the output. The syntax described here forms the backbone of the AMTAIR project's extraction pipeline, transforming natural language arguments into formal models that can inform AI governance decision-making.

## 2. Core Concepts and Foundations

BayesDown extends ArgDown (a markdown-like notation for argument mapping) with probabilistic information that enables Bayesian network construction. The fundamental elements include:

- **Nodes**: Representing variables or statements in an argument
- **Relationships**: Hierarchical connections between nodes
- **Instantiations**: Possible states of each variable
- **Probabilities**: Prior and conditional distributions quantifying uncertainties

What distinguishes BayesDown is its ability to retain the narrative richness of the original arguments while adding the precision of probabilistic quantification—a duality that makes it particularly valuable for modeling complex domains like AI risk.

## 3. Basic Syntax Elements

### 3.1 Node Definition

```
[Node_Name]: Description {"instantiations": ["state1", "state2"], ...metadata...}
```

or

```
<Node_Name>: Description {"instantiations": ["state1", "state2"], ...metadata...}
```

Where:

- **Node_Name**: The identifier for the node (variable)
- **Description**: A textual description of what the node represents
- **Metadata**: JSON-structured probabilistic information in curly braces

### 3.2 Hierarchy Definition

Hierarchical relationships use indentation and symbols:

```
[Parent_Node]: Description {...metadata...}
 + [Child_Node]: Description {...metadata...}
   + [Grandchild_Node]: Description {...metadata...}
```

The `+` symbol preceding a node, combined with indentation, indicates parent-child relationships. The parser detects these relationships based on both the explicit symbols and the indentation levels.

### 3.3 Metadata Structure

The metadata must be valid JSON with this structure:

```json
{
  "instantiations": ["state1", "state2", ...],
  "priors": {
    "p(state1)": "0.7",
    "p(state2)": "0.3",
    ...
  },
  "posteriors": {
    "p(state1|parent1,parent2)": "0.8",
    "p(state1|parent1,not_parent2)": "0.6",
    ...
  }
}
```

#### Required Fields

- `instantiations`: Array of strings representing all possible states of the variable

#### Optional Fields

- `priors`: Object mapping state probability expressions to values
- `posteriors`: Object mapping conditional probability expressions to values

## 4. Extraction Guidelines

### 4.1 For LLMs: Extraction Prompt Patterns

When instructing an LLM to extract BayesDown from source material, follow these prompt patterns:

```
Given the following text about [topic], extract key variables and their causal relationships into BayesDown format:

1. Identify the main variables discussed (examples: AI capabilities, governance mechanisms, risk factors)
2. Determine possible states for each variable (typically binary TRUE/FALSE or domain-specific states)
3. Extract explicit or implicit probability judgments
4. Structure the hierarchical relationships between variables
5. Format your response using BayesDown syntax as follows:

[Variable1]: Description of this variable/concept. {"instantiations": ["state1", "state2"], "priors": {"p(state1)": "0.7", "p(state2)": "0.3"}}
 + [Variable2]: Description of this related variable. {"instantiations": ["TRUE", "FALSE"], "posteriors": {"p(Variable2|Variable1)": "0.8", "p(Variable2|not_Variable1)": "0.2"}}

SOURCE TEXT:
[paste source text here]
```

When the source material has implicit probability judgments, add:

```
For implicit probability judgments, use these mapping guidelines:
- "Almost certain" → 0.95-0.99
- "Very likely" → 0.85-0.95
- "Likely" → 0.65-0.85
- "Possible" or "Could" → 0.35-0.65
- "Unlikely" → 0.15-0.35
- "Very unlikely" → 0.05-0.15
- "Almost impossible" → 0.01-0.05
```

### 4.2 For Human Extractors

When manually extracting BayesDown, follow this workflow:

1. **Read through the source material** completely first to understand the overall argument structure
2. **Identify key variables** that represent important concepts, factors, or outcomes
3. **Determine possible states** for each variable (prefer binary states unless more are clearly needed)
4. **Map relationships** between variables, noting which factors influence others
5. **Estimate probabilities** based on explicit statements or implied strength of belief
6. **Construct BayesDown syntax** with careful attention to formatting requirements
7. **Validate the extraction** by checking that it captures the core argument accurately

### 4.3 Source Material Considerations

Different types of source material require different extraction approaches:

- **Research Papers**: Focus on extracting the causal model underlying the argument, paying special attention to explicitly stated probabilities and confidence levels
- **Blog Posts/Essays**: Look for qualitative expressions of certainty that can be mapped to quantitative probabilities
- **Expert Interviews**: Capture ranges of probability when exact values aren't provided
- **Technical Reports**: Pay attention to conditional dependencies and relationship structures

## 5. Advanced Syntax Features

### 5.1 Multi-state Variables

BayesDown supports variables with more than two states:

```
[AI_Capability_Level]: The level of capabilities achieved by frontier AI systems.
{"instantiations": ["narrow", "general", "superhuman"], 
 "priors": {"p(narrow)": "0.7", "p(general)": "0.2", "p(superhuman)": "0.1"}}
```

### 5.2 Cross-referencing Nodes

You can reference the same node in multiple places in the hierarchy:

```
[Governance_Failure]: Failure of governance mechanisms to control AI development.
{"instantiations": ["TRUE", "FALSE"], "priors": {"p(TRUE)": "0.4", "p(FALSE)": "0.6"}}
 + [Coordination_Failure]: Failure of different stakeholders to coordinate effectively.
   {"instantiations": ["TRUE", "FALSE"], "priors": {"p(TRUE)": "0.6", "p(FALSE)": "0.4"}}
 + [Technical_Challenge]: Technical difficulty of implementing governance mechanisms.
   {"instantiations": ["TRUE", "FALSE"], "priors": {"p(TRUE)": "0.7", "p(FALSE)": "0.3"}}
   + [Coordination_Failure]
```

The repeated reference to `[Coordination_Failure]` indicates that it influences both `[Governance_Failure]` directly and via `[Technical_Challenge]`.

### 5.3 Comments

You can include comments that will be ignored by the parser:

```
/* This is a comment explaining something important about the following node */
[Node_Name]: Description {...metadata...}
```

## 6. Comprehensive Examples

### 6.1 Classic Rain-Sprinkler-Grass Example

```
[Grass_Wet]: Concentrated moisture on, between and around the blades of grass.
{"instantiations": ["grass_wet_TRUE", "grass_wet_FALSE"], 
 "priors": {"p(grass_wet_TRUE)": "0.322", "p(grass_wet_FALSE)": "0.678"},
 "posteriors": {"p(grass_wet_TRUE|sprinkler_TRUE,rain_TRUE)": "0.99",
                "p(grass_wet_TRUE|sprinkler_TRUE,rain_FALSE)": "0.9",
                "p(grass_wet_TRUE|sprinkler_FALSE,rain_TRUE)": "0.8",
                "p(grass_wet_TRUE|sprinkler_FALSE,rain_FALSE)": "0.0",
                "p(grass_wet_FALSE|sprinkler_TRUE,rain_TRUE)": "0.01",
                "p(grass_wet_FALSE|sprinkler_TRUE,rain_FALSE)": "0.1",
                "p(grass_wet_FALSE|sprinkler_FALSE,rain_TRUE)": "0.2",
                "p(grass_wet_FALSE|sprinkler_FALSE,rain_FALSE)": "1.0"}}
 + [Rain]: Tears of angles crying high up in the skies hitting the ground.
   {"instantiations": ["rain_TRUE", "rain_FALSE"],
    "priors": {"p(rain_TRUE)": "0.2", "p(rain_FALSE)": "0.8"}}
 + [Sprinkler]: Activation of a centrifugal force based CO2 droplet distribution system.
   {"instantiations": ["sprinkler_TRUE", "sprinkler_FALSE"],
    "priors": {"p(sprinkler_TRUE)": "0.44838", "p(sprinkler_FALSE)": "0.55162"},
    "posteriors": {"p(sprinkler_TRUE|rain_TRUE)": "0.01",
                   "p(sprinkler_TRUE|rain_FALSE)": "0.4",
                   "p(sprinkler_FALSE|rain_TRUE)": "0.99",
                   "p(sprinkler_FALSE|rain_FALSE)": "0.6"}}
   + [Rain]
```

### 6.2 AI Safety Example: Carlsmith Power-Seeking AI

```
[Existential_Catastrophe]: Human extinction or permanent disempowerment due to AI.
{"instantiations": ["TRUE", "FALSE"], 
 "priors": {"p(TRUE)": "0.05", "p(FALSE)": "0.95"},
 "posteriors": {"p(TRUE|Power_Seeking_TRUE,Misalignment_TRUE,Capability_TRUE)": "0.9",
                "p(TRUE|Power_Seeking_TRUE,Misalignment_TRUE,Capability_FALSE)": "0.1",
                "p(TRUE|Power_Seeking_TRUE,Misalignment_FALSE,Capability_TRUE)": "0.01",
                "p(TRUE|Power_Seeking_TRUE,Misalignment_FALSE,Capability_FALSE)": "0.001",
                "p(TRUE|Power_Seeking_FALSE,Misalignment_TRUE,Capability_TRUE)": "0.2",
                "p(TRUE|Power_Seeking_FALSE,Misalignment_TRUE,Capability_FALSE)": "0.01",
                "p(TRUE|Power_Seeking_FALSE,Misalignment_FALSE,Capability_TRUE)": "0.001",
                "p(TRUE|Power_Seeking_FALSE,Misalignment_FALSE,Capability_FALSE)": "0.0001"}}
 + [Power_Seeking]: AI systems with instrumental convergence toward power-seeking behavior.
   {"instantiations": ["TRUE", "FALSE"], 
    "priors": {"p(TRUE)": "0.8", "p(FALSE)": "0.2"}}
 + [Misalignment]: AI systems with goals misaligned with human values.
   {"instantiations": ["TRUE", "FALSE"], 
    "priors": {"p(TRUE)": "0.7", "p(FALSE)": "0.3"}}
 + [Capability]: AI systems achieving capabilities sufficient for existential threat.
   {"instantiations": ["TRUE", "FALSE"], 
    "priors": {"p(TRUE)": "0.6", "p(FALSE)": "0.4"}}
```

### 6.3 AI Governance Example: The Coordination Crisis

```
[Coordination_Crisis]: Systematic governance failure despite unprecedented investment in AI safety.
{"instantiations": ["TRUE", "FALSE"], 
 "priors": {"p(TRUE)": "0.75", "p(FALSE)": "0.25"},
 "posteriors": {"p(TRUE|Narrative_Fragmentation_TRUE,Computing_Failure_TRUE)": "0.95",
                "p(TRUE|Narrative_Fragmentation_TRUE,Computing_Failure_FALSE)": "0.7",
                "p(TRUE|Narrative_Fragmentation_FALSE,Computing_Failure_TRUE)": "0.8",
                "p(TRUE|Narrative_Fragmentation_FALSE,Computing_Failure_FALSE)": "0.1"}}
 + [Narrative_Fragmentation]: Communities operating with different terminologies, priorities, and theories of change.
   {"instantiations": ["TRUE", "FALSE"], 
    "priors": {"p(TRUE)": "0.9", "p(FALSE)": "0.1"}}
 + [Computing_Failure]: Organizations functioning as independent processors without shared protocols.
   {"instantiations": ["TRUE", "FALSE"], 
    "priors": {"p(TRUE)": "0.85", "p(FALSE)": "0.15"}}
 + [P_Doom_Increase]: Increased probability of existential catastrophe due to coordination failures.
   {"instantiations": ["TRUE", "FALSE"], 
    "priors": {"p(TRUE)": "0.8", "p(FALSE)": "0.2"},
    "posteriors": {"p(TRUE|Coordination_Crisis_TRUE)": "0.9",
                   "p(TRUE|Coordination_Crisis_FALSE)": "0.2"}}
```

## 7. Integration with AMTAIR Codebase

### 7.1 The Extraction Pipeline

The AMTAIR codebase processes BayesDown through several transformation steps:

1. **Text Preprocessing**: Comments removal and initial cleaning
2. **Structure Extraction**: Identification of nodes, descriptions, and hierarchical relationships
3. **Metadata Parsing**: Extraction of instantiations, priors, and posteriors
4. **Relationship Establishment**: Determining parent-child connections
5. **DataFrame Construction**: Organizing extracted data into structured format
6. **Network Building**: Constructing Bayesian networks from structured data

Understanding this pipeline helps in creating BayesDown that integrates seamlessly with the system.

### 7.2 Working with the Colab Notebook

The AMTAIR Prototype Colab Notebook provides a working environment for processing BayesDown files. Key sections include:

- **Library Setup (0.1)**: Import necessary dependencies
- **GitHub Connection (0.2-0.3)**: Load example files from repository
- **BayesDown Extraction (3.0-3.4)**: Process BayesDown into structured data
- **Analysis and Visualization (4.0)**: Create interactive Bayesian networks

To test a BayesDown file in the notebook:

1. Upload the file to a GitHub repository or use a direct URL
2. Update the file path in Section 0.3 or 3.1 to point to your file
3. Run the extraction functions to process your BayesDown
4. Check the output DataFrame for correct extraction
5. Run the visualization function to see the Bayesian network

### 7.3 Parser Expectations

The BayesDown parser has specific expectations that must be met:

1. **Line Integration**: All metadata must be on the same line as the node definition (line breaks within metadata will cause parsing failures)
2. **Valid JSON**: Metadata must be valid JSON (watch for common errors like missing commas)
3. **Required Fields**: The `instantiations` field must be present for all nodes
4. **Consistent Indentation**: Use consistent spacing for indentation levels
5. **Valid Relationships**: References to other nodes must match exactly

## 8. Verification and Validation

### 8.1 Syntax Validation

To verify BayesDown syntax:

1. **Check JSON validity** of all metadata blocks
2. **Verify required fields** (instantiations) exist for all nodes
3. **Ensure probability coherence** (values between 0-1, priors sum to 1)
4. **Validate node references** in hierarchies and posteriors
5. **Check for formatting issues** like line breaks in metadata

### 8.2 Semantic Validation

Beyond syntax, validate semantic correctness:

1. **Causal consistency**: Ensure causal relationships make logical sense
2. **Probability coherence**: Check that conditional probabilities are logically coherent
3. **Completeness**: Verify all important variables from the source are captured
4. **Fidelity**: Check that the extraction preserves the original argument's meaning

### 8.3 Quality Checklist

Use this checklist for manual or automated validation:

- [ ] All node definitions have descriptions
- [ ] All nodes have instantiations defined
- [ ] Prior probabilities sum to approximately 1 for each node
- [ ] Conditional probabilities reference actual parent nodes
- [ ] Hierarchical structure accurately reflects the original argument
- [ ] No duplicate node definitions with conflicting metadata
- [ ] Metadata is valid JSON without syntax errors
- [ ] Node references are consistent throughout the document

## 9. Best Practices

### 9.1 Naming Conventions

- **Node Names**: Use CamelCase without spaces (e.g., `PowerSeeking`, `AICapabilities`)
- **Instantiations**: For binary variables, use `TRUE` and `FALSE` (or `varname_TRUE` and `varname_FALSE`)
- **Probability Expressions**: Follow the p(X|Y) format consistently

### 9.2 Structure and Organization

- **Modularity**: Break complex arguments into logical components
- **Hierarchy**: Use hierarchy to reflect causal or inferential structure
- **Granularity**: Choose appropriate detail level (too many nodes increases complexity)
- **Progressive Disclosure**: Start with high-level nodes, then add detail through children

### 9.3 Probability Estimation

- **Consistency**: Use consistent methods for estimating probabilities
- **Ranges**: When uncertain, provide ranges or conservative estimates
- **Calibration**: Calibrate probability estimates across different nodes
- **Transparency**: Document assumptions behind probability estimates

### 9.4 Documentation

- **Source Attribution**: Include references to source material
- **Assumption Documentation**: Use comments to explain key assumptions
- **Version Control**: Track changes to BayesDown files over time
- **Meta-uncertainty**: Note confidence levels in extractions when appropriate

## 10. Common Errors and Solutions

### 10.1 Syntax Errors

|Error|Example|Solution|
|---|---|---|
|**Metadata not on same line**|`[Node]: Description`<br>`{"instantiations": ["TRUE", "FALSE"]}`|Put metadata on the same line:<br>`[Node]: Description {"instantiations": ["TRUE", "FALSE"]}`|
|**Invalid JSON**|`[Node]: Description {"instantiations": ["TRUE" "FALSE"]}`|Fix JSON syntax:<br>`[Node]: Description {"instantiations": ["TRUE", "FALSE"]}`|
|**Missing required field**|`[Node]: Description {"priors": {"p(TRUE)": "0.7"}}`|Include instantiations:<br>`[Node]: Description {"instantiations": ["TRUE", "FALSE"], "priors": {"p(TRUE)": "0.7"}}`|
|**Probabilities don't sum to 1**|`{"priors": {"p(TRUE)": "0.7", "p(FALSE)": "0.4"}}`|Correct to sum to 1:<br>`{"priors": {"p(TRUE)": "0.6", "p(FALSE)": "0.4"}}`|
|**Inconsistent indentation**|Mixing spaces and tabs for indentation|Use consistent indentation (preferably spaces)|

### 10.2 Extraction Errors

|Error|Description|Solution|
|---|---|---|
|**Missed variables**|Important concepts from source not captured as nodes|Review source material comprehensively before extraction|
|**Overlooked relationships**|Failing to capture implied causal connections|Look for implicit causal language and context clues|
|**Probability misinterpretation**|Misinterpreting qualitative uncertainty expressions|Use consistent mapping for terms like "likely", "possible"|
|**Overly complex structure**|Creating unnecessarily complex hierarchies|Simplify to essential causal relationships|
|**Inconsistent node references**|Referring to the same concept with different node names|Standardize node naming and reference consistently|

### 10.3 Processing Errors

|Error|Description|Solution|
|---|---|---|
|**Parser fails on valid BayesDown**|Parser error despite seemingly correct syntax|Check for invisible characters or encoding issues|
|**Visualization shows unexpected structure**|Network visualization doesn't match expected structure|Review parent-child relationships in the raw BayesDown|
|**Missing probability information**|Some probability values not appearing in output|Check format of probability expressions|
|**Node duplication**|Same node appears multiple times in network|Check for inconsistent capitalization or spacing in node names|
|**Memory errors during processing**|Out of memory errors when processing large files|Break into smaller files or increase memory allocation|

## 11. Working with LLMs for Extraction

### 11.1 Optimal Prompting Strategies

When using LLMs to extract BayesDown from sources, these approaches work best:

1. **Context Setting**: Provide clear context on the purpose of the extraction and BayesDown format
2. **Step-by-Step Instructions**: Break down the extraction process into discrete steps
3. **Few-Shot Examples**: Provide relevant examples of similar extractions
4. **Format Templates**: Include explicit formatting templates for the expected output
5. **Validation Guidelines**: Instruct the LLM to validate its own output for common errors

### 11.2 Example Extraction Prompt

```
You are an expert in formal modeling and causal reasoning. I need you to extract a structured BayesDown representation from the following text about AI risk.

Follow these steps:

1. Identify the key variables/concepts mentioned in the text
2. For each variable, determine possible states (typically TRUE/FALSE)
3. Identify causal relationships between variables
4. Extract or infer probability judgments for each relationship
5. Format the result in BayesDown syntax

BayesDown format looks like this:
[VariableName]: Description of the variable. {"instantiations": ["state1", "state2"], "priors": {"p(state1)": "0.7", "p(state2)": "0.3"}}
 + [RelatedVariable]: Description of this related variable. {"instantiations": ["TRUE", "FALSE"], "posteriors": {"p(RelatedVariable|VariableName)": "0.8"}}

For qualitative probability terms, use this mapping:
- "Almost certain" → 0.95
- "Very likely" → 0.9
- "Likely" → 0.75
- "Possibly" → 0.5
- "Unlikely" → 0.25
- "Very unlikely" → 0.1
- "Almost impossible" → 0.05

Here's the text to analyze:
[SOURCE TEXT HERE]
```

### 11.3 Post-Processing LLM Output

After receiving LLM-generated BayesDown:

1. **Syntax Check**: Verify all JSON is valid and on the same line as node definitions
2. **Probability Validation**: Ensure probability distributions sum to 1
3. **Structure Review**: Confirm the hierarchical structure matches the source argument
4. **Completeness Check**: Verify all key concepts are included
5. **Manual Refinement**: Adjust any probabilities or relationships that seem incorrect

## 12. Advanced Usage in the AMTAIR Project

### 12.1 Extending BayesDown

For the AMTAIR project, BayesDown can be extended with additional metadata fields:

```json
{
  "instantiations": ["state1", "state2"],
  "priors": { "p(state1)": "0.7", "p(state2)": "0.3" },
  "posteriors": { "p(state1|parent1)": "0.8" },
  "sources": ["Author (Year)", "URL"],
  "confidence": "high",
  "notes": "Additional context about this variable"
}
```

These extensions won't interfere with core functionality but can enrich the extracted models.

### 12.2 Integration with Live Forecasting

The AMTAIR project aims to integrate BayesDown models with live forecasting data. When creating models intended for this integration:

1. Use node names that can be easily mapped to forecasting questions
2. Include metadata about relevant forecasting platforms when available
3. Structure nodes to facilitate updating from forecasting data
4. Consider using confidence intervals for probabilities that may be updated

### 12.3 Policy Impact Evaluation

For BayesDown models focused on policy evaluation:

1. Include specific intervention nodes that represent policy actions
2. Structure hierarchies to make intervention points explicit
3. Use conditional probabilities to model intervention effects
4. Include counterfactual scenarios in your model structure

## 13. Conclusion: The Art and Science of BayesDown

BayesDown walks the line between art and science—requiring both creative interpretation to extract the essence of arguments and technical precision to represent them formally. Like any translation process, it involves judgment calls about what to include, how to structure relationships, and how to quantify uncertainties.

The most valuable BayesDown extractions capture not just the surface claims of the source material but the underlying mental models—the causal reasoning that connects facts to conclusions. In this way, it functions as a kind of cognitive archaeology, unearthing the implicit theories that shape our thinking about complex domains like AI governance.

As you become fluent in BayesDown, you'll develop an intuition for identifying the critical variables and relationships in any argument. This skill serves not only the immediate purpose of model construction but also enhances your own understanding of the complex causal landscapes that surround transformative technologies.

Remember that the ultimate goal isn't perfect representation—it's useful abstraction that enables better reasoning, more effective coordination, and wiser governance decisions in domains where the stakes couldn't be higher.
