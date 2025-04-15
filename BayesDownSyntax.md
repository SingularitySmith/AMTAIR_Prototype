# BayesDown Syntax Specification - Revised

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

### 3.2 Hierarchy Definition - CORRECTED

```
[Child_Node]: Description of child. {"metadata": "in JSON format"}
 + [Parent_Node]: Description of parent. {"metadata": "in JSON format"}
   + [Grandparent_Node]: Description of grandparent. {"metadata": "in JSON format"}
```

In BayesDown, the hierarchy flows in reverse compared to typical tree structures:
- The top-level node represents the effect or outcome
- Nodes preceded by `+` symbols represent parent causes (direct influences)
- Further indented nodes with additional `+` symbols represent grandparent causes (indirect influences)

This structure maps naturally to Bayesian networks where arrows flow from causes (parents) to effects (children).

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
3. Extract explicit probability judgments
4. Structure the hierarchical relationships between variables
5. Format your response using BayesDown syntax as follows:

[Effect_Variable]: Description of this variable as the effect. {"instantiations": ["state1", "state2"], "priors": {"p(state1)": "0.7", "p(state2)": "0.3"}}
 + [Cause_Variable1]: Description of this causal variable. {"instantiations": ["TRUE", "FALSE"], "posteriors": {"p(Cause_Variable1|Effect_Variable)": "0.8", "p(Cause_Variable1|not_Effect_Variable)": "0.2"}}
   + [Root_Cause]: Description of this root cause. {"instantiations": ["TRUE", "FALSE"]}

SOURCE TEXT:
[paste source text here]
```

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
[Outcome]: Description of the outcome.
{"instantiations": ["TRUE", "FALSE"], "priors": {"p(TRUE)": "0.4", "p(FALSE)": "0.6"}}
 + [Cause1]: Description of first cause.
   {"instantiations": ["TRUE", "FALSE"], "priors": {"p(TRUE)": "0.6", "p(FALSE)": "0.4"}}
 + [Cause2]: Description of second cause.
   {"instantiations": ["TRUE", "FALSE"], "priors": {"p(TRUE)": "0.7", "p(FALSE)": "0.3"}}
   + [Cause1]
```

The repeated reference to `[Cause1]` indicates that it influences both `[Outcome]` directly and via `[Cause2]`.

### 5.3 Comments

You can include comments that will be ignored by the parser:

```
/* This is a comment explaining something important about the following node */
[Node_Name]: Description {...metadata...}
```

## Comprehensive BayesDown Syntax Definition

```
# BayesDown Syntax Definition

## Core Concepts
BayesDown represents a Bayesian network as a hierarchical argument structure with probabilistic metadata.
- Nodes represent variables or statements with possible states (instantiations)
- Edges represent causal or inferential relationships
- Probabilities quantify uncertainty in both prior beliefs and conditional relationships

## Node Definition
[Node_Name]: Description of the node or variable. {"metadata": "in JSON format"}
<Node_Name>: Alternative syntax for node definition. {"metadata": "in JSON format"}

## Hierarchy Definition
[Effect_Node]: Description of effect. {"metadata": "in JSON format"}
 + [Cause_Node]: Description of direct cause. {"metadata": "in JSON format"}
   + [Root_Cause]: Description of indirect cause. {"metadata": "in JSON format"}

The hierarchy flows from effects (top) to causes (indented with + symbols).
Multiple causes can influence the same effect by listing them at the same indentation level.

## Metadata Structure
{
  "instantiations": ["state1", "state2", ...],  // Required - all possible states
  "priors": {  // Optional - unconditional probabilities
    "p(state1)": "0.7",  // Probability of state1
    "p(state2)": "0.3"   // Probability of state2 (must sum to 1.0)
  },
  "posteriors": {  // Optional - conditional probabilities
    // Single parent, single state
    "p(effect_state|cause_state)": "0.8",
    
    // Single parent, negated state
    "p(effect_state|not_cause_state)": "0.2",
    
    // Multiple parents, all positive states
    "p(effect_state|cause1_state,cause2_state)": "0.9",
    
    // Multiple parents, mixed states
    "p(effect_state|cause1_state,not_cause2_state)": "0.6",
    
    // Multiple parents, all negated
    "p(effect_state|not_cause1_state,not_cause2_state)": "0.1",
    
    // Complete set of conditions must be specified for all parent state combinations
    // For binary variables with n parents, need 2^n conditional probabilities
  }
}

## Validation Rules
1. Metadata must appear on the same line as the node definition
2. Metadata must be valid JSON with required "instantiations" field
3. Prior probabilities must sum to 1.0 for each node
4. Conditional probabilities must specify values for all parent state combinations
5. All probability values must be between 0.0 and 1.0
6. Referenced nodes must be defined elsewhere in the document
7. Node names must be unique within the document

## Comments
/* Comments use this syntax and are ignored by the parser */

## Advanced Features
1. Multi-state variables (more than two states)
2. Cross-referenced nodes (same node appearing multiple times)
3. Complex conditional probability expressions

## Example Fragments
[Outcome]: A variable of interest. {
  "instantiations": ["outcome_TRUE", "outcome_FALSE"],
  "priors": {
    "p(outcome_TRUE)": "0.3",
    "p(outcome_FALSE)": "0.7"
  },
  "posteriors": {
    "p(outcome_TRUE|cause1_TRUE,cause2_TRUE)": "0.9",
    "p(outcome_TRUE|cause1_TRUE,cause2_FALSE)": "0.7",
    "p(outcome_TRUE|cause1_FALSE,cause2_TRUE)": "0.5",
    "p(outcome_TRUE|cause1_FALSE,cause2_FALSE)": "0.1",
    "p(outcome_FALSE|cause1_TRUE,cause2_TRUE)": "0.1",
    "p(outcome_FALSE|cause1_TRUE,cause2_FALSE)": "0.3",
    "p(outcome_FALSE|cause1_FALSE,cause2_TRUE)": "0.5",
    "p(outcome_FALSE|cause1_FALSE,cause2_FALSE)": "0.9"
  }
}
 + [Cause1]: First causal factor. {
   "instantiations": ["cause1_TRUE", "cause1_FALSE"],
   "priors": {
     "p(cause1_TRUE)": "0.4",
     "p(cause1_FALSE)": "0.6"
   }
 }
 + [Cause2]: Second causal factor. {
   "instantiations": ["cause2_TRUE", "cause2_FALSE"],
   "priors": {
     "p(cause2_TRUE)": "0.5",
     "p(cause2_FALSE)": "0.5"
   },
   "posteriors": {
     "p(cause2_TRUE|root_TRUE)": "0.8",
     "p(cause2_TRUE|root_FALSE)": "0.3",
     "p(cause2_FALSE|root_TRUE)": "0.2",
     "p(cause2_FALSE|root_FALSE)": "0.7"
   }
 }
   + [Root]: Root cause. {
     "instantiations": ["root_TRUE", "root_FALSE"],
     "priors": {
       "p(root_TRUE)": "0.25",
       "p(root_FALSE)": "0.75"
     }
   }
```
