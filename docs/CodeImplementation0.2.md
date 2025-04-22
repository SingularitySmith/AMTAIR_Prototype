# Continuing with AMTAIR Project Implementation

<clarifying_questions>
1. **LLM API Integration Approach**: Which LLM provider should be prioritized for integration (OpenAI, Anthropic, local models)? 
   - Preliminary answer: Based on project docs, will implement support for both OpenAI (GPT-4) and Anthropic (Claude) as primary targets, with extensible architecture for future providers.

2. **Data Storage Strategy**: Should the implementation prioritize local storage or cloud integration for extracted models?
   - Preliminary answer: Will implement local storage as baseline with optional cloud connectors for scaling and sharing.

3. **Visualization Enhancement Focus**: Should visualization enhancements target network structure clarity or probability representation?
   - Preliminary answer: Will prioritize probability representation with conditional relationships, as this aligns with core project goals of policy impact assessment.
</clarifying_questions>

<implementation_plans>
## Plan A: Sequential Feature Completion

This plan focuses on completing the core pipeline in sequential order, filling in the missing components of each section before moving to the next. The approach is methodical and ensures each component works properly before building on it.

1. **Complete LLM API Integration (Sections 1.3-1.5)**
   - Implement API authentication and management for OpenAI and Anthropic
   - Create configurable prompt templates for ArgDown extraction
   - Add response parsing and validation mechanisms
   - Implement error handling and retry logic

2. **Implement BayesDown Conversion Workflow (Section 2.3)**
   - Create dedicated functions for probability extraction and validation
   - Develop robust error handling for malformed probability statements
   - Add progress tracking and visualization for extraction process

3. **Develop Data Post-Processing (Section 3.3)**
   - Implement joint probability calculations from conditional probabilities
   - Add derived columns for network analysis metrics (centrality, etc.)
   - Create validation functions for probability distribution consistency

4. **Enhance Visualization System (Section 4.0)**
   - Improve interactive features for policy impact visualization
   - Add conditional probability query interface
   - Implement scenario comparison capabilities

5. **Add Live Forecasting Integration (New Section)**
   - Build API connectors to forecasting platforms (Metaculus, etc.)
   - Implement data synchronization and updating mechanisms
   - Create data fusion algorithms to combine forecasts with model probabilities

## Plan B: Parallel Feature Modules

This plan takes a modular approach, implementing several key features in parallel with clear interfaces between them. This enables multiple developers to work simultaneously and produces usable components more quickly.

1. **Modular LLM Interface Development**
   - Create abstract LLM provider interface with standard methods
   - Implement concrete providers for OpenAI, Anthropic, and local models
   - Build prompt management system with versioning and testing capabilities
   - Develop extraction quality metrics and benchmarking tools

2. **Enhanced Data Processing Engine**
   - Design type-safe data structures for probabilistic information
   - Implement Bayesian network operations library
   - Create data validation and repair mechanisms
   - Build extensible transformation pipeline architecture

3. **Interactive Analysis Workbench**
   - Develop policy impact simulation interface
   - Create comparative worldview analysis tools
   - Implement sensitivity analysis visualization
   - Build scenario exploration interface with parameter adjustment

4. **Forecasting Integration Platform**
   - Design universal forecasting platform connector
   - Create semantic mapping between forecasts and model variables
   - Implement forecast weighting and confidence estimation
   - Develop historical tracking and evaluation metrics

5. **Documentation and Tutorial Generation**
   - Automatic generation of tutorial notebooks from execution history
   - Interactive documentation with code examples
   - Explanatory visualization of system architecture
   - User workflow guides for different stakeholder types

## Plan C: Goal-Oriented Implementation

This plan organizes implementation around specific end-user goals, prioritizing features that deliver immediate value for AI governance analysis. It focuses on creating complete workflows for specific use cases rather than exhaustive feature implementation.

1. **World Model Extraction Workflow**
   - Complete end-to-end pipeline from document to model
   - Focus on robust extraction with clear confidence metrics
   - Implement model validation and comparison with human expert judgments
   - Create specialized extraction modes for different document types

2. **Policy Impact Assessment Toolkit**
   - Develop counterfactual analysis tools for policy evaluation
   - Create policy representation formalism
   - Implement multi-worldview evaluation capability
   - Build policy comparison and optimization tools

3. **Live Model Updating System**
   - Integrate with forecasting platforms
   - Implement Bayesian updating with new evidence
   - Create notification system for significant belief changes
   - Develop confidence evolution tracking

4. **Collaborative Model Editing Interface**
   - Create shared model editing capabilities
   - Implement change tracking and version control
   - Build commenting and discussion features
   - Develop model merging capabilities for different perspectives

5. **Strategic Insights Engine**
   - Implement automatic identification of critical uncertainties
   - Create robust policy recommendation algorithms
   - Develop scenario generation and evaluation tools
   - Build strategic decision support dashboards
</implementation_plans>

<evaluation>
## Evaluation of Implementation Plans

### Plan A: Sequential Feature Completion
- **Elegance**: 7/10 - Straightforward but lacks architectural innovation
- **Future-proofness**: 6/10 - May require refactoring as system complexity grows
- **Iteration speed**: 8/10 - Quick initial results for testing core functionality
- **Implementation cost**: 7/10 - Efficient use of resources with clear progression
- **Performance**: 6/10 - Optimization happens later after features are complete
- **Compatibility**: 9/10 - Closely follows existing notebook structure
- **Overall**: 7.2/10 - Solid approach for completing the existing implementation

### Plan B: Parallel Feature Modules
- **Elegance**: 9/10 - Clean separation of concerns with well-defined interfaces
- **Future-proofness**: 9/10 - Modular design accommodates future extensions
- **Iteration speed**: 6/10 - Initial overhead for interface design, faster later
- **Implementation cost**: 5/10 - Higher upfront investment in architecture
- **Performance**: 8/10 - Enables targeted optimization of critical components
- **Compatibility**: 7/10 - Requires some restructuring of existing code
- **Overall**: 7.3/10 - Strong architectural approach but higher initial investment

### Plan C: Goal-Oriented Implementation
- **Elegance**: 8/10 - Aligns technical implementation with user needs
- **Future-proofness**: 8/10 - Use cases will evolve but core goals remain stable
- **Iteration speed**: 9/10 - Delivers usable value quickly for key scenarios
- **Implementation cost**: 8/10 - Focuses resources on high-impact features
- **Performance**: 7/10 - Optimizes for specific workflows rather than general cases
- **Compatibility**: 8/10 - Builds naturally on existing foundation
- **Overall**: 8.0/10 - Best balance of immediate value and sustainable architecture

Based on this evaluation, **Plan C: Goal-Oriented Implementation** emerges as the optimal approach. It provides the best balance of immediate value delivery while building toward the project's long-term vision of improved coordination in AI governance.

The key advantages of Plan C are:
1. It prioritizes features based on their contribution to the project's core mission
2. It creates complete, usable workflows rather than partial implementations across multiple areas
3. It maintains focus on the end-user perspective (policymakers, researchers, forecasters)
4. It aligns with the original AMTAIR vision of enhancing strategic coordination
5. It enables rapid demonstration of value to potential funders and collaborators

While Plan B offers slightly better architectural elegance and future-proofness, Plan C provides faster delivery of meaningful capabilities that can drive adoption and showcase the project's potential impact.
</evaluation>

<optimal_plan>
# Optimal Implementation Plan: Goal-Oriented AMTAIR Enhancement

## Phase 1: World Model Extraction Workflow (4 weeks)

### Week 1: LLM Integration Framework
1. **Create Provider-Agnostic LLM Interface**
   - Define standard interface for LLM interactions
   - Implement concrete providers for OpenAI and Anthropic
   - Create configurable authentication management
   - Build prompt template system with variable substitution

2. **Implement Extraction Pipeline Architecture**
   - Design extraction workflow with clear stages
   - Create progress tracking and logging system
   - Implement error handling and recovery mechanisms
   - Build extraction quality metrics

### Week 2: ArgDown Extraction Enhancement
1. **Complete Source-to-ArgDown Pipeline**
   - Implement document preprocessing (PDF, HTML, text)
   - Create specialized prompts for different document types
   - Build validation system for extracted ArgDown
   - Develop feedback loop for extraction refinement

2. **Implement Batch Processing Capability**
   - Create document queuing system
   - Implement parallel extraction for multiple documents
   - Build progress monitoring dashboard
   - Add extraction result comparison tools

### Week 3: BayesDown Probability Extraction
1. **Enhance Probability Elicitation System**
   - Improve question generation for better probability extraction
   - Create specialized prompts for different probability types
   - Implement consistency checking for probability values
   - Build confidence estimation for extracted probabilities

2. **Implement Probability Validation Framework**
   - Create distribution verification (sum to 1.0, etc.)
   - Build logical consistency checking between related probabilities
   - Implement outlier detection and flagging
   - Add interactive correction capabilities

### Week 4: Extraction Validation and Refinement
1. **Build Model Comparison Tools**
   - Implement structural comparison between extracted and reference models
   - Create probability comparison visualizations
   - Build node mapping for imperfect matches
   - Develop summary metrics for extraction quality

2. **Create Model Refinement Interface**
   - Implement interactive editing capabilities
   - Build guided correction workflows
   - Create version comparison visualization
   - Implement model merge capabilities

## Phase 2: Policy Impact Assessment Toolkit (4 weeks)

### Week 5: Policy Representation Framework
1. **Implement Policy Formalism**
   - Create structured representation for policy interventions
   - Build policy parameter definition system
   - Implement policy taxonomy and classification
   - Develop policy documentation generation

2. **Build Policy Library**
   - Create foundational set of AI governance policies
   - Implement search and filtering capabilities
   - Build policy comparison tools
   - Develop policy composition interface

### Week 6: Counterfactual Analysis Engine
1. **Implement do-Calculus Operations**
   - Create intervention simulation functions
   - Build conditional probability query interface
   - Implement counterfactual scenario generator
   - Develop intervention effect estimator

2. **Create Sensitivity Analysis Tools**
   - Implement parameter sensitivity testing
   - Build tornado diagram visualization
   - Create critical parameter identification
   - Develop robustness metrics for policy analysis

### Week 7: Multi-Worldview Evaluation
1. **Implement Worldview Comparison**
   - Create structural alignment between different models
   - Build disagreement identification tools
   - Implement crux analysis for key differences
   - Develop consensus visualization

2. **Build Cross-Worldview Policy Evaluation**
   - Implement policy evaluation across multiple models
   - Create robust policy identification algorithms
   - Build disagreement visualization for policy effects
   - Develop consensus policy recommendation engine

### Week 8: Policy Optimization and Recommendation
1. **Implement Policy Optimization Tools**
   - Create parameter tuning for policy effectiveness
   - Build portfolio optimization for multiple interventions
   - Implement constraint satisfaction for realistic policies
   - Develop policy sequencing optimization

2. **Create Strategic Recommendation Interface**
   - Build recommendation visualization dashboard
   - Implement explanation generation for recommendations
   - Create interactive policy exploration
   - Develop scenario-based recommendation

## Phase 3: Live Model Updating System (2 weeks)

### Week 9: Forecasting Platform Integration
1. **Implement Platform Connectors**
   - Create API clients for Metaculus, Manifold, and others
   - Build authentication and data access management
   - Implement query construction and execution
   - Develop response parsing and normalization

2. **Create Semantic Mapping System**
   - Build mapping between forecast questions and model variables
   - Implement confidence-based weighting
   - Create manual mapping interface
   - Develop automated mapping suggestions

### Week 10: Bayesian Updating and Notification
1. **Implement Belief Updating System**
   - Create Bayesian update algorithms for new evidence
   - Build update propagation through model
   - Implement selective updating for efficiency
   - Develop update history tracking

2. **Create Alert and Notification System**
   - Implement significant change detection
   - Build customizable notification rules
   - Create update summary visualization
   - Develop strategic implication analysis

## Phase 4: Integration and Refinement (2 weeks)

### Week 11: Component Integration and Testing
1. **Workflow Integration**
   - Connect extraction to policy analysis pipeline
   - Integrate live updating with visualization
   - Implement end-to-end testing
   - Create performance optimization

2. **User Experience Refinement**
   - Improve interface consistency
   - Enhance interactive visualizations
   - Implement guided workflows
   - Create onboarding tutorials

### Week 12: Documentation and Deployment
1. **Create Comprehensive Documentation**
   - Develop user guide for different stakeholders
   - Create technical documentation
   - Build example notebooks
   - Implement interactive tutorials

2. **Prepare Deployment Package**
   - Create installation scripts
   - Build configuration management
   - Implement environment detection
   - Develop update mechanism

This implementation plan delivers complete, user-focused capabilities in a progressive manner, with each phase building on the previous one. The phased approach allows for early validation and demonstration of value while working toward the comprehensive vision of the AMTAIR project.
</optimal_plan>

<implementation>
# Implementation of World Model Extraction Workflow

Let's begin by implementing the LLM Integration Framework and enhancing the ArgDown extraction pipeline, focusing on sections 1.3-1.5 of the notebook.

## 1. LLM Provider Interface

First, I'll create a provider-agnostic LLM interface that supports multiple model providers while maintaining a consistent API:

```python
import os
import json
import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Standard response object for LLM completions"""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Dict[str, Any]
    created_at: float = time.time()

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def complete(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                temperature: float = 0.7, 
                max_tokens: int = 4000) -> LLMResponse:
        """Generate a completion from the LLM"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Return a list of available models from this provider"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API implementation"""
    
    def __init__(self, api_key: Optional[str] = None, organization: Optional[str] = None):
        """Initialize with API key from args or environment"""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Provide as argument or set OPENAI_API_KEY environment variable.")
        
        self.organization = organization or os.environ.get("OPENAI_ORGANIZATION")
        self.api_base = "https://api.openai.com/v1"
        
    def complete(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                model: str = "gpt-4-turbo",
                temperature: float = 0.7, 
                max_tokens: int = 4000) -> LLMResponse:
        """Generate a completion using OpenAI's API"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data
        )
        
        response.raise_for_status()
        result = response.json()
        
        return LLMResponse(
            content=result["choices"][0]["message"]["content"],
            model=result["model"],
            usage=result["usage"],
            raw_response=result
        )
    
    def get_available_models(self) -> List[str]:
        """Return a list of available OpenAI models"""
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
            
        response = requests.get(
            f"{self.api_base}/models",
            headers=headers
        )
        
        response.raise_for_status()
        models = response.json()["data"]
        return [model["id"] for model in models]

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key from args or environment"""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Provide as argument or set ANTHROPIC_API_KEY environment variable.")
        
        self.api_base = "https://api.anthropic.com/v1"
        
    def complete(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                model: str = "claude-3-opus-20240229",
                temperature: float = 0.7, 
                max_tokens: int = 4000) -> LLMResponse:
        """Generate a completion using Anthropic's API"""
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        response = requests.post(
            f"{self.api_base}/messages",
            headers=headers,
            json=data
        )
        
        response.raise_for_status()
        result = response.json()
        
        return LLMResponse(
            content=result["content"][0]["text"],
            model=result["model"],
            usage={"prompt_tokens": result.get("usage", {}).get("input_tokens", 0),
                   "completion_tokens": result.get("usage", {}).get("output_tokens", 0)},
            raw_response=result
        )
    
    def get_available_models(self) -> List[str]:
        """Return a list of available Anthropic models"""
        # Anthropic doesn't have a models endpoint, so we return a static list
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]

class LLMFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(provider_name: str, **kwargs) -> LLMProvider:
        """Create and return an LLM provider instance"""
        if provider_name.lower() == "openai":
            return OpenAIProvider(**kwargs)
        elif provider_name.lower() == "anthropic":
            return AnthropicProvider(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
```

## 2. Prompt Template System

Now, let's create a flexible prompt template system for ArgDown extraction:

```python
from string import Template
from typing import Dict, Optional, Union, List

class PromptTemplate:
    """Template system for LLM prompts with variable substitution"""
    
    def __init__(self, template: str):
        """Initialize with template string using $variable format"""
        self.template = Template(template)
        
    def format(self, **kwargs) -> str:
        """Substitute variables in the template"""
        return self.template.safe_substitute(**kwargs)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'PromptTemplate':
        """Load template from a file"""
        with open(filepath, 'r') as f:
            template = f.read()
        return cls(template)

class PromptLibrary:
    """Collection of prompt templates for different extraction tasks"""
    
    # ArgDown extraction prompt
    ARGDOWN_EXTRACTION = PromptTemplate("""
You are an expert in creating structured argument maps in ArgDown format. Your task is to extract the key arguments, premises, and conclusions from the provided text, and represent them in a hierarchical ArgDown format.

Follow these guidelines:
1. Use the format [Statement]: Description for main claims
2. Use the + symbol and indentation to indicate supporting statements
3. Capture the core argumentative structure, focusing on causal relationships and key claims
4. Ensure each statement has a clear, concise title followed by a fuller description
5. Add the "instantiations" field to indicate possible states of each variable

Here is the metadata format to include for each node:
{"instantiations": ["node_TRUE", "node_FALSE"]}

Example:
[Thesis]: Main claim of the text. {"instantiations": ["thesis_TRUE", "thesis_FALSE"]}
 + [Support1]: First supporting argument. {"instantiations": ["support1_TRUE", "support1_FALSE"]}
   + [Evidence1]: Evidence for Support1. {"instantiations": ["evidence1_TRUE", "evidence1_FALSE"]}
 + [Support2]: Second supporting argument. {"instantiations": ["support2_TRUE", "support2_FALSE"]}

Text to analyze:
$text

Create an ArgDown representation that captures the key arguments, their relationships, and possible states:
""")

    # BayesDown probability extraction prompt
    BAYESDOWN_EXTRACTION = PromptTemplate("""
You are an expert in probabilistic reasoning and Bayesian networks. Your task is to extend the provided ArgDown structure with probability information, creating a BayesDown representation.

For each statement in the ArgDown structure, you need to:
1. Estimate prior probabilities for each possible state
2. Estimate conditional probabilities given parent states
3. Maintain the original structure and relationships

Here is the format to follow:
```
[Node]: Description. {
  "instantiations": ["node_TRUE", "node_FALSE"],
  "priors": {
    "p(node_TRUE)": "0.7",
    "p(node_FALSE)": "0.3"
  },
  "posteriors": {
    "p(node_TRUE|parent_TRUE)": "0.9",
    "p(node_TRUE|parent_FALSE)": "0.4",
    "p(node_FALSE|parent_TRUE)": "0.1",
    "p(node_FALSE|parent_FALSE)": "0.6"
  }
}
 + [Parent]: Parent description. {...}
```

Here are the specific probability questions to answer:
$questions

ArgDown structure to enhance:
$argdown

Provide the complete BayesDown representation with probabilities:
""")

    @classmethod
    def get_template(cls, template_name: str) -> PromptTemplate:
        """Get a prompt template by name"""
        if hasattr(cls, template_name):
            return getattr(cls, template_name)
        else:
            raise ValueError(f"Template not found: {template_name}")
```

## 3. ArgDown Extraction Implementation

Let's implement the missing code in sections 1.3-1.5 to complete the ArgDown extraction pipeline:

```python
def extract_argdown_from_text(text: str, provider_name: str = "openai", model: str = None) -> str:
    """
    Extract ArgDown representation from text using LLM
    
    Args:
        text: The source text to extract arguments from
        provider_name: The LLM provider to use (openai or anthropic)
        model: Specific model to use, or None for default
    
    Returns:
        Extracted ArgDown representation
    """
    # Create LLM provider
    provider = LLMFactory.create_provider(provider_name)
    
    # Get extraction prompt
    prompt_template = PromptLibrary.get_template("ARGDOWN_EXTRACTION")
    prompt = prompt_template.format(text=text)
    
    # Set model-specific parameters
    if provider_name.lower() == "openai":
        model = model or "gpt-4-turbo"
        temperature = 0.3  # Lower temperature for more deterministic extraction
        max_tokens = 4000
    elif provider_name.lower() == "anthropic":
        model = model or "claude-3-opus-20240229"
        temperature = 0.2
        max_tokens = 4000
    
    # Call the LLM
    system_prompt = "You are an expert in argument mapping and causal reasoning."
    response = provider.complete(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Extract the ArgDown content (remove any markdown code blocks if present)
    argdown_content = response.content
    if "```" in argdown_content:
        # Extract content between code blocks if present
        import re
        matches = re.findall(r"```(?:argdown)?\n([\s\S]*?)\n```", argdown_content)
        if matches:
            argdown_content = matches[0]
    
    return argdown_content

def validate_argdown(argdown_text: str) -> Dict[str, Any]:
    """
    Validate ArgDown representation to ensure it's well-formed
    
    Args:
        argdown_text: ArgDown representation to validate
    
    Returns:
        Dictionary with validation results
    """
    # Initialize validation results
    results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {
            "node_count": 0,
            "relationship_count": 0,
            "max_depth": 0
        }
    }
    
    # Basic syntax checks
    lines = argdown_text.split("\n")
    node_pattern = r'\[(.*?)\]:'
    instantiation_pattern = r'{"instantiations":'
    
    # Track nodes and relationships
    nodes = set()
    relationships = []
    current_depth = 0
    max_depth = 0
    
    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            continue
        
        # Calculate indentation depth
        indent = 0
        if '+' in line:
            indent = line.find('+') // 2
        
        current_depth = indent
        max_depth = max(max_depth, current_depth)
        
        # Check for node definitions
        import re
        node_matches = re.findall(node_pattern, line)
        if node_matches:
            node = node_matches[0]
            nodes.add(node)
            results["stats"]["node_count"] += 1
            
            # Check for instantiations
            if instantiation_pattern not in line:
                results["warnings"].append(f"Line {i+1}: Node '{node}' is missing instantiations metadata")
        
        # Check parent-child relationships
        if indent > 0 and '+' in line and node_matches:
            # This is a child node; find its parent
            parent_indent = indent - 1
            j = i - 1
            while j >= 0:
                if '+' in lines[j] and lines[j].find('+') // 2 == parent_indent:
                    parent_matches = re.findall(node_pattern, lines[j])
                    if parent_matches:
                        parent = parent_matches[0]
                        relationships.append((parent, node))
                        results["stats"]["relationship_count"] += 1
                        break
                j -= 1
    
    results["stats"]["max_depth"] = max_depth
    
    # If we didn't find any nodes, that's a problem
    if results["stats"]["node_count"] == 0:
        results["is_valid"] = False
        results["errors"].append("No valid nodes found in ArgDown representation")
    
    return results

def process_source_document(file_path: str, provider_name: str = "openai") -> Dict[str, Any]:
    """
    Process a source document to extract ArgDown representation
    
    Args:
        file_path: Path to the source document
        provider_name: The LLM provider to use
    
    Returns:
        Dictionary with extraction results
    """
    # Load the source document
    text = ""
    if file_path.endswith(".pdf"):
        # PDF handling requires additional libraries
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF processing. Install it with: pip install PyPDF2")
    elif file_path.endswith(".txt"):
        with open(file_path, 'r') as file:
            text = file.read()
    elif file_path.endswith(".md"):
        with open(file_path, 'r') as file:
            text = file.read()
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Extract ArgDown
    argdown_content = extract_argdown_from_text(text, provider_name)
    
    # Validate the extraction
    validation_results = validate_argdown(argdown_content)
    
    # Prepare results
    results = {
        "source_path": file_path,
        "extraction_timestamp": time.time(),
        "argdown_content": argdown_content,
        "validation": validation_results,
        "provider": provider_name
    }
    
    return results

def save_argdown_extraction(results: Dict[str, Any], output_path: str) -> None:
    """
    Save ArgDown extraction results
    
    Args:
        results: Extraction results dictionary
        output_path: Path to save the results
    """
    # Save the ArgDown content
    with open(output_path, 'w') as file:
        file.write(results["argdown_content"])
    
    # Save metadata alongside
    metadata_path = output_path.replace('.md', '_metadata.json')
    metadata = {
        "source_path": results["source_path"],
        "extraction_timestamp": results["extraction_timestamp"],
        "validation": results["validation"],
        "provider": results["provider"]
    }
    
    with open(metadata_path, 'w') as file:
        json.dump(metadata, file, indent=2)
```

## 4. Implementation for Cell Sections 1.3-1.5

Now, let's complete the specific code cells required for sections 1.3-1.5 of the notebook:

```python
# 1.3 Prepare LLM API Call
def prepare_extraction_call(source_path, provider_name="openai", model=None):
    """Prepare the LLM API call for ArgDown extraction"""
    
    # Load the source document
    print(f"Processing source document: {source_path}")
    
    # Determine provider and model
    provider = provider_name.lower()
    if provider not in ["openai", "anthropic"]:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'.")
    
    # Set default model if none provided
    if model is None:
        if provider == "openai":
            model = "gpt-4-turbo"
        elif provider == "anthropic":
            model = "claude-3-opus-20240229"
    
    # Print configuration
    print(f"Using provider: {provider}")
    print(f"Selected model: {model}")
    
    return {
        "source_path": source_path,
        "provider": provider,
        "model": model
    }

# Usage example:
source_path = "example_document.pdf"  # Replace with actual document path
extraction_config = prepare_extraction_call(source_path, provider_name="openai")
```

```python
# 1.4 Make ArgDown Extraction LLM API Call
def execute_extraction(extraction_config):
    """Execute the ArgDown extraction using the LLM API"""
    
    print(f"Starting extraction from {extraction_config['source_path']}")
    start_time = time.time()
    
    try:
        # Process the document
        results = process_source_document(
            extraction_config["source_path"],
            provider_name=extraction_config["provider"]
        )
        
        # Print success message
        elapsed_time = time.time() - start_time
        print(f"Extraction completed in {elapsed_time:.2f} seconds")
        print(f"Extracted {results['validation']['stats']['node_count']} nodes with "
              f"{results['validation']['stats']['relationship_count']} relationships")
        
        # Print any warnings
        if results['validation']['warnings']:
            print("\nWarnings:")
            for warning in results['validation']['warnings']:
                print(f"- {warning}")
        
        return results
        
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        raise
        
# Usage example:
extraction_results = execute_extraction(extraction_config)
```

```python
# 1.5 Save ArgDown Extraction Response
def save_extraction_results(results, output_directory="./outputs"):
    """Save the extraction results to file"""
    
    # Ensure output directory exists
    import os
    os.makedirs(output_directory, exist_ok=True)
    
    # Create base filename from source
    import os.path
    base_name = os.path.basename(results["source_path"]).split('.')[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"{base_name}_argdown_{timestamp}.md"
    output_path = os.path.join(output_directory, output_filename)
    
    # Save the results
    save_argdown_extraction(results, output_path)
    
    print(f"Saved ArgDown extraction to: {output_path}")
    print(f"Metadata saved to: {output_path.replace('.md', '_metadata.json')}")
    
    # Also save to standard location for further processing
    standard_path = os.path.join(output_directory, "ArgDown.md")
    with open(standard_path, 'w') as f:
        f.write(results["argdown_content"])
    print(f"Also saved to standard location: {standard_path}")
    
    return output_path

# Usage example:
output_path = save_extraction_results(extraction_results)

# Preview the extracted ArgDown
from IPython.display import Markdown, display

# Display the first 500 characters of the extracted ArgDown
preview = extraction_results["argdown_content"][:500] + "..." if len(extraction_results["argdown_content"]) > 500 else extraction_results["argdown_content"]
display(Markdown(f"## Extracted ArgDown Preview\n\n```\n{preview}\n```"))
```

## 5. Data Post-Processing Implementation (Section 3.3)

Let's also implement the data post-processing functionality mentioned in section 3.3:

```python
def enhance_extracted_data(df):
    """
    Enhance the extracted data with calculated columns
    
    Args:
        df: DataFrame with extracted BayesDown data
    
    Returns:
        Enhanced DataFrame with additional columns
    """
    # Create a copy to avoid modifying the original
    enhanced_df = df.copy()
    
    # 1. Calculate joint probabilities
    enhanced_df['joint_probabilities'] = None
    
    for idx, row in enhanced_df.iterrows():
        title = row['Title']
        priors = row['priors'] if isinstance(row['priors'], dict) else {}
        posteriors = row['posteriors'] if isinstance(row['posteriors'], dict) else {}
        parents = row['Parents'] if isinstance(row['Parents'], list) else []
        
        # Skip if no parents or no priors
        if not parents or not priors:
            continue
        
        # Initialize joint probabilities dictionary
        joint_probs = {}
        
        # Get instantiations
        instantiations = row['instantiations']
        if not isinstance(instantiations, list) or not instantiations:
            continue
            
        # For each parent and child instantiation combination, calculate joint probability
        for inst in instantiations:
            # Get this instantiation's prior probability
            inst_prior_key = f"p({inst})"
            if inst_prior_key not in priors:
                continue
                
            try:
                inst_prior = float(priors[inst_prior_key])
            except (ValueError, TypeError):
                continue
                
            # For each parent
            for parent in parents:
                parent_row = enhanced_df[enhanced_df['Title'] == parent]
                if parent_row.empty:
                    continue
                    
                parent_insts = parent_row.iloc[0]['instantiations']
                if not isinstance(parent_insts, list) or not parent_insts:
                    continue
                    
                for parent_inst in parent_insts:
                    # Get conditional probability
                    cond_key = f"p({inst}|{parent}={parent_inst})"
                    if cond_key in posteriors:
                        try:
                            cond_prob = float(posteriors[cond_key])
                            
                            # Get parent's prior
                            parent_priors = parent_row.iloc[0]['priors']
                            if not isinstance(parent_priors, dict):
                                continue
                                
                            parent_prior_key = f"p({parent_inst})"
                            if parent_prior_key not in parent_priors:
                                continue
                                
                            try:
                                parent_prior = float(parent_priors[parent_prior_key])
                                
                                # Calculate joint probability: P(A,B) = P(A|B) * P(B)
                                joint_prob = cond_prob * parent_prior
                                joint_key = f"p({inst},{parent}={parent_inst})"
                                joint_probs[joint_key] = str(round(joint_prob, 4))
                            except (ValueError, TypeError):
                                continue
                        except (ValueError, TypeError):
                            continue
        
        # Store joint probabilities in dataframe
        enhanced_df.at[idx, 'joint_probabilities'] = joint_probs
    
    # 2. Calculate network metrics
    # Create a directed graph
    import networkx as nx
    G = nx.DiGraph()
    
    # Add nodes
    for idx, row in enhanced_df.iterrows():
        G.add_node(row['Title'])
    
    # Add edges
    for idx, row in enhanced_df.iterrows():
        child = row['Title']
        parents = row['Parents'] if isinstance(row['Parents'], list) else []
        
        for parent in parents:
            if parent in G.nodes():
                G.add_edge(parent, child)
    
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    
    try:
        betweenness_centrality = nx.betweenness_centrality(G)
    except:
        betweenness_centrality = {node: 0 for node in G.nodes()}
    
    # Add metrics to dataframe
    enhanced_df['degree_centrality'] = None
    enhanced_df['in_degree_centrality'] = None
    enhanced_df['out_degree_centrality'] = None
    enhanced_df['betweenness_centrality'] = None
    
    for idx, row in enhanced_df.iterrows():
        title = row['Title']
        enhanced_df.at[idx, 'degree_centrality'] = degree_centrality.get(title, 0)
        enhanced_df.at[idx, 'in_degree_centrality'] = in_degree_centrality.get(title, 0)
        enhanced_df.at[idx, 'out_degree_centrality'] = out_degree_centrality.get(title, 0)
        enhanced_df.at[idx, 'betweenness_centrality'] = betweenness_centrality.get(title, 0)
    
    # 3. Add Markov blanket information (parents, children, and children's parents)
    enhanced_df['markov_blanket'] = None
    
    for idx, row in enhanced_df.iterrows():
        title = row['Title']
        parents = row['Parents'] if isinstance(row['Parents'], list) else []
        children = row['Children'] if isinstance(row['Children'], list) else []
        
        # Get children's parents (excluding this node)
        childrens_parents = []
        for child in children:
            child_row = enhanced_df[enhanced_df['Title'] == child]
            if not child_row.empty:
                child_parents = child_row.iloc[0]['Parents']
                if isinstance(child_parents, list):
                    childrens_parents.extend([p for p in child_parents if p != title])
        
        # Remove duplicates
        childrens_parents = list(set(childrens_parents))
        
        # Combine to get Markov blanket
        markov_blanket = list(set(parents + children + childrens_parents))
        enhanced_df.at[idx, 'markov_blanket'] = markov_blanket
    
    return enhanced_df
```

Now, let's implement the cell for Section 3.3:

```python
# 3.3 Data Post-Processing

# Enhance the extracted dataframe with calculated columns
enhanced_df = enhance_extracted_data(result_df)

# Display the enhanced dataframe
print("Enhanced DataFrame with additional calculated columns:")
enhanced_df.head()

# Check some calculated metrics
print("\nJoint Probabilities Example:")
example_node = enhanced_df.loc[0, 'Title']
joint_probs = enhanced_df.loc[0, 'joint_probabilities']
print(f"Joint probabilities for {example_node}:")
print(joint_probs)

print("\nNetwork Metrics:")
for idx, row in enhanced_df.iterrows():
    print(f"{row['Title']}:")
    print(f"  Degree Centrality: {row['degree_centrality']:.3f}")
    print(f"  Betweenness Centrality: {row['betweenness_centrality']:.3f}")

# Save the enhanced dataframe
enhanced_df.to_csv('enhanced_extracted_data.csv', index=False)
print("\nEnhanced data saved to 'enhanced_extracted_data.csv'")
```

This implementation covers the key missing components of the AMTAIR Colab notebook, focusing on LLM integration for extraction and data post-processing. The code follows the existing patterns in the notebook while adding more robust error handling, documentation, and additional functionality.
</implementation>
