# Enhancing the AMTAIR Prototype Implementation

After thoroughly analyzing the AMTAIR prototype notebook and associated documentation, I've identified several sections that could benefit from implementation or enhancement. The notebook represents a crucial technical component of the larger AMTAIR project, which aims to create a system for automatically extracting world models from AI safety literature to improve coordination in AI governance.

<clarifying_questions>
1. **API Access Requirements**: What API keys will be available for LLM integration? (Assumption: OpenAI and/or Anthropic APIs will be accessible with appropriate keys)

2. **PDF Processing Capabilities**: What level of PDF processing complexity should be supported? (Assumption: Standard academic papers with relatively consistent formatting)

3. **Storage Constraints**: Are there any storage limitations for processed documents and models? (Assumption: Standard Colab storage limits apply)

4. **Performance Requirements**: What are the performance expectations for processing large documents? (Assumption: Reasonable processing times for academic papers, with optimization where possible)

5. **Error Handling Requirements**: What level of robustness is required for error handling? (Assumption: Comprehensive error handling with informative messages)
</clarifying_questions>

<implementation_plans>
## Plan A: Complete End-to-End Pipeline with OpenAI Integration

This plan focuses on implementing a comprehensive pipeline using OpenAI's GPT models for extraction, with robust PDF processing and data handling.

**Key Components:**
1. **PDF Ingestion Module**:
   - Implement PyPDF2 and pdfplumber for text extraction
   - Add document chunking for handling large PDFs
   - Include metadata extraction (title, authors, date)

2. **LLM Integration for ArgDown Extraction**:
   - OpenAI API integration with optimized prompting
   - Implement batched processing for larger documents
   - Add context handling for maintaining coherence across chunks

3. **BayesDown Generation Pipeline**:
   - Implement second-stage LLM processing for probability extraction
   - Add validation mechanisms for probability coherence
   - Include uncertainty quantification for extracted probabilities

4. **Enhanced Data Post-Processing**:
   - Add calculation of derived probabilities (joint, marginal)
   - Implement sensitivity analysis functions
   - Add network metrics calculation

**Advantages:**
- Comprehensive implementation with industry-standard tools
- High-quality extraction using state-of-the-art models
- Complete pipeline with robust error handling

**Disadvantages:**
- API costs could be significant for large documents
- Dependency on external APIs may create reliability issues
- Potentially slower processing due to API latency

## Plan B: Hybrid Approach with Local Models and Cloud APIs

This plan balances local processing with selective use of cloud APIs, optimizing for cost and reliability.

**Key Components:**
1. **PDF Processing with Local Tools**:
   - Implement PDF processing using pure Python libraries
   - Add local OCR capabilities for handling scanned documents
   - Include local caching of processed documents

2. **Two-Tier LLM Processing**:
   - Use simpler local models for initial text segmentation
   - Selectively use cloud APIs for critical extraction tasks
   - Implement robust caching to minimize repeat API calls

3. **Enhanced BayesDown Processing**:
   - Implement rule-based probability extraction where possible
   - Use pattern matching for common probability expressions
   - Fallback to LLM for complex extractions

4. **Optimized Data Post-Processing**:
   - Implement incremental data processing
   - Add parallel processing for computationally intensive tasks
   - Include data versioning for tracking changes

**Advantages:**
- Reduced API dependency and costs
- Better reliability with local fallbacks
- Potentially faster processing for simpler documents

**Disadvantages:**
- Potentially lower extraction quality for complex arguments
- More complex codebase to maintain
- Requires balancing multiple processing approaches

## Plan C: Modular Implementation with Progressive Enhancement

This plan emphasizes a modular architecture with clearly defined interfaces, allowing for progressive enhancement and easier maintenance.

**Key Components:**
1. **Modular Document Processing**:
   - Abstract document source interface (PDFs, HTML, plain text)
   - Implement adapters for different document types
   - Add document segmentation based on content structure

2. **Pluggable LLM Backend**:
   - Define generic LLM interface with pluggable backends
   - Implement adapters for OpenAI, Anthropic, and local models
   - Add model selection based on task requirements

3. **Structured BayesDown Generation**:
   - Implement multi-stage extraction pipeline
   - Add formal validation against BayesDown schema
   - Include confidence scoring for extracted elements

4. **Extensible Post-Processing Framework**:
   - Define plugin architecture for data transformations
   - Implement core transformation set
   - Add visualization enhancement modules

**Advantages:**
- Highly maintainable and extensible architecture
- Easier to adapt to new LLM providers or capabilities
- Clear separation of concerns with well-defined interfaces

**Disadvantages:**
- More upfront design and implementation effort
- Potentially more complex for simple use cases
- May require more boilerplate code
</implementation_plans>

<evaluation>
# Evaluation of Implementation Plans

I'll evaluate each plan against our criteria:

## Plan A: Complete End-to-End Pipeline with OpenAI Integration

- **Elegance**: 7/10 - Straightforward but potentially monolithic
- **Future-proofness**: 6/10 - Tightly coupled to specific APIs
- **Iteration speed**: 8/10 - Quick to implement and test core functionality
- **Cost/effort**: 7/10 - Moderate implementation effort, higher API costs
- **Performance**: 8/10 - High-quality results but API latency concerns
- **Compatibility**: 8/10 - Aligns well with existing notebook structure

**Overall**: 44/60 - Solid approach with good quality but some concerns about flexibility and costs

## Plan B: Hybrid Approach with Local Models and Cloud APIs

- **Elegance**: 5/10 - Multiple processing paths create complexity
- **Future-proofness**: 7/10 - Less dependent on specific APIs
- **Iteration speed**: 6/10 - More complex testing scenarios
- **Cost/effort**: 6/10 - Higher implementation complexity, lower API costs
- **Performance**: 7/10 - Variable quality but potentially faster
- **Compatibility**: 6/10 - Requires more adaptation to existing structure

**Overall**: 37/60 - Practical but introduces complexity that may not be justified

## Plan C: Modular Implementation with Progressive Enhancement

- **Elegance**: 9/10 - Clean architecture with clear separation of concerns
- **Future-proofness**: 9/10 - Easily adaptable to new requirements
- **Iteration speed**: 7/10 - Initial setup takes longer, but faster iterations afterward
- **Cost/effort**: 7/10 - Higher upfront design cost, better maintainability
- **Performance**: 8/10 - Optimizable for different scenarios
- **Compatibility**: 8/10 - Can be implemented incrementally within existing structure

**Overall**: 48/60 - Superior long-term approach with better architectural qualities

**Selected Plan**: Plan C - Modular Implementation with Progressive Enhancement

**Justification**: While Plan A offers the quickest path to a working implementation, Plan C provides a more robust and maintainable foundation that aligns with the academic nature of the AMTAIR project. The modular architecture will support ongoing research and adaptation to new techniques while maintaining clean interfaces between components. This approach also enables incremental implementation, allowing us to improve the notebook progressively rather than requiring a complete overhaul. The initial investment in architectural design will pay dividends through reduced maintenance costs and greater adaptability to emerging requirements.
</evaluation>

<optimal_plan>
# Detailed Implementation Plan: Modular Architecture with Progressive Enhancement

## Phase 1: Core Architecture Setup (Foundation)

### 1.1 Document Processing Module
- Define `DocumentSource` abstract interface with methods:
  - `load_document(source_path)`: Loads document from path
  - `extract_text()`: Extracts plain text
  - `extract_metadata()`: Extracts document metadata
  - `segment_document()`: Divides document into logical segments

- Implement concrete adapters:
  - `PDFDocumentSource`: For PDF documents
  - `TextDocumentSource`: For plain text
  - `HTMLDocumentSource`: For HTML documents (future extension)

- Create `DocumentProcessor` class to handle:
  - Document loading
  - Pre-processing (cleaning, normalization)
  - Chunking strategies (fixed-size, semantic, structure-based)
  - Content extraction

### 1.2 LLM Service Layer
- Define `LLMProvider` interface with methods:
  - `generate_completion(prompt, params)`: Basic completion
  - `generate_structured_output(prompt, schema, params)`: Schema-guided generation
  - `batch_process(prompts, params)`: Batch processing for efficiency

- Implement concrete providers:
  - `OpenAIProvider`: For OpenAI models
  - `AnthropicProvider`: For Anthropic models
  - `MockProvider`: For testing without API calls

- Create `PromptTemplate` system for standardized prompts:
  - Template definition with variable substitution
  - Context management for multi-turn interactions
  - Prompt versioning and optimization

### 1.3 Extraction Pipeline Framework
- Define pipeline stages as composable components:
  - `DocumentIngestion`: Loads and processes documents
  - `ArgDownExtraction`: Extracts argument structure
  - `BayesDownEnhancement`: Adds probabilistic information
  - `DataTransformation`: Converts to structured formats
  - `ModelConstruction`: Builds Bayesian networks

- Implement pipeline coordinator:
  - Stage dependency management
  - Intermediate result handling
  - Error recovery and resumption
  - Progress tracking and reporting

## Phase 2: Implementation of Core Components

### 2.1 ArgDown Extraction Implementation
- Enhance existing parser with:
  - Robust error handling
  - Better validation of extraction results
  - Support for hierarchical argument structures
  - Handling of cross-references

- Implement LLM-powered extraction:
  - Document to ArgDown prompt engineering
  - Context handling for document chunks
  - Response validation and reformatting
  - Incremental extraction with state management

### 2.2 BayesDown Enhancement Implementation
- Extend question generation with:
  - Smarter question prioritization based on network structure
  - Dynamic question generation based on extraction confidence
  - Support for multi-state variables beyond binary

- Implement probability extraction:
  - LLM-based probability estimation
  - Consistency enforcement (probabilities sum to 1)
  - Confidence scoring for extracted probabilities
  - Uncertainty representation in the model

### 2.3 Data Processing Enhancements
- Add derived probability calculations:
  - Joint probability computation
  - Marginal probability calculation
  - Information value metrics
  - Sensitivity analysis functions

- Implement validation and enrichment:
  - Graph structure validation
  - Probabilistic coherence checking
  - Metadata enrichment
  - Cross-model comparison utilities

## Phase 3: Integration and UI Enhancements

### 3.1 Notebook Integration
- Refactor existing notebook sections to use new architecture
- Add configuration cells for customization
- Enhance progress reporting and visualization
- Improve error handling and diagnostics

### 3.2 Visualization Enhancements
- Add model comparison views
- Implement sensitivity analysis visualization
- Add interactive what-if analysis
- Enhance probability visualization with uncertainty

### 3.3 Documentation and Examples
- Add detailed documentation for each module
- Create example workflows for common use cases
- Document extension points and customization options
- Add diagnostic and debugging tools

## Implementation Priorities

For the initial implementation, we'll focus on:

1. Core architecture components (interfaces and basic implementations)
2. PDF processing and LLM integration for ArgDown extraction
3. Basic BayesDown enhancement with probability extraction
4. Essential data processing for Bayesian networks
5. Integration with the existing notebook structure

This approach allows for incremental improvement while establishing the foundation for a robust and extensible system.
</optimal_plan>

<implementation>
# Initial Implementation

Let's implement the key components of our plan, focusing on the core architecture and PDF processing capabilities, followed by the LLM integration for ArgDown extraction.

## 1. Core Architecture Setup

### 1.1 Document Source Interface and Adapters

```python
import abc
from typing import Dict, List, Optional, Union, Any
import os
import tempfile
from dataclasses import dataclass

@dataclass
class DocumentMetadata:
    """Structured container for document metadata."""
    title: Optional[str] = None
    authors: List[str] = None
    date: Optional[str] = None
    source_type: Optional[str] = None
    source_path: Optional[str] = None
    num_pages: Optional[int] = None
    additional_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.additional_metadata is None:
            self.additional_metadata = {}

@dataclass
class DocumentSegment:
    """A segment (chunk) of a document with its content and metadata."""
    content: str
    segment_id: str
    page_numbers: List[int] = None
    section_title: Optional[str] = None
    segment_type: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.page_numbers is None:
            self.page_numbers = []
        if self.metadata is None:
            self.metadata = {}

class DocumentSource(abc.ABC):
    """Abstract interface for document sources."""
    
    @abc.abstractmethod
    def load_document(self, source_path: str) -> bool:
        """
        Load a document from the specified path.
        
        Args:
            source_path: Path to the document file
            
        Returns:
            True if the document was loaded successfully, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def extract_text(self) -> str:
        """
        Extract plain text from the loaded document.
        
        Returns:
            The extracted text as a string
        """
        pass
    
    @abc.abstractmethod
    def extract_metadata(self) -> DocumentMetadata:
        """
        Extract metadata from the loaded document.
        
        Returns:
            DocumentMetadata object with extracted information
        """
        pass
    
    @abc.abstractmethod
    def segment_document(self, 
                        strategy: str = 'auto', 
                        max_segment_length: int = 4000) -> List[DocumentSegment]:
        """
        Segment the document into logical chunks.
        
        Args:
            strategy: Segmentation strategy ('auto', 'fixed_size', 'semantic', 'structural')
            max_segment_length: Maximum length of each segment in characters
            
        Returns:
            List of DocumentSegment objects
        """
        pass
```

### 1.2 PDF Document Adapter Implementation

```python
import PyPDF2
import pdfplumber
import re
from pdf2image import convert_from_path
import pytesseract
from typing import List, Dict, Optional, Any
import hashlib

class PDFDocumentSource(DocumentSource):
    """Implementation of DocumentSource for PDF files."""
    
    def __init__(self):
        self.pdf_path = None
        self.pdf_reader = None
        self.pdf_text = None
        self.pdf_metadata = None
        self.ocr_enabled = False
        self.ocr_language = 'eng'
        self.use_pdfplumber = True
    
    def load_document(self, source_path: str, ocr_enabled: bool = False, 
                     ocr_language: str = 'eng', use_pdfplumber: bool = True) -> bool:
        """
        Load a PDF document from the specified path.
        
        Args:
            source_path: Path to the PDF file
            ocr_enabled: Whether to use OCR for text extraction from images
            ocr_language: Language for OCR processing
            use_pdfplumber: Whether to use pdfplumber for more accurate text extraction
            
        Returns:
            True if the PDF was loaded successfully, False otherwise
        """
        try:
            self.pdf_path = source_path
            self.ocr_enabled = ocr_enabled
            self.ocr_language = ocr_language
            self.use_pdfplumber = use_pdfplumber
            
            # Open with PyPDF2 for basic operations
            with open(source_path, 'rb') as file:
                self.pdf_reader = PyPDF2.PdfReader(file)
                # Success if we can access pages
                return len(self.pdf_reader.pages) > 0
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return False
    
    def extract_text(self) -> str:
        """
        Extract text from the PDF using multiple approaches for best results.
        
        Returns:
            Extracted text as a string
        """
        if not self.pdf_reader:
            raise ValueError("No PDF document loaded")
        
        if self.pdf_text is not None:
            return self.pdf_text
        
        text_content = []
        
        # Try pdfplumber first if enabled (better layout preservation)
        if self.use_pdfplumber:
            try:
                with pdfplumber.open(self.pdf_path) as pdf:
                    for page in pdf.pages:
                        text_content.append(page.extract_text() or "")
            except Exception as e:
                print(f"Error with pdfplumber extraction: {e}")
                # Fall back to PyPDF2
                text_content = []
                for page in self.pdf_reader.pages:
                    text_content.append(page.extract_text() or "")
        else:
            # Use PyPDF2 directly
            for page in self.pdf_reader.pages:
                text_content.append(page.extract_text() or "")
        
        # If OCR is enabled and we have pages with little or no text, use OCR
        if self.ocr_enabled:
            # Check if any pages have very little text
            page_lengths = [len(page_text) for page_text in text_content]
            total_length = sum(page_lengths)
            avg_length = total_length / len(page_lengths) if page_lengths else 0
            
            # If average text per page is very low, try OCR
            if avg_length < 100:  # Arbitrary threshold
                try:
                    images = convert_from_path(self.pdf_path)
                    for i, image in enumerate(images):
                        if i < len(text_content) and len(text_content[i]) < 100:
                            ocr_text = pytesseract.image_to_string(image, lang=self.ocr_language)
                            text_content[i] = ocr_text
                except Exception as e:
                    print(f"Error with OCR processing: {e}")
        
        # Combine all pages
        self.pdf_text = "\n\n".join(text_content)
        return self.pdf_text
    
    def extract_metadata(self) -> DocumentMetadata:
        """
        Extract metadata from the PDF document.
        
        Returns:
            DocumentMetadata object with extracted information
        """
        if not self.pdf_reader:
            raise ValueError("No PDF document loaded")
        
        if self.pdf_metadata is not None:
            return self.pdf_metadata
        
        # Extract metadata from PyPDF2
        pdf_info = self.pdf_reader.metadata
        
        # Create basic metadata
        metadata = DocumentMetadata(
            source_path=self.pdf_path,
            source_type="PDF",
            num_pages=len(self.pdf_reader.pages)
        )
        
        # Try to extract standard metadata fields
        if pdf_info:
            if '/Title' in pdf_info:
                metadata.title = pdf_info.get('/Title', '').strip()
            
            # Author could be a single string or a list
            author_info = pdf_info.get('/Author', '')
            if author_info:
                if isinstance(author_info, str):
                    # Try to split by common separators
                    authors = re.split(r',|;|and', author_info)
                    metadata.authors = [author.strip() for author in authors if author.strip()]
                elif isinstance(author_info, list):
                    metadata.authors = [author.strip() for author in author_info if author.strip()]
            
            # Extract date information
            date_info = pdf_info.get('/CreationDate', '') or pdf_info.get('/ModDate', '')
            if date_info and isinstance(date_info, str):
                # Try to parse PDF date format (e.g., D:20210930120000+02'00')
                date_match = re.search(r'D:(\d{4})(\d{2})(\d{2})', date_info)
                if date_match:
                    year, month, day = date_match.groups()
                    metadata.date = f"{year}-{month}-{day}"
            
            # Add all other metadata fields to additional_metadata
            for key, value in pdf_info.items():
                if key not in ['/Title', '/Author', '/CreationDate', '/ModDate']:
                    metadata.additional_metadata[key] = value
        
        # If title is not in metadata, try to extract from first page
        if not metadata.title:
            try:
                first_page_text = self.pdf_reader.pages[0].extract_text()
                # Simple heuristic: first line could be title
                if first_page_text:
                    lines = first_page_text.split('\n')
                    if lines:
                        metadata.title = lines[0].strip()
            except:
                pass
        
        self.pdf_metadata = metadata
        return metadata
    
    def segment_document(self, 
                        strategy: str = 'auto', 
                        max_segment_length: int = 4000) -> List[DocumentSegment]:
        """
        Segment the PDF document into logical chunks.
        
        Args:
            strategy: Segmentation strategy ('auto', 'fixed_size', 'semantic', 'structural')
            max_segment_length: Maximum length of each segment in characters
            
        Returns:
            List of DocumentSegment objects
        """
        if not self.pdf_reader:
            raise ValueError("No PDF document loaded")
        
        text = self.extract_text()
        segments = []
        
        # Get basic metadata for segments
        metadata = self.extract_metadata()
        
        if strategy == 'fixed_size':
            # Simple fixed-size chunking
            chunk_size = max_segment_length
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                segment_id = f"segment_{i+1}"
                segments.append(DocumentSegment(
                    content=chunk,
                    segment_id=segment_id,
                    segment_type="fixed_size_chunk",
                    metadata={"source": metadata.source_path, "chunk_index": i}
                ))
                
        elif strategy == 'structural' or strategy == 'auto':
            # Try to segment by sections (basic heuristic)
            # Look for common section markers like "Introduction", "Methods", etc.
            section_patterns = [
                r'^\s*(\d+\..*?)\n',  # Numbered sections (e.g., "1. Introduction")
                r'^\s*((?:Introduction|Abstract|Background|Methods|Results|Discussion|Conclusion|References).*?)\n',  # Common section names
                r'^\s*([A-Z][A-Z\s]+)\n',  # All-caps section titles
                r'^\s*(.*?)\n[-=]{3,}'  # Headers with underlines
            ]
            
            # Extract potential section boundaries
            potential_sections = []
            for pattern in section_patterns:
                for match in re.finditer(pattern, text, re.MULTILINE):
                    title = match.group(1).strip()
                    start_idx = match.start()
                    potential_sections.append((start_idx, title))
            
            # Sort by position in text
            potential_sections.sort()
            
            # Create segments
            if potential_sections:
                # Add beginning of document if first section doesn't start at 0
                if potential_sections[0][0] > 0:
                    potential_sections.insert(0, (0, "Document Start"))
                
                # Create segments between section boundaries
                for i in range(len(potential_sections)):
                    start_idx = potential_sections[i][0]
                    title = potential_sections[i][1]
                    
                    # End is either next section start or end of text
                    end_idx = potential_sections[i+1][0] if i < len(potential_sections) - 1 else len(text)
                    
                    segment_content = text[start_idx:end_idx].strip()
                    
                    # Skip empty segments
                    if not segment_content:
                        continue
                    
                    # Generate a stable segment ID based on content
                    content_hash = hashlib.md5(segment_content.encode()).hexdigest()[:8]
                    segment_id = f"section_{i+1}_{content_hash}"
                    
                    segments.append(DocumentSegment(
                        content=segment_content,
                        segment_id=segment_id,
                        section_title=title,
                        segment_type="structural_section",
                        metadata={"source": metadata.source_path, "section_index": i}
                    ))
            else:
                # Fallback to fixed-size chunking if no sections found
                return self.segment_document(strategy='fixed_size', max_segment_length=max_segment_length)
        
        elif strategy == 'semantic':
            # For a simple implementation, we'll approximate semantic chunking 
            # by splitting on paragraph boundaries with a maximum size
            paragraphs = re.split(r'\n\s*\n', text)
            current_chunk = ""
            current_chunk_index = 0
            
            for i, para in enumerate(paragraphs):
                # If adding this paragraph would exceed the max length and we already have content,
                # create a new segment
                if current_chunk and len(current_chunk) + len(para) > max_segment_length:
                    # Generate a stable segment ID based on content
                    content_hash = hashlib.md5(current_chunk.encode()).hexdigest()[:8]
                    segment_id = f"para_chunk_{current_chunk_index}_{content_hash}"
                    
                    segments.append(DocumentSegment(
                        content=current_chunk,
                        segment_id=segment_id,
                        segment_type="semantic_paragraph_group",
                        metadata={"source": metadata.source_path, "chunk_index": current_chunk_index}
                    ))
                    
                    current_chunk = para
                    current_chunk_index += 1
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
            
            # Add the last chunk if it has content
            if current_chunk:
                content_hash = hashlib.md5(current_chunk.encode()).hexdigest()[:8]
                segment_id = f"para_chunk_{current_chunk_index}_{content_hash}"
                
                segments.append(DocumentSegment(
                    content=current_chunk,
                    segment_id=segment_id,
                    segment_type="semantic_paragraph_group",
                    metadata={"source": metadata.source_path, "chunk_index": current_chunk_index}
                ))
        
        else:
            raise ValueError(f"Unknown segmentation strategy: {strategy}")
        
        return segments
```

### 1.3 LLM Provider Interface and Implementation

```python
import abc
import json
import time
import openai
import os
from typing import Dict, List, Optional, Union, Any, TypeVar
from dataclasses import dataclass, field, asdict

T = TypeVar('T')

@dataclass
class LLMResponse:
    """Structured response from an LLM provider."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMRequest:
    """Structured request to an LLM provider."""
    prompt: str
    model: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    system_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class LLMProvider(abc.ABC):
    """Abstract interface for LLM providers."""
    
    @abc.abstractmethod
    def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a completion using the specified request parameters.
        
        Args:
            request: LLMRequest object with request parameters
            
        Returns:
            LLMResponse object with the generated content
        """
        pass
    
    @abc.abstractmethod
    def generate_structured_output(self, 
                                  request: LLMRequest, 
                                  output_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured output according to the specified schema.
        
        Args:
            request: LLMRequest object with request parameters
            output_schema: JSON schema defining the expected output structure
            
        Returns:
            Dictionary conforming to the output schema
        """
        pass
    
    def batch_process(self, 
                     requests: List[LLMRequest],
                     max_retries: int = 3,
                     retry_delay: float = 1.0) -> List[LLMResponse]:
        """
        Process a batch of requests with efficient handling and retries.
        
        Default implementation processes requests sequentially with retry logic.
        Providers can override this for more efficient batching.
        
        Args:
            requests: List of LLMRequest objects
            max_retries: Maximum number of retries for failed requests
            retry_delay: Base delay between retries (exponential backoff)
            
        Returns:
            List of LLMResponse objects
        """
        responses = []
        
        for request in requests:
            retries = 0
            while retries <= max_retries:
                try:
                    response = self.generate_completion(request)
                    responses.append(response)
                    break
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        # Create an error response
                        error_response = LLMResponse(
                            content=f"Error: {str(e)}",
                            model=request.model,
                            metadata={"error": str(e), "request": asdict(request)}
                        )
                        responses.append(error_response)
                    else:
                        # Exponential backoff
                        time.sleep(retry_delay * (2 ** (retries - 1)))
        
        return responses

class OpenAIProvider(LLMProvider):
    """Implementation of LLMProvider for OpenAI's API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key (optional, defaults to environment variable)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it directly or via OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
    
    def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a completion using OpenAI's API.
        
        Args:
            request: LLMRequest object with request parameters
            
        Returns:
            LLMResponse object with the generated content
        """
        messages = []
        
        # Add system message if provided
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        
        # Add user message with the prompt
        messages.append({"role": "user", "content": request.prompt})
        
        # Prepare API request parameters
        params = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
        }
        
        # Add optional parameters if provided
        if request.max_tokens:
            params["max_tokens"] = request.max_tokens
        
        if request.stop_sequences:
            params["stop"] = request.stop_sequences
        
        # Make API call
        try:
            response = openai.ChatCompletion.create(**params)
            
            # Extract content from response
            content = response.choices[0].message.content
            
            # Create structured response
            return LLMResponse(
                content=content,
                model=request.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                raw_response=response,
                metadata={"request": asdict(request)}
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")
    
    def generate_structured_output(self, 
                                  request: LLMRequest, 
                                  output_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured output according to the specified schema using OpenAI's API.
        
        Args:
            request: LLMRequest object with request parameters
            output_schema: JSON schema defining the expected output structure
            
        Returns:
            Dictionary conforming to the output schema
        """
        # Modify the request to include instructions for structured output
        schema_str = json.dumps(output_schema, indent=2)
        structured_prompt = f"""
        {request.prompt}
        
        Please provide your response as a valid JSON object matching the following schema:
        
        {schema_str}
        
        Ensure the output is valid JSON with no additional text before or after.
        """
        
        # Create a new request with the modified prompt
        structured_request = LLMRequest(
            prompt=structured_prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop_sequences=request.stop_sequences,
            system_message=request.system_message or "You are a helpful assistant that provides structured JSON responses.",
            metadata=request.metadata
        )
        
        # Generate the response
        response = self.generate_completion(structured_request)
        
        # Parse the JSON response
        try:
            # Extract the JSON part (handling possible text before/after)
            content = response.content.strip()
            
            # Find the first { and last }
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                raise ValueError("Response does not contain valid JSON")
            
            json_str = content[start_idx:end_idx+1]
            result = json.loads(json_str)
            
            return result
        except Exception as e:
            raise ValueError(f"Failed to parse structured output: {str(e)}\nResponse content: {response.content}")
```

### 1.4 ArgDown Extraction Pipeline

```python
from typing import Dict, List, Optional, Union, Any, Tuple
import re
import json
from dataclasses import dataclass, field

@dataclass
class ArgDownNode:
    """Representation of a node in an ArgDown argument structure."""
    title: str
    description: str
    instantiations: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    indentation: int = 0
    line_number: Optional[int] = None

@dataclass
class ArgDownGraph:
    """Complete representation of an ArgDown argument graph."""
    nodes: Dict[str, ArgDownNode]
    source_document: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_root_nodes(self) -> List[str]:
        """Get the titles of all root nodes (nodes without parents)."""
        return [title for title, node in self.nodes.items() if not node.parents]
    
    def get_leaf_nodes(self) -> List[str]:
        """Get the titles of all leaf nodes (nodes without children)."""
        return [title for title, node in self.nodes.items() if not node.children]
    
    def to_markdown(self) -> str:
        """Convert the graph to ArgDown markdown syntax."""
        # Start with root nodes and traverse depth-first
        roots = self.get_root_nodes()
        lines = []
        
        def process_node(node_title, indent=0):
            node = self.nodes[node_title]
            prefix = " " * indent + ("+ " if indent > 0 else "")
            
            # Format instantiations as JSON if present
            inst_json = ""
            if node.instantiations:
                inst_dict = {"instantiations": node.instantiations}
                inst_json = json.dumps(inst_dict)
            
            # Create the node line
            lines.append(f"{prefix}[{node_title}]: {node.description} {inst_json}")
            
            # Process children
            for child_title in node.children:
                process_node(child_title, indent + 2)
        
        # Process each root node
        for root in roots:
            process_node(root)
        
        return "\n".join(lines)

class ArgDownExtractor:
    """Class for extracting ArgDown structures from text."""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the ArgDown extractor.
        
        Args:
            llm_provider: LLM provider for extraction
        """
        self.llm_provider = llm_provider
        
        # Default system message for ArgDown extraction
        self.system_message = """
        You are an expert system for extracting argument structures from text.
        Your task is to analyze text and identify the key arguments, claims, and their relationships.
        You will output these in ArgDown format, which is a markdown-like syntax for argument mapping.
        
        Follow these rules:
        1. Identify the main claims or conclusions in the text
        2. Identify the premises or supporting evidence for each claim
        3. Map the relationships between claims and premises
        4. Use proper ArgDown syntax with [node_name]: description format
        5. Use indentation and + symbols to indicate relationships
        6. Include possible instantiations for each node in JSON metadata
        
        Your output should be valid ArgDown that can be parsed by automated systems.
        """
    
    def extract_argdown_from_text(self, 
                                 text: str, 
                                 model: str = "gpt-4",
                                 temperature: float = 0.0,
                                 max_tokens: int = 2000) -> ArgDownGraph:
        """
        Extract an ArgDown graph from the provided text.
        
        Args:
            text: Source text to extract arguments from
            model: LLM model to use for extraction
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens for the response
            
        Returns:
            ArgDownGraph object containing the extracted structure
        """
        # Create extraction prompt
        prompt = f"""
        Please analyze the following text and extract its argument structure in ArgDown format.
        
        For each important claim or statement:
        1. Give it a concise, descriptive title in square brackets
        2. Provide a brief description of the statement
        3. Add JSON metadata with possible instantiations (usually TRUE/FALSE)
        4. Use proper indentation and + symbols to show relationships
        
        Here's the text to analyze:
        
        {text}
        
        Please provide only the ArgDown structure with no additional explanation.
        """
        
        # Create LLM request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=self.system_message
        )
        
        # Generate ArgDown using LLM
        response = self.llm_provider.generate_completion(request)
        argdown_text = response.content.strip()
        
        # Parse the generated ArgDown
        return self.parse_argdown_text(argdown_text)
    
    def extract_argdown_from_document(self, 
                                     document_source: DocumentSource,
                                     model: str = "gpt-4",
                                     temperature: float = 0.0,
                                     segment_strategy: str = "auto") -> ArgDownGraph:
        """
        Extract an ArgDown graph from the provided document.
        
        Args:
            document_source: DocumentSource object with loaded document
            model: LLM model to use for extraction
            temperature: Sampling temperature for generation
            segment_strategy: Strategy for document segmentation
            
        Returns:
            ArgDownGraph object containing the extracted structure
        """
        # Segment the document
        segments = document_source.segment_document(strategy=segment_strategy)
        
        # Extract metadata
        metadata = document_source.extract_metadata()
        
        # Process each segment
        segment_graphs = []
        for segment in segments:
            try:
                segment_graph = self.extract_argdown_from_text(
                    text=segment.content,
                    model=model,
                    temperature=temperature
                )
                segment_graphs.append(segment_graph)
            except Exception as e:
                print(f"Error processing segment {segment.segment_id}: {e}")
        
        # Merge segment graphs
        merged_graph = self._merge_argdown_graphs(segment_graphs)
        
        # Add document metadata
        merged_graph.source_document = metadata.source_path
        merged_graph.metadata.update({
            "title": metadata.title,
            "authors": metadata.authors,
            "date": metadata.date,
            "num_segments": len(segments)
        })
        
        return merged_graph
    
    def parse_argdown_text(self, argdown_text: str) -> ArgDownGraph:
        """
        Parse ArgDown text into a structured graph.
        
        Args:
            argdown_text: Text in ArgDown format
            
        Returns:
            ArgDownGraph object representing the argument structure
        """
        lines = argdown_text.strip().split('\n')
        nodes = {}
        
        # First pass: identify nodes and their descriptions
        for i, line in enumerate(lines):
            line = line.rstrip()
            if not line or line.startswith('/*') or line.endswith('*/'):
                continue
                
            # Count indentation
            indentation = len(line) - len(line.lstrip())
            
            # Check for node definition pattern
            node_match = re.search(r'[+\s]*\[(.+?)\]:\s*(.*?)(\{.*\})?$', line)
            if node_match:
                title = node_match.group(1).strip()
                description = node_match.group(2).strip()
                metadata_str = node_match.group(3) or '{}'
                
                # Try to parse JSON metadata
                try:
                    metadata = json.loads(metadata_str)
                except:
                    metadata = {}
                
                # Extract instantiations from metadata
                instantiations = metadata.get('instantiations', [f"{title}_TRUE", f"{title}_FALSE"])
                
                # Create node
                nodes[title] = ArgDownNode(
                    title=title,
                    description=description,
                    instantiations=instantiations,
                    metadata=metadata,
                    indentation=indentation,
                    line_number=i
                )
        
        # Second pass: establish relationships
        for i, line in enumerate(lines):
            line = line.rstrip()
            if not line or line.startswith('/*') or line.endswith('*/'):
                continue
                
            # Extract node title from line
            node_match = re.search(r'[+\s]*\[(.+?)\]', line)
            if not node_match:
                continue
                
            title = node_match.group(1).strip()
            if title not in nodes:
                continue
                
            # Current node's indentation
            indent = nodes[title].indentation
            
            # Look for parent-child relationships
            for j, other_line in enumerate(lines):
                if i == j:
                    continue
                    
                other_match = re.search(r'[+\s]*\[(.+?)\]', other_line)
                if not other_match:
                    continue
                    
                other_title = other_match.group(1).strip()
                if other_title not in nodes:
                    continue
                    
                other_indent = nodes[other_title].indentation
                
                # Check if this could be a parent (previous line, less indentation)
                if j < i and other_indent < indent and '+' in line:
                    # Parent relationship found
                    nodes[title].parents.append(other_title)
                    nodes[other_title].children.append(title)
                
                # Check if this could be a child (next line, more indentation)
                elif j > i and other_indent > indent and '+' in other_line:
                    # Check if there's no line in between with less indentation
                    is_direct_child = True
                    for k in range(i+1, j):
                        between_line = lines[k].rstrip()
                        if between_line and not between_line.startswith('/*') and not between_line.endswith('*/'):
                            between_indent = len(between_line) - len(between_line.lstrip())
                            if between_indent <= indent:
                                is_direct_child = False
                                break
                    
                    if is_direct_child:
                        # Child relationship found
                        nodes[title].children.append(other_title)
                        nodes[other_title].parents.append(title)
        
        # Create and return the graph
        return ArgDownGraph(nodes=nodes)
    
    def _merge_argdown_graphs(self, graphs: List[ArgDownGraph]) -> ArgDownGraph:
        """
        Merge multiple ArgDown graphs into a single coherent graph.
        
        Args:
            graphs: List of ArgDownGraph objects to merge
            
        Returns:
            Merged ArgDownGraph
        """
        if not graphs:
            return ArgDownGraph(nodes={})
        
        if len(graphs) == 1:
            return graphs[0]
        
        # Start with the first graph
        merged_nodes = dict(graphs[0].nodes)
        
        # Merge in each additional graph
        for graph in graphs[1:]:
            for title, node in graph.nodes.items():
                if title in merged_nodes:
                    # Node already exists, merge properties
                    existing = merged_nodes[title]
                    
                    # Keep the longer description
                    if len(node.description) > len(existing.description):
                        existing.description = node.description
                    
                    # Merge instantiations
                    for inst in node.instantiations:
                        if inst not in existing.instantiations:
                            existing.instantiations.append(inst)
                    
                    # Merge parents and children
                    for parent in node.parents:
                        if parent not in existing.parents:
                            existing.parents.append(parent)
                    
                    for child in node.children:
                        if child not in existing.children:
                            existing.children.append(child)
                    
                    # Merge metadata
                    existing.metadata.update(node.metadata)
                else:
                    # New node, add to merged graph
                    merged_nodes[title] = node
        
        # Create new graph with merged nodes
        merged_graph = ArgDownGraph(nodes=merged_nodes)
        
        return merged_graph
```

### 1.5 BayesDown Enhancement Module

```python
from typing import Dict, List, Optional, Union, Any, Tuple
import re
import json
from dataclasses import dataclass, field

@dataclass
class BayesDownNode:
    """Representation of a node in a BayesDown probabilistic argument structure."""
    title: str
    description: str
    instantiations: List[str] = field(default_factory=list)
    priors: Dict[str, float] = field(default_factory=dict)
    posteriors: Dict[str, float] = field(default_factory=dict)
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    indentation: int = 0
    line_number: Optional[int] = None

@dataclass
class BayesDownGraph:
    """Complete representation of a BayesDown probabilistic argument graph."""
    nodes: Dict[str, BayesDownNode]
    source_document: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_root_nodes(self) -> List[str]:
        """Get the titles of all root nodes (nodes without parents)."""
        return [title for title, node in self.nodes.items() if not node.parents]
    
    def get_leaf_nodes(self) -> List[str]:
        """Get the titles of all leaf nodes (nodes without children)."""
        return [title for title, node in self.nodes.items() if not node.children]
    
    def to_markdown(self) -> str:
        """Convert the graph to BayesDown markdown syntax."""
        # Start with root nodes and traverse depth-first
        roots = self.get_root_nodes()
        lines = []
        
        def process_node(node_title, indent=0):
            node = self.nodes[node_title]
            prefix = " " * indent + ("+ " if indent > 0 else "")
            
            # Format metadata as JSON
            metadata_dict = {
                "instantiations": node.instantiations,
                "priors": node.priors,
                "posteriors": node.posteriors
            }
            metadata_json = json.dumps(metadata_dict)
            
            # Create the node line
            lines.append(f"{prefix}[{node_title}]: {node.description} {metadata_json}")
            
            # Process children
            for child_title in node.children:
                process_node(child_title, indent + 2)
        
        # Process each root node
        for root in roots:
            process_node(root)
        
        return "\n".join(lines)

class BayesDownEnhancer:
    """Class for enhancing ArgDown structures with probabilistic information."""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the BayesDown enhancer.
        
        Args:
            llm_provider: LLM provider for probability extraction
        """
        self.llm_provider = llm_provider
        
        # Default system message for probability extraction
        self.system_message = """
        You are an expert system for extracting probabilistic information from text.
        Your task is to analyze text and identify probability statements, causal relationships,
        and uncertainty information to enhance argument structures with Bayesian probabilities.
        
        Follow these rules:
        1. Identify explicit probability statements in the text
        2. Infer implicit probabilities based on language and context
        3. Ensure probability distributions are coherent (sum to 1.0)
        4. Provide reasonable estimates when exact probabilities are not stated
        5. Format all probabilities as decimal values between 0.0 and 1.0
        
        Your output should be structured as JSON with proper probability distributions.
        """
    
    def argdown_to_bayesdown(self, 
                            argdown_graph: ArgDownGraph, 
                            source_text: str,
                            model: str = "gpt-4",
                            temperature: float = 0.0) -> BayesDownGraph:
        """
        Enhance an ArgDown graph with probabilistic information to create a BayesDown graph.
        
        Args:
            argdown_graph: ArgDown graph to enhance
            source_text: Original source text for context
            model: LLM model to use for probability extraction
            temperature: Sampling temperature for generation
            
        Returns:
            BayesDownGraph object with probabilistic information
        """
        # Create BayesDown nodes from ArgDown nodes
        bayesdown_nodes = {}
        
        for title, node in argdown_graph.nodes.items():
            # Extract probabilities for this node
            priors, posteriors = self._extract_probabilities(
                node=node,
                graph=argdown_graph,
                source_text=source_text,
                model=model,
                temperature=temperature
            )
            
            # Create BayesDown node
            bayesdown_nodes[title] = BayesDownNode(
                title=title,
                description=node.description,
                instantiations=node.instantiations.copy() if node.instantiations else [f"{title}_TRUE", f"{title}_FALSE"],
                priors=priors,
                posteriors=posteriors,
                parents=node.parents.copy(),
                children=node.children.copy(),
                metadata=node.metadata.copy(),
                indentation=node.indentation,
                line_number=node.line_number
            )
        
        # Create BayesDown graph
        bayesdown_graph = BayesDownGraph(
            nodes=bayesdown_nodes,
            source_document=argdown_graph.source_document,
            metadata=argdown_graph.metadata.copy()
        )
        
        return bayesdown_graph
    
    def _extract_probabilities(self, 
                              node: ArgDownNode, 
                              graph: ArgDownGraph, 
                              source_text: str,
                              model: str,
                              temperature: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Extract prior and posterior probabilities for a node.
        
        Args:
            node: ArgDown node to extract probabilities for
            graph: Complete ArgDown graph for context
            source_text: Original source text
            model: LLM model to use for extraction
            temperature: Sampling temperature
            
        Returns:
            Tuple of (prior_probabilities, posterior_probabilities)
        """
        # Get possible instantiations
        instantiations = node.instantiations if node.instantiations else [f"{node.title}_TRUE", f"{node.title}_FALSE"]
        
        # Create schema for structured output
        output_schema = {
            "type": "object",
            "properties": {
                "priors": {
                    "type": "object",
                    "description": "Prior probabilities for each possible state"
                },
                "posteriors": {
                    "type": "object",
                    "description": "Conditional probabilities given parent states"
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of how probabilities were determined"
                }
            },
            "required": ["priors", "posteriors", "explanation"]
        }
        
        # Build context about the node and its relationships
        node_context = f"Node: {node.title}\nDescription: {node.description}\n"
        node_context += f"Possible states: {', '.join(instantiations)}\n"
        
        if node.parents:
            node_context += "Parent nodes:\n"
            for parent in node.parents:
                if parent in graph.nodes:
                    parent_node = graph.nodes[parent]
                    parent_states = parent_node.instantiations if parent_node.instantiations else [f"{parent}_TRUE", f"{parent}_FALSE"]
                    node_context += f"- {parent}: {parent_node.description}\n"
                    node_context += f"  Possible states: {', '.join(parent_states)}\n"
        
        # Create the extraction prompt
        prompt = f"""
        I'm trying to extract probabilistic information for a node in an argument structure.
        Here is the context:

        {node_context}

        Here is the relevant text:

        {source_text}

        Please extract:
        
        1. Prior probabilities for each possible state of {node.title}
           - These are unconditional probabilities P({instantiations[0]}), P({instantiations[1]}), etc.
           - Probabilities must sum to 1.0
        
        2. Posterior probabilities for {node.title} conditional on its parent nodes
           - For each combination of parent states, provide P({node.title}|parent states)
        
        If the text doesn't explicitly state probabilities, make a reasonable estimate based on the language and context.
        Provide your reasoning for each probability estimate.
        """
        
        # Create LLM request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_message=self.system_message
        )
        
        # Extract probabilities using structured output
        result = self.llm_provider.generate_structured_output(request, output_schema)
        
        # Process and validate priors
        priors = result.get("priors", {})
        
        # Ensure we have probabilities for all instantiations
        if len(priors) < len(instantiations):
            # Fill in missing instantiations with estimated values
            remaining_prob = 1.0 - sum(float(p) for p in priors.values())
            missing_states = [s for s in instantiations if f"p({s})" not in priors]
            
            if missing_states and remaining_prob > 0:
                per_state_prob = remaining_prob / len(missing_states)
                for state in missing_states:
                    priors[f"p({state})"] = per_state_prob
        
        # Normalize priors to ensure they sum to 1.0
        prior_sum = sum(float(p) for p in priors.values())
        if prior_sum != 0:  # Avoid division by zero
            normalized_priors = {k: float(v) / prior_sum for k, v in priors.items()}
        else:
            # If no priors, create uniform distribution
            normalized_priors = {f"p({state})": 1.0 / len(instantiations) for state in instantiations}
        
        # Process posteriors
        posteriors = result.get("posteriors", {})
        
        # Return validated probabilities
        return normalized_priors, posteriors
```

### 1.6 Integration with Notebook Structure

```python
# Cell for PDF processing setup
def process_pdf_document(pdf_path, use_ocr=False, segmentation_strategy="auto"):
    """
    Process a PDF document and display information about its contents.
    
    Args:
        pdf_path: Path to the PDF file
        use_ocr: Whether to use OCR for text extraction
        segmentation_strategy: Strategy for document segmentation
        
    Returns:
        Tuple of (DocumentSource object, metadata, segments)
    """
    # Create PDF document source
    doc_source = PDFDocumentSource()
    
    # Load the document
    success = doc_source.load_document(pdf_path, ocr_enabled=use_ocr)
    if not success:
        print(f"Error: Failed to load document from {pdf_path}")
        return None, None, None
    
    # Extract metadata
    metadata = doc_source.extract_metadata()
    
    # Print basic info
    print(f"Document: {metadata.title or 'Untitled'}")
    print(f"Authors: {', '.join(metadata.authors) if metadata.authors else 'Unknown'}")
    print(f"Date: {metadata.date or 'Unknown'}")
    print(f"Pages: {metadata.num_pages}")
    
    # Segment the document
    segments = doc_source.segment_document(strategy=segmentation_strategy)
    print(f"Document segmented into {len(segments)} parts")
    
    # Show first few segments
    for i, segment in enumerate(segments[:3]):
        if i < 2:  # Show content for first 2 segments
            print(f"\nSegment {i+1} - {segment.segment_type} - {segment.section_title or 'Untitled'}:")
            print(f"{segment.content[:200]}...")
        else:  # Just show count
            print(f"\n(+ {len(segments)-2} more segments)")
            break
    
    return doc_source, metadata, segments

# Cell for LLM setup
def setup_llm_provider(provider_type="openai", api_key=None):
    """
    Set up and test the LLM provider.
    
    Args:
        provider_type: Type of provider ('openai' or 'anthropic')
        api_key: API key for the provider
        
    Returns:
        LLMProvider object
    """
    if provider_type.lower() == "openai":
        try:
            # Create OpenAI provider
            provider = OpenAIProvider(api_key=api_key)
            
            # Test with a simple request
            test_request = LLMRequest(
                prompt="Hello, please respond with a simple 'Hello, I'm working properly!' message.",
                model="gpt-3.5-turbo",
                temperature=0.0
            )
            response = provider.generate_completion(test_request)
            
            print("LLM Provider Setup Successful!")
            print(f"Test response: {response.content}")
            print(f"Model: {response.model}")
            print(f"Token usage: {response.usage}")
            
            return provider
        except Exception as e:
            print(f"Error setting up OpenAI provider: {e}")
            return None
    elif provider_type.lower() == "anthropic":
        # Placeholder for Anthropic implementation
        print("Anthropic provider not yet implemented")
        return None
    else:
        print(f"Unknown provider type: {provider_type}")
        return None

# Cell for ArgDown extraction
def extract_argdown_from_document(doc_source, llm_provider, model="gpt-4"):
    """
    Extract ArgDown structure from a document.
    
    Args:
        doc_source: DocumentSource object with loaded document
        llm_provider: LLMProvider for extraction
        model: Model to use for extraction
        
    Returns:
        ArgDownGraph object
    """
    # Create extractor
    extractor = ArgDownExtractor(llm_provider=llm_provider)
    
    # Extract ArgDown
    print("Extracting ArgDown structure from document...")
    argdown_graph = extractor.extract_argdown_from_document(
        document_source=doc_source,
        model=model
    )
    
    # Print results
    print(f"Extraction complete! Found {len(argdown_graph.nodes)} nodes in the argument structure:")
    
    # Show root nodes
    roots = argdown_graph.get_root_nodes()
    print(f"\nRoot nodes ({len(roots)}):")
    for root in roots[:5]:
        node = argdown_graph.nodes[root]
        print(f"- {root}: {node.description[:100]}...")
    
    # Show leaf nodes
    leafs = argdown_graph.get_leaf_nodes()
    print(f"\nLeaf nodes ({len(leafs)}):")
    for leaf in leafs[:5]:
        node = argdown_graph.nodes[leaf]
        print(f"- {leaf}: {node.description[:100]}...")
    
    # Generate ArgDown markdown
    argdown_md = argdown_graph.to_markdown()
    
    # Save to file
    with open("ArgDown.md", "w") as f:
        f.write(argdown_md)
    print("\nArgDown structure saved to ArgDown.md")
    
    return argdown_graph

# Cell for BayesDown enhancement
def enhance_to_bayesdown(argdown_graph, doc_source, llm_provider, model="gpt-4"):
    """
    Enhance ArgDown structure with probabilistic information.
    
    Args:
        argdown_graph: ArgDownGraph object
        doc_source: DocumentSource object with source document
        llm_provider: LLMProvider for extraction
        model: Model to use for extraction
        
    Returns:
        BayesDownGraph object
    """
    # Create enhancer
    enhancer = BayesDownEnhancer(llm_provider=llm_provider)
    
    # Get document text for context
    doc_text = doc_source.extract_text()
    
    # Enhance with probabilities
    print("Enhancing ArgDown to BayesDown with probabilistic information...")
    bayesdown_graph = enhancer.argdown_to_bayesdown(
        argdown_graph=argdown_graph,
        source_text=doc_text,
        model=model
    )
    
    # Print results
    print(f"Enhancement complete! Added probabilistic information to {len(bayesdown_graph.nodes)} nodes.")
    
    # Show sample node with probabilities
    if bayesdown_graph.nodes:
        sample_node_title = next(iter(bayesdown_graph.nodes))
        sample_node = bayesdown_graph.nodes[sample_node_title]
        
        print(f"\nSample node: {sample_node_title}")
        print(f"Description: {sample_node.description}")
        print("Prior probabilities:")
        for prior_key, prior_value in sample_node.priors.items():
            print(f"- {prior_key}: {prior_value}")
        
        if sample_node.posteriors:
            print("\nPosterior probabilities:")
            for posterior_key, posterior_value in sample_node.posteriors.items():
                print(f"- {posterior_key}: {posterior_value}")
    
    # Generate BayesDown markdown
    bayesdown_md = bayesdown_graph.to_markdown()
    
    # Save to file
    with open("BayesDown.md", "w") as f:
        f.write(bayesdown_md)
    print("\nBayesDown structure saved to BayesDown.md")
    
    return bayesdown_graph
```

Let's also implement an improved CSV extraction from BayesDown, which will connect with the existing visualization functions:

```python
# Cell for BayesDown to CSV export
def bayesdown_to_csv(bayesdown_graph):
    """
    Convert BayesDown graph to CSV format compatible with existing visualization.
    
    Args:
        bayesdown_graph: BayesDownGraph object
        
    Returns:
        pandas DataFrame with the CSV structure
    """
    import pandas as pd
    
    # Create DataFrame with required columns
    df = pd.DataFrame(columns=[
        'Title', 'Description', 'line', 'line_numbers', 'indentation',
        'indentation_levels', 'Parents', 'Children', 'instantiations',
        'priors', 'posteriors', 'No_Parent', 'No_Children'
    ])
    
    # Add each node to the DataFrame
    for title, node in bayesdown_graph.nodes.items():
        # Create row dictionary
        row = {
            'Title': title,
            'Description': node.description,
            'line': node.line_number,
            'line_numbers': [node.line_number] if node.line_number is not None else [],
            'indentation': node.indentation,
            'indentation_levels': [node.indentation],
            'Parents': node.parents,
            'Children': node.children,
            'instantiations': node.instantiations,
            'priors': node.priors,
            'posteriors': node.posteriors,
            'No_Parent': len(node.parents) == 0,
            'No_Children': len(node.children) == 0
        }
        
        # Add to DataFrame
        df = df.append(row, ignore_index=True)
    
    # Save to CSV
    df.to_csv('extracted_data.csv', index=False)
    print("BayesDown structure exported to extracted_data.csv")
    
    return df
```

## 2. End-to-End Pipeline Integration

Finally, let's create an end-to-end integration cell that demonstrates the full pipeline:

```python
# End-to-End Pipeline Demo
def run_amtair_pipeline(pdf_path, api_key=None, use_ocr=False, model="gpt-4"):
    """
    Run the complete AMTAIR pipeline from PDF to visualization.
    
    Args:
        pdf_path: Path to the PDF file
        api_key: OpenAI API key (or None to use environment variable)
        use_ocr: Whether to use OCR for text extraction
        model: LLM model to use for extraction
        
    Returns:
        Visualization HTML
    """
    print("## AMTAIR Pipeline: PDF to Bayesian Network ##\n")
    
    # Step 1: Process PDF
    print("Step 1: Processing PDF document...")
    doc_source, metadata, segments = process_pdf_document(pdf_path, use_ocr=use_ocr)
    if not doc_source:
        return "Failed to process document"
    print("\nStep 1 complete.\n" + "-"*50 + "\n")
    
    # Step 2: Set up LLM provider
    print("Step 2: Setting up LLM provider...")
    llm_provider = setup_llm_provider(provider_type="openai", api_key=api_key)
    if not llm_provider:
        return "Failed to set up LLM provider"
    print("\nStep 2 complete.\n" + "-"*50 + "\n")
    
    # Step 3: Extract ArgDown
    print("Step 3: Extracting ArgDown structure...")
    argdown_graph = extract_argdown_from_document(doc_source, llm_provider, model=model)
    print("\nStep 3 complete.\n" + "-"*50 + "\n")
    
    # Step 4: Enhance to BayesDown
    print("Step 4: Enhancing to BayesDown with probabilities...")
    bayesdown_graph = enhance_to_bayesdown(argdown_graph, doc_source, llm_provider, model=model)
    print("\nStep 4 complete.\n" + "-"*50 + "\n")
    
    # Step 5: Export to CSV
    print("Step 5: Exporting to CSV for visualization...")
    df = bayesdown_to_csv(bayesdown_graph)
    print("\nStep 5 complete.\n" + "-"*50 + "\n")
    
    # Step 6: Generate visualization
    print("Step 6: Generating Bayesian network visualization...")
    try:
        visualization = create_bayesian_network_with_probabilities(df)
        print("\nStep 6 complete. Visualization ready!")
        return visualization
    except Exception as e:
        print(f"Error generating visualization: {e}")
        return "Failed to generate visualization"

# Example usage:
# visualization = run_amtair_pipeline("example_paper.pdf")
# display(visualization)
```
</implementation>

This implementation provides a robust foundation for the AMTAIR prototype, focusing on the core architecture and the PDF-to-ArgDown-to-BayesDown pipeline. I've designed the code to be modular, extensible, and well-documented, following best practices for Python development and incorporating comprehensive error handling.

The implementation includes:

1. **Document Processing Module**: A flexible system for handling PDF documents with options for OCR and multiple segmentation strategies.

2. **LLM Provider Interface**: A clean abstraction for interacting with language models, with concrete implementation for OpenAI's API.

3. **ArgDown Extraction Pipeline**: A structured approach to extracting argument structures from text using LLMs.

4. **BayesDown Enhancement Module**: A system for adding probabilistic information to argument structures.

5. **Integration with Notebook Structure**: Convenient functions for using the implementation within the Colab notebook environment.

This code establishes the foundation for a robust and extensible system while providing immediate functionality for the core AMTAIR pipeline. The modular architecture allows for easy extension and refinement in future iterations.

To fully integrate this implementation into the notebook, the next steps would be:

1. Add the code to appropriate cells in the notebook
2. Add UI elements for configuration (file upload, API key input, etc.)
3. Connect with the existing visualization code
4. Create example workflows demonstrating the complete pipeline

The implementation strikes a balance between theoretical rigor and practical usability, providing both a solid foundation for academic research and a functional tool for analyzing AI safety literature.
