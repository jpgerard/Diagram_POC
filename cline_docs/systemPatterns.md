# System Patterns

## Architecture Overview
The system follows a modular architecture with distinct components for each stage of the workflow:

```
[Diagram Upload] → [OCR Processing] → [Graph Construction] → [Requirement Generation] → [Review & Edit] → [Export]
```

## Key Technical Decisions

### 1. Streamlit as Frontend Framework
- Chosen for rapid prototyping capabilities
- Provides built-in widgets for file upload, text editing, and visualization
- Enables quick iteration on UI components

### 2. Modular Component Design
- Each step in the workflow is implemented as a separate module
- Modules communicate through well-defined interfaces
- Enables independent testing and development of each component

### 3. OCR Integration
- Using OCR to extract text from diagrams
- Text extraction results are post-processed to identify components and relationships

### 4. Graph-Based Representation
- NetworkX used to create a graph representation of the diagram
- Nodes represent components identified in the diagram
- Edges represent relationships between components
- Graph structure preserves the semantic meaning of the diagram

### 5. AI-Powered Requirement Generation
- OpenAI GPT-4 used to generate requirements based on graph analysis
- Structured prompting to ensure consistent requirement format
- Requirements are linked back to source components in the graph

### 6. Interactive Editing
- Requirements presented in an editable interface
- Changes tracked and can be applied to specific requirements
- Maintains traceability between edited requirements and source components

### 7. JSON Export
- Standardized JSON format for exporting requirements
- Format includes metadata, traceability information, and requirement details
- Designed for compatibility with common requirement management tools

## Design Patterns

### 1. Pipeline Pattern
- Sequential processing of data through distinct stages
- Each stage has a specific responsibility and output

### 2. Repository Pattern
- Centralized data management for the graph and requirements
- Provides consistent access to data across different components

### 3. Observer Pattern
- UI components observe changes to the underlying data
- Updates are reflected in real-time as requirements are generated or modified

### 4. Strategy Pattern
- Pluggable strategies for OCR processing and requirement generation
- Allows for future extensions with different OCR engines or AI models
