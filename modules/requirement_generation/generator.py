import streamlit as st
import json
import uuid
import time
import pandas as pd
from datetime import datetime

def generate_requirements(graph_results):
    """
    Generate system requirements based on the graph analysis.
    
    Args:
        graph_results (dict): The results from graph construction.
        
    Returns:
        dict: A dictionary containing the generated requirements.
    """
    st.write("### Requirement Generation")
    
    # Extract graph information
    nodes = graph_results["nodes"]
    edges = graph_results["edges"]
    node_attributes = graph_results["node_attributes"]
    
    # Display generation options
    st.write("#### Generation Options")
    
    requirement_types = st.multiselect(
        "Select requirement types to generate",
        ["Functional", "Non-functional", "Interface", "Data", "Security"],
        default=["Functional", "Interface", "Security"]
    )
    
    detail_level = st.slider(
        "Detail level",
        min_value=1,
        max_value=5,
        value=3,
        help="Higher values generate more detailed requirements"
    )
    
    # Call the OpenAI API to generate requirements
    with st.spinner("Generating requirements using GPT-4..."):
        try:
            # Generate requirements based on the graph
            requirements = generate_requirements_with_openai(
                nodes, 
                edges, 
                node_attributes, 
                requirement_types, 
                detail_level
            )
        except Exception as e:
            st.error(f"Error generating requirements: {str(e)}")
            st.error("Please check your OpenAI API key in the .env file.")
            return None
        
        # Display the generated requirements
        display_requirements(requirements)
        
        # Return the requirements
        return requirements

def simulate_requirement_generation(nodes, edges, node_attributes, requirement_types, detail_level):
    """
    Simulate requirement generation for the prototype.
    
    Args:
        nodes (list): List of nodes in the graph.
        edges (list): List of edges in the graph.
        node_attributes (dict): Dictionary of node attributes.
        requirement_types (list): Types of requirements to generate.
        detail_level (int): Level of detail for the requirements.
        
    Returns:
        dict: A dictionary containing the generated requirements.
    """
    # Create a list to store the requirements
    requirements_list = []
    
    # Generate functional requirements
    if "Functional" in requirement_types:
        # Generate requirements for each component
        for node in nodes:
            node_type = node_attributes[node].get("type", "Unknown")
            
            # Generate different requirements based on component type
            if node_type == "Frontend":
                requirements_list.append({
                    "id": f"FR-{str(uuid.uuid4())[:8]}",
                    "type": "Functional",
                    "component": node,
                    "description": f"The {node} shall provide a user interface for interacting with the system.",
                    "priority": "High",
                    "source": "System Diagram",
                    "verification_method": "Demonstration"
                })
                
                if detail_level >= 3:
                    requirements_list.append({
                        "id": f"FR-{str(uuid.uuid4())[:8]}",
                        "type": "Functional",
                        "component": node,
                        "description": f"The {node} shall validate user inputs before submission.",
                        "priority": "Medium",
                        "source": "System Diagram",
                        "verification_method": "Test"
                    })
                
                if detail_level >= 4:
                    requirements_list.append({
                        "id": f"FR-{str(uuid.uuid4())[:8]}",
                        "type": "Functional",
                        "component": node,
                        "description": f"The {node} shall provide feedback for user actions.",
                        "priority": "Medium",
                        "source": "System Diagram",
                        "verification_method": "Demonstration"
                    })
            
            elif node_type == "Service":
                requirements_list.append({
                    "id": f"FR-{str(uuid.uuid4())[:8]}",
                    "type": "Functional",
                    "component": node,
                    "description": f"The {node} shall process requests according to business rules.",
                    "priority": "High",
                    "source": "System Diagram",
                    "verification_method": "Test"
                })
                
                if detail_level >= 3:
                    requirements_list.append({
                        "id": f"FR-{str(uuid.uuid4())[:8]}",
                        "type": "Functional",
                        "component": node,
                        "description": f"The {node} shall handle error conditions gracefully.",
                        "priority": "Medium",
                        "source": "System Diagram",
                        "verification_method": "Test"
                    })
                
                if detail_level >= 5:
                    requirements_list.append({
                        "id": f"FR-{str(uuid.uuid4())[:8]}",
                        "type": "Functional",
                        "component": node,
                        "description": f"The {node} shall log all operations for audit purposes.",
                        "priority": "Low",
                        "source": "System Diagram",
                        "verification_method": "Inspection"
                    })
            
            elif node_type == "Database":
                requirements_list.append({
                    "id": f"FR-{str(uuid.uuid4())[:8]}",
                    "type": "Functional",
                    "component": node,
                    "description": f"The {node} shall store and retrieve data as requested by services.",
                    "priority": "High",
                    "source": "System Diagram",
                    "verification_method": "Test"
                })
                
                if detail_level >= 3:
                    requirements_list.append({
                        "id": f"FR-{str(uuid.uuid4())[:8]}",
                        "type": "Functional",
                        "component": node,
                        "description": f"The {node} shall maintain data integrity through transactions.",
                        "priority": "High",
                        "source": "System Diagram",
                        "verification_method": "Test"
                    })
                
                if detail_level >= 4:
                    requirements_list.append({
                        "id": f"FR-{str(uuid.uuid4())[:8]}",
                        "type": "Functional",
                        "component": node,
                        "description": f"The {node} shall provide backup and recovery mechanisms.",
                        "priority": "Medium",
                        "source": "System Diagram",
                        "verification_method": "Demonstration"
                    })
    
    # Generate interface requirements
    if "Interface" in requirement_types:
        # Generate requirements for each edge (relationship)
        for source, target in edges:
            source_type = node_attributes[source].get("type", "Unknown")
            target_type = node_attributes[target].get("type", "Unknown")
            
            requirements_list.append({
                "id": f"IR-{str(uuid.uuid4())[:8]}",
                "type": "Interface",
                "component": f"{source} -> {target}",
                "description": f"The {source} shall communicate with the {target} using a well-defined API.",
                "priority": "High",
                "source": "System Diagram",
                "verification_method": "Test"
            })
            
            if detail_level >= 3:
                requirements_list.append({
                    "id": f"IR-{str(uuid.uuid4())[:8]}",
                    "type": "Interface",
                    "component": f"{source} -> {target}",
                    "description": f"The interface between {source} and {target} shall handle communication errors.",
                    "priority": "Medium",
                    "source": "System Diagram",
                    "verification_method": "Test"
                })
            
            if detail_level >= 5 and source_type == "Service" and target_type == "Database":
                requirements_list.append({
                    "id": f"IR-{str(uuid.uuid4())[:8]}",
                    "type": "Interface",
                    "component": f"{source} -> {target}",
                    "description": f"The {source} shall use connection pooling when accessing the {target}.",
                    "priority": "Low",
                    "source": "System Diagram",
                    "verification_method": "Inspection"
                })
    
    # Generate security requirements
    if "Security" in requirement_types:
        # Add general security requirements
        requirements_list.append({
            "id": f"SR-{str(uuid.uuid4())[:8]}",
            "type": "Security",
            "component": "System-wide",
            "description": "The system shall encrypt all sensitive data in transit and at rest.",
            "priority": "High",
            "source": "System Diagram",
            "verification_method": "Test"
        })
        
        # Add component-specific security requirements
        for node in nodes:
            node_type = node_attributes[node].get("type", "Unknown")
            
            if node_type == "Service" and node == "Authentication Service":
                requirements_list.append({
                    "id": f"SR-{str(uuid.uuid4())[:8]}",
                    "type": "Security",
                    "component": node,
                    "description": f"The {node} shall implement multi-factor authentication.",
                    "priority": "High",
                    "source": "System Diagram",
                    "verification_method": "Test"
                })
                
                if detail_level >= 4:
                    requirements_list.append({
                        "id": f"SR-{str(uuid.uuid4())[:8]}",
                        "type": "Security",
                        "component": node,
                        "description": f"The {node} shall lock accounts after multiple failed login attempts.",
                        "priority": "Medium",
                        "source": "System Diagram",
                        "verification_method": "Test"
                    })
            
            elif node_type == "Database":
                requirements_list.append({
                    "id": f"SR-{str(uuid.uuid4())[:8]}",
                    "type": "Security",
                    "component": node,
                    "description": f"The {node} shall restrict access to authorized services only.",
                    "priority": "High",
                    "source": "System Diagram",
                    "verification_method": "Inspection"
                })
    
    # Generate non-functional requirements
    if "Non-functional" in requirement_types:
        # Add performance requirements
        requirements_list.append({
            "id": f"NFR-{str(uuid.uuid4())[:8]}",
            "type": "Non-functional",
            "component": "System-wide",
            "description": "The system shall respond to user requests within 2 seconds under normal load.",
            "priority": "Medium",
            "source": "System Diagram",
            "verification_method": "Test"
        })
        
        # Add reliability requirements
        if detail_level >= 3:
            requirements_list.append({
                "id": f"NFR-{str(uuid.uuid4())[:8]}",
                "type": "Non-functional",
                "component": "System-wide",
                "description": "The system shall have an uptime of at least 99.9%.",
                "priority": "High",
                "source": "System Diagram",
                "verification_method": "Analysis"
            })
        
        # Add scalability requirements
        if detail_level >= 4:
            requirements_list.append({
                "id": f"NFR-{str(uuid.uuid4())[:8]}",
                "type": "Non-functional",
                "component": "System-wide",
                "description": "The system shall support at least 1000 concurrent users.",
                "priority": "Medium",
                "source": "System Diagram",
                "verification_method": "Test"
            })
    
    # Generate data requirements
    if "Data" in requirement_types:
        # Add data retention requirements
        requirements_list.append({
            "id": f"DR-{str(uuid.uuid4())[:8]}",
            "type": "Data",
            "component": "System-wide",
            "description": "The system shall retain user data for a period of 7 years.",
            "priority": "Medium",
            "source": "System Diagram",
            "verification_method": "Inspection"
        })
        
        # Add data validation requirements
        if detail_level >= 3:
            requirements_list.append({
                "id": f"DR-{str(uuid.uuid4())[:8]}",
                "type": "Data",
                "component": "System-wide",
                "description": "The system shall validate all data inputs against defined schemas.",
                "priority": "High",
                "source": "System Diagram",
                "verification_method": "Test"
            })
        
        # Add specific data requirements for databases
        for node in nodes:
            node_type = node_attributes[node].get("type", "Unknown")
            
            if node_type == "Database" and detail_level >= 4:
                requirements_list.append({
                    "id": f"DR-{str(uuid.uuid4())[:8]}",
                    "type": "Data",
                    "component": node,
                    "description": f"The {node} shall implement data versioning for audit purposes.",
                    "priority": "Low",
                    "source": "System Diagram",
                    "verification_method": "Inspection"
                })
    
    # Create the requirements dictionary
    requirements = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_diagram": "Uploaded Diagram",
            "requirement_types": requirement_types,
            "detail_level": detail_level
        },
        "requirements": requirements_list
    }
    
    return requirements

def generate_requirements_with_openai(nodes, edges, node_attributes, requirement_types, detail_level):
    """
    Generate requirements using the OpenAI API.
    
    Args:
        nodes (list): List of nodes in the graph.
        edges (list): List of edges in the graph.
        node_attributes (dict): Dictionary of node attributes.
        requirement_types (list): Types of requirements to generate.
        detail_level (int): Level of detail for the requirements.
        
    Returns:
        dict: A dictionary containing the generated requirements.
    """
    import os
    from openai import OpenAI
    from datetime import datetime
    
    # Get OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("OpenAI API key not found. Using simulated requirements instead.")
        return simulate_requirement_generation(nodes, edges, node_attributes, requirement_types, detail_level)
    
    try:
        # Prepare the system diagram information
        system_info = {
            "nodes": [],
            "relationships": []
        }
        
        # Add node information
        for node in nodes:
            node_info = {
                "name": node,
                "type": node_attributes[node].get("type", "Unknown"),
                "description": node_attributes[node].get("description", "")
            }
            system_info["nodes"].append(node_info)
        
        # Add relationship information
        for source, target in edges:
            relationship = {
                "source": source,
                "target": target,
                "source_type": node_attributes[source].get("type", "Unknown"),
                "target_type": node_attributes[target].get("type", "Unknown")
            }
            system_info["relationships"].append(relationship)
        
        # Create the prompt for OpenAI
        prompt = f"""
        You are a requirements engineering expert. Based on the system diagram information provided below, 
        generate a set of system requirements. The requirements should be traceable to the components in the diagram.
        
        System Diagram Information:
        Components:
        {json.dumps(system_info["nodes"], indent=2)}
        
        Relationships:
        {json.dumps(system_info["relationships"], indent=2)}
        
        Generate the following types of requirements: {', '.join(requirement_types)}
        
        Detail level: {detail_level} (on a scale of 1-5, where 5 is most detailed)
        
        For each requirement, provide:
        1. A unique ID (format: [Type prefix]-[UUID], e.g., FR-12345678 for Functional Requirements)
        2. Type (one of: {', '.join(requirement_types)})
        3. Component (the component it relates to, or "System-wide")
        4. Description (clear, concise, and testable)
        5. Priority (High, Medium, or Low)
        6. Source (e.g., "System Diagram")
        7. Verification method (Test, Demonstration, Inspection, or Analysis)
        
        Return the requirements in JSON format with the following structure:
        {{
          "metadata": {{
            "generated_at": "YYYY-MM-DD HH:MM:SS",
            "source_diagram": "Uploaded Diagram",
            "requirement_types": ["Type1", "Type2", ...],
            "detail_level": X
          }},
          "requirements": [
            {{
              "id": "XX-12345678",
              "type": "Type",
              "component": "Component Name",
              "description": "The component shall...",
              "priority": "Priority",
              "source": "Source",
              "verification_method": "Method"
            }},
            ...
          ]
        }}
        """
        
        # Initialize the OpenAI client with default settings
        # Initialize the OpenAI client with default settings
        client = OpenAI(api_key=api_key)
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a requirements engineering expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        # Extract the response content
        content = response.choices[0].message.content
        
        # Parse the JSON response
        # Find the JSON part in the response (it might be surrounded by markdown code blocks)
        import re
        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = content
        
        # Clean up the string and parse it
        json_str = json_str.strip()
        requirements = json.loads(json_str)
        
        # Add generation timestamp if not present
        if "metadata" not in requirements:
            requirements["metadata"] = {}
        
        if "generated_at" not in requirements["metadata"]:
            requirements["metadata"]["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if "source_diagram" not in requirements["metadata"]:
            requirements["metadata"]["source_diagram"] = "Uploaded Diagram"
        
        if "requirement_types" not in requirements["metadata"]:
            requirements["metadata"]["requirement_types"] = requirement_types
        
        if "detail_level" not in requirements["metadata"]:
            requirements["metadata"]["detail_level"] = detail_level
        
        return requirements
        
    except Exception as e:
        # If OpenAI API fails, fall back to simulated requirements
        st.warning(f"OpenAI API call failed: {str(e)}. Falling back to simulated requirements.")
        return simulate_requirement_generation(nodes, edges, node_attributes, requirement_types, detail_level)

def display_requirements(requirements):
    """
    Display the generated requirements in the Streamlit UI.
    
    Args:
        requirements (dict): The generated requirements.
    """
    try:
        # Display metadata
        st.write("#### Requirement Generation Metadata")
        st.write(f"Generated at: {requirements['metadata']['generated_at']}")
        st.write(f"Source: {requirements['metadata']['source_diagram']}")
        st.write(f"Requirement types: {', '.join(requirements['metadata']['requirement_types'])}")
        st.write(f"Detail level: {requirements['metadata']['detail_level']}")
        
        # Create a dataframe for the requirements
        if "requirements" in requirements and requirements["requirements"]:
            df = pd.DataFrame(requirements["requirements"])
            
            # Check if the dataframe has a 'type' column
            if "type" in df.columns:
                # Display requirements by type
                st.write("#### Generated Requirements")
                
                # Create tabs for different requirement types
                requirement_types = df["type"].unique().tolist()
                tabs = st.tabs(requirement_types)
                
                for i, req_type in enumerate(requirement_types):
                    with tabs[i]:
                        type_df = df[df["type"] == req_type]
                        st.dataframe(type_df, use_container_width=True)
                        st.write(f"Total {req_type} Requirements: {len(type_df)}")
                
                # Display total requirements
                st.write(f"Total Requirements: {len(df)}")
            else:
                # If no 'type' column, display all requirements in a single table
                st.write("#### Generated Requirements")
                st.dataframe(df, use_container_width=True)
                st.write(f"Total Requirements: {len(df)}")
        else:
            st.warning("No requirements were generated.")
        
        # Display requirements as JSON
        if st.checkbox("Show Requirements as JSON"):
            st.json(requirements)
    except Exception as e:
        st.error(f"Error displaying requirements: {str(e)}")
        st.json(requirements)  # Display the raw JSON as a fallback
