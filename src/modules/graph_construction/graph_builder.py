import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def build_graph(ocr_results):
    """
    Build a graph representation of the components identified in the diagram.
    
    Args:
        ocr_results (dict): The results from OCR processing.
        
    Returns:
        dict: A dictionary containing the graph and metadata.
    """
    st.write("### Graph Construction")
    
    # Extract components from OCR results
    components = ocr_results["components"]
    
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for component in components:
        G.add_node(
            component["name"],
            type=component["type"],
            description=component["description"]
        )
    
    # Extract relationships from the OCR text
    relationships = extract_relationships(ocr_results["text"], components)
    
    # Add edges to the graph
    for source, target in relationships:
        # Check if both source and target exist in the graph
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target)
        else:
            st.warning(f"Relationship {source} -> {target} contains components not found in the graph.")
    
    # Display graph statistics
    st.write("#### Graph Statistics")
    st.write(f"Number of nodes: {G.number_of_nodes()}")
    st.write(f"Number of edges: {G.number_of_edges()}")
    
    # Display node types
    node_types = {}
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "Unknown")
        if node_type in node_types:
            node_types[node_type] += 1
        else:
            node_types[node_type] = 1
    
    st.write("#### Component Types")
    for node_type, count in node_types.items():
        st.write(f"- {node_type}: {count}")
    
    # No visualization as requested
    viz_type = "None"
    
    # Create a dataframe of nodes and edges for display
    nodes_df = pd.DataFrame([
        {
            "Name": node,
            "Type": data.get("type", "Unknown"),
            "Description": data.get("description", "")
        }
        for node, data in G.nodes(data=True)
    ])
    
    edges_df = pd.DataFrame([
        {
            "Source": source,
            "Target": target
        }
        for source, target in G.edges()
    ])
    
    # Display node and edge tables
    st.write("#### Component Details")
    st.dataframe(nodes_df)
    
    st.write("#### Relationship Details")
    st.dataframe(edges_df)
    
    # Return the graph and metadata
    graph_results = {
        "graph": G,
        "nodes": list(G.nodes()),
        "edges": list(G.edges()),
        "node_attributes": {node: data for node, data in G.nodes(data=True)},
        "visualization_type": viz_type
    }
    
    return graph_results

def display_networkx_graph(G):
    """
    Display the graph using NetworkX and Matplotlib.
    
    Args:
        G (networkx.DiGraph): The graph to display.
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define node colors based on type
    node_colors = []
    for node, data in G.nodes(data=True):
        if data.get("type") == "Frontend":
            node_colors.append("skyblue")
        elif data.get("type") == "Service":
            node_colors.append("lightgreen")
        elif data.get("type") == "Database":
            node_colors.append("salmon")
        else:
            node_colors.append("gray")
    
    # Define node positions using a layout algorithm
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color="gray", arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Remove axis
    plt.axis("off")
    
    # Display the plot
    st.pyplot(fig)
    
    # Add a legend
    st.write("**Legend:**")
    st.write("- Blue: Frontend components")
    st.write("- Green: Service components")
    st.write("- Red: Database components")

def extract_relationships(text, components):
    """
    Extract relationships between components from the OCR text.
    
    Args:
        text (str): The OCR extracted text.
        components (list): List of component dictionaries.
        
    Returns:
        list: A list of tuples representing relationships (source, target).
    """
    relationships = []
    component_names = [comp["name"] for comp in components]
    
    # Split the text into lines
    lines = text.strip().split('\n')
    
    # Look for explicit relationship statements
    for i, line in enumerate(lines):
        line = line.lower().strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Look for relationship indicators
        relationship_indicators = [
            "connects to", "connected to", "links to", "linked to",
            "talks to", "communicates with", "sends data to",
            "receives data from", "depends on", "uses", "calls",
            "invokes", "triggers", "->", "=>", "→"
        ]
        
        for indicator in relationship_indicators:
            if indicator in line:
                parts = line.split(indicator)
                if len(parts) == 2:
                    source_text = parts[0].strip()
                    target_text = parts[1].strip()
                    
                    # Find the closest matching component names
                    source = find_closest_component(source_text, component_names)
                    target = find_closest_component(target_text, component_names)
                    
                    if source and target and source != target:
                        relationships.append((source, target))
    
    # If no explicit relationships were found, try to infer from diagram structure
    if not relationships:
        # Look for arrow patterns like "A -> B" or "A | v | B"
        arrow_patterns = [
            "->", "=>", "→", "|", "v", "V"
        ]
        
        for i in range(len(lines) - 1):
            current_line = lines[i].strip()
            next_line = lines[i + 1].strip()
            
            # Skip lines that don't look like component names
            if not current_line or any(p in current_line for p in arrow_patterns):
                continue
                
            # If the next line contains an arrow pattern, look at the line after
            if any(p in next_line for p in arrow_patterns) and i + 2 < len(lines):
                target_line = lines[i + 2].strip()
                
                # Find the closest matching component names
                source = find_closest_component(current_line, component_names)
                target = find_closest_component(target_line, component_names)
                
                if source and target and source != target:
                    relationships.append((source, target))
    
    # If still no relationships, try to infer based on component types
    if not relationships:
        # Typical patterns: Frontend -> Service -> Database
        frontends = [comp["name"] for comp in components if comp["type"] == "Frontend"]
        services = [comp["name"] for comp in components if comp["type"] == "Service"]
        databases = [comp["name"] for comp in components if comp["type"] == "Database"]
        
        # Connect frontends to services
        for frontend in frontends:
            for service in services:
                relationships.append((frontend, service))
                
        # Connect services to databases
        for service in services:
            for database in databases:
                # Try to match services with related databases
                if any(word in service.lower() and word in database.lower() 
                       for word in ["user", "product", "order", "auth", "payment"]):
                    relationships.append((service, database))
    
    # Remove duplicates while preserving order
    unique_relationships = []
    for rel in relationships:
        if rel not in unique_relationships:
            unique_relationships.append(rel)
    
    return unique_relationships

def find_closest_component(text, component_names):
    """
    Find the closest matching component name in the text.
    
    Args:
        text (str): The text to search in.
        component_names (list): List of component names.
        
    Returns:
        str: The closest matching component name, or None if no match.
    """
    # First try exact matches
    for name in component_names:
        if name.lower() in text.lower():
            return name
            
    # If no exact match, try partial matches
    for name in component_names:
        name_parts = name.lower().split()
        for part in name_parts:
            if len(part) > 3 and part in text.lower():  # Only consider parts longer than 3 chars
                return name
                
    # If still no match, return None
    return None

def display_plotly_graph(G):
    """
    Display the graph using Plotly for interactive visualization.
    
    Args:
        G (networkx.DiGraph): The graph to display.
    """
    # Define node positions using a layout algorithm
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Type: {data.get('type', 'Unknown')}<br>{data.get('description', '')}")
        
        # Set node color based on type
        if data.get("type") == "Frontend":
            node_color.append("skyblue")
        elif data.get("type") == "Service":
            node_color.append("lightgreen")
        elif data.get("type") == "Database":
            node_color.append("salmon")
        else:
            node_color.append("gray")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=15,
            line=dict(width=2, color='#333')
        )
    )
    
    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Interactive System Component Graph',
            titlefont=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Add a legend
    st.write("**Legend:**")
    st.write("- Blue: Frontend components")
    st.write("- Green: Service components")
    st.write("- Red: Database components")
