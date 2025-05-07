import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime

def export_to_json(requirements):
    """
    Export the requirements to JSON format.
    
    Args:
        requirements (dict): The requirements to export.
        
    Returns:
        None
    """
    st.write("### Export Requirements")
    
    # Display export options
    st.write("#### Export Options")
    
    # Option to include metadata
    include_metadata = st.checkbox("Include metadata", value=True)
    
    # Option to filter by requirement type
    requirement_types = [req["type"] for req in requirements["requirements"]]
    requirement_types = list(set(requirement_types))
    
    selected_types = st.multiselect(
        "Select requirement types to export",
        requirement_types,
        default=requirement_types
    )
    
    # Option to filter by priority
    priorities = ["High", "Medium", "Low"]
    selected_priorities = st.multiselect(
        "Select priorities to export",
        priorities,
        default=priorities
    )
    
    # Option to customize the filename
    default_filename = f"requirements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filename = st.text_input("Filename", default_filename)
    
    # Filter the requirements based on the selected options
    filtered_requirements = [
        req for req in requirements["requirements"]
        if req["type"] in selected_types and req["priority"] in selected_priorities
    ]
    
    # Create the export data
    export_data = {}
    
    if include_metadata:
        export_data["metadata"] = requirements["metadata"]
    
    export_data["requirements"] = filtered_requirements
    
    # Add export timestamp
    export_data["export_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Display a preview of the export
    st.write("#### Export Preview")
    
    # Create a dataframe for the filtered requirements
    df = pd.DataFrame(filtered_requirements)
    
    # Display the dataframe
    st.dataframe(df)
    
    st.write(f"Total requirements to export: {len(filtered_requirements)}")
    
    # Display the JSON preview
    if st.checkbox("Show JSON Preview"):
        st.json(export_data)
    
    # Export the requirements
    if st.button("Export Requirements"):
        # Create the output directory if it doesn't exist
        output_dir = os.path.join("data", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the full file path
        file_path = os.path.join(output_dir, filename)
        
        # Write the JSON file
        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2)
        
        st.success(f"Requirements exported to {file_path}")
        
        # Provide a download link
        with open(file_path, "r") as f:
            file_content = f.read()
        
        st.download_button(
            label="Download JSON File",
            data=file_content,
            file_name=filename,
            mime="application/json"
        )
    
    # Option to export as CSV
    if st.checkbox("Export as CSV"):
        csv_filename = filename.replace(".json", ".csv")
        
        if st.button("Export CSV"):
            # Create the output directory if it doesn't exist
            output_dir = os.path.join("data", "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create the full file path
            file_path = os.path.join(output_dir, csv_filename)
            
            # Convert to CSV
            df.to_csv(file_path, index=False)
            
            st.success(f"Requirements exported to {file_path}")
            
            # Provide a download link
            with open(file_path, "r") as f:
                file_content = f.read()
            
            st.download_button(
                label="Download CSV File",
                data=file_content,
                file_name=csv_filename,
                mime="text/csv"
            )
    
    # Display export summary
    st.write("#### Export Summary")
    st.write(f"File format: {'JSON' + (' and CSV' if st.session_state.get('export_csv', False) else '')}")
    st.write(f"Total requirements: {len(filtered_requirements)}")
    st.write(f"Requirement types: {', '.join(selected_types)}")
    st.write(f"Priorities: {', '.join(selected_priorities)}")
    
    # Display traceability information
    st.write("#### Traceability Information")
    st.write("Each requirement includes:")
    st.write("- Unique ID")
    st.write("- Component source")
    st.write("- Requirement type")
    st.write("- Priority")
    st.write("- Verification method")
    
    # Return the export data for potential further processing
    return export_data
