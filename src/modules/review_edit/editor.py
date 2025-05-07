import streamlit as st
import pandas as pd
import json
import uuid
from datetime import datetime

def display_requirements_editor(requirements):
    """
    Display an interface for reviewing and editing the generated requirements.
    
    Args:
        requirements (dict): The generated requirements.
        
    Returns:
        dict: The edited requirements.
    """
    st.write("### Review & Edit Requirements")
    
    # Make a copy of the requirements to avoid modifying the original
    edited_requirements = requirements.copy()
    
    # Display metadata
    st.write("#### Requirement Metadata")
    st.write(f"Generated at: {requirements['metadata']['generated_at']}")
    st.write(f"Source: {requirements['metadata']['source_diagram']}")
    st.write(f"Requirement types: {', '.join(requirements['metadata']['requirement_types'])}")
    st.write(f"Detail level: {requirements['metadata']['detail_level']}")
    
    # Create a dataframe for the requirements
    df = pd.DataFrame(requirements["requirements"])
    
    # Add editing options
    st.write("#### Editing Options")
    
    edit_option = st.radio(
        "Select editing mode",
        ["Bulk Edit", "Individual Edit"],
        index=0
    )
    
    if edit_option == "Bulk Edit":
        edited_requirements = bulk_edit_requirements(requirements, df)
    else:
        edited_requirements = individual_edit_requirements(requirements, df)
    
    # Update metadata
    edited_requirements["metadata"]["last_edited"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Display summary of changes
    st.write("#### Summary of Changes")
    
    original_count = len(requirements["requirements"])
    edited_count = len(edited_requirements["requirements"])
    
    st.write(f"Original requirements: {original_count}")
    st.write(f"Edited requirements: {edited_count}")
    st.write(f"Added: {max(0, edited_count - original_count)}")
    st.write(f"Removed: {max(0, original_count - edited_count)}")
    
    # Option to view the edited requirements as JSON
    if st.checkbox("Show Edited Requirements as JSON"):
        st.json(edited_requirements)
    
    return edited_requirements

def bulk_edit_requirements(requirements, df):
    """
    Provide a bulk editing interface for the requirements.
    
    Args:
        requirements (dict): The original requirements.
        df (pd.DataFrame): DataFrame of the requirements.
        
    Returns:
        dict: The edited requirements.
    """
    st.write("#### Bulk Edit Requirements")
    
    # Create tabs for different requirement types
    requirement_types = df["type"].unique().tolist()
    tabs = st.tabs(requirement_types)
    
    # Create a copy of the requirements list
    edited_requirements_list = requirements["requirements"].copy()
    
    # Track which requirements have been deleted
    deleted_ids = []
    
    for i, req_type in enumerate(requirement_types):
        with tabs[i]:
            # Filter requirements by type
            type_df = df[df["type"] == req_type].reset_index(drop=True)
            
            # Display the requirements in an editable table
            st.write(f"#### {req_type} Requirements")
            
            # For each requirement, provide editing options
            for j, row in type_df.iterrows():
                req_id = row["id"]
                
                # Create an expander for each requirement
                with st.expander(f"{req_id}: {row['description'][:100]}..."):
                    # Display the current requirement details
                    st.write(f"**Component:** {row['component']}")
                    st.write(f"**Priority:** {row['priority']}")
                    st.write(f"**Verification Method:** {row['verification_method']}")
                    
                    # Provide editing fields
                    new_description = st.text_area(
                        f"Description ({req_id})",
                        row["description"],
                        key=f"desc_{req_id}"
                    )
                    
                    new_priority = st.selectbox(
                        f"Priority ({req_id})",
                        ["High", "Medium", "Low"],
                        index=["High", "Medium", "Low"].index(row["priority"]),
                        key=f"prio_{req_id}"
                    )
                    
                    new_verification = st.selectbox(
                        f"Verification Method ({req_id})",
                        ["Test", "Demonstration", "Inspection", "Analysis"],
                        index=["Test", "Demonstration", "Inspection", "Analysis"].index(row["verification_method"]),
                        key=f"verif_{req_id}"
                    )
                    
                    # Option to delete the requirement
                    if st.checkbox(f"Delete this requirement ({req_id})", key=f"del_{req_id}"):
                        deleted_ids.append(req_id)
                        st.warning(f"Requirement {req_id} will be deleted.")
                    
                    # Update the requirement in the list
                    for k, req in enumerate(edited_requirements_list):
                        if req["id"] == req_id:
                            edited_requirements_list[k]["description"] = new_description
                            edited_requirements_list[k]["priority"] = new_priority
                            edited_requirements_list[k]["verification_method"] = new_verification
                            break
    
    # Add option to create a new requirement
    st.write("#### Add New Requirement")
    
    new_req_type = st.selectbox(
        "Requirement Type",
        requirement_types
    )
    
    new_req_component = st.text_input(
        "Component",
        "System-wide"
    )
    
    new_req_description = st.text_area(
        "Description",
        "The system shall..."
    )
    
    new_req_priority = st.selectbox(
        "Priority",
        ["High", "Medium", "Low"]
    )
    
    new_req_verification = st.selectbox(
        "Verification Method",
        ["Test", "Demonstration", "Inspection", "Analysis"]
    )
    
    if st.button("Add Requirement"):
        # Generate a new ID based on the requirement type
        prefix = "FR"
        if new_req_type == "Interface":
            prefix = "IR"
        elif new_req_type == "Security":
            prefix = "SR"
        elif new_req_type == "Non-functional":
            prefix = "NFR"
        elif new_req_type == "Data":
            prefix = "DR"
        
        new_id = f"{prefix}-{str(uuid.uuid4())[:8]}"
        
        # Create the new requirement
        new_requirement = {
            "id": new_id,
            "type": new_req_type,
            "component": new_req_component,
            "description": new_req_description,
            "priority": new_req_priority,
            "source": "User Added",
            "verification_method": new_req_verification
        }
        
        # Add the new requirement to the list
        edited_requirements_list.append(new_requirement)
        
        st.success(f"Added new requirement with ID: {new_id}")
    
    # Remove deleted requirements
    edited_requirements_list = [req for req in edited_requirements_list if req["id"] not in deleted_ids]
    
    # Create the edited requirements dictionary
    edited_requirements = {
        "metadata": requirements["metadata"].copy(),
        "requirements": edited_requirements_list
    }
    
    return edited_requirements

def individual_edit_requirements(requirements, df):
    """
    Provide an individual editing interface for the requirements.
    
    Args:
        requirements (dict): The original requirements.
        df (pd.DataFrame): DataFrame of the requirements.
        
    Returns:
        dict: The edited requirements.
    """
    st.write("#### Individual Edit Requirements")
    
    # Create a copy of the requirements list
    edited_requirements_list = requirements["requirements"].copy()
    
    # Create a selectbox to choose a requirement to edit
    requirement_options = [f"{row['id']}: {row['description'][:50]}..." for _, row in df.iterrows()]
    selected_req_index = st.selectbox(
        "Select a requirement to edit",
        range(len(requirement_options)),
        format_func=lambda x: requirement_options[x]
    )
    
    # Get the selected requirement
    selected_req_id = df.iloc[selected_req_index]["id"]
    selected_req = None
    
    for req in edited_requirements_list:
        if req["id"] == selected_req_id:
            selected_req = req
            break
    
    if selected_req:
        st.write("#### Edit Requirement")
        
        # Display the current requirement details
        st.write(f"**ID:** {selected_req['id']}")
        st.write(f"**Type:** {selected_req['type']}")
        
        # Provide editing fields
        new_component = st.text_input(
            "Component",
            selected_req["component"]
        )
        
        new_description = st.text_area(
            "Description",
            selected_req["description"]
        )
        
        new_priority = st.selectbox(
            "Priority",
            ["High", "Medium", "Low"],
            index=["High", "Medium", "Low"].index(selected_req["priority"])
        )
        
        new_verification = st.selectbox(
            "Verification Method",
            ["Test", "Demonstration", "Inspection", "Analysis"],
            index=["Test", "Demonstration", "Inspection", "Analysis"].index(selected_req["verification_method"])
        )
        
        # Update the requirement
        if st.button("Update Requirement"):
            for i, req in enumerate(edited_requirements_list):
                if req["id"] == selected_req_id:
                    edited_requirements_list[i]["component"] = new_component
                    edited_requirements_list[i]["description"] = new_description
                    edited_requirements_list[i]["priority"] = new_priority
                    edited_requirements_list[i]["verification_method"] = new_verification
                    break
            
            st.success(f"Updated requirement {selected_req_id}")
        
        # Option to delete the requirement
        if st.button("Delete Requirement"):
            edited_requirements_list = [req for req in edited_requirements_list if req["id"] != selected_req_id]
            st.warning(f"Deleted requirement {selected_req_id}")
    
    # Add option to create a new requirement
    st.write("#### Add New Requirement")
    
    new_req_type = st.selectbox(
        "Requirement Type",
        df["type"].unique()
    )
    
    new_req_component = st.text_input(
        "Component",
        "System-wide",
        key="new_component"
    )
    
    new_req_description = st.text_area(
        "Description",
        "The system shall...",
        key="new_description"
    )
    
    new_req_priority = st.selectbox(
        "Priority",
        ["High", "Medium", "Low"],
        key="new_priority"
    )
    
    new_req_verification = st.selectbox(
        "Verification Method",
        ["Test", "Demonstration", "Inspection", "Analysis"],
        key="new_verification"
    )
    
    if st.button("Add Requirement", key="add_button"):
        # Generate a new ID based on the requirement type
        prefix = "FR"
        if new_req_type == "Interface":
            prefix = "IR"
        elif new_req_type == "Security":
            prefix = "SR"
        elif new_req_type == "Non-functional":
            prefix = "NFR"
        elif new_req_type == "Data":
            prefix = "DR"
        
        new_id = f"{prefix}-{str(uuid.uuid4())[:8]}"
        
        # Create the new requirement
        new_requirement = {
            "id": new_id,
            "type": new_req_type,
            "component": new_req_component,
            "description": new_req_description,
            "priority": new_req_priority,
            "source": "User Added",
            "verification_method": new_req_verification
        }
        
        # Add the new requirement to the list
        edited_requirements_list.append(new_requirement)
        
        st.success(f"Added new requirement with ID: {new_id}")
    
    # Create the edited requirements dictionary
    edited_requirements = {
        "metadata": requirements["metadata"].copy(),
        "requirements": edited_requirements_list
    }
    
    return edited_requirements
