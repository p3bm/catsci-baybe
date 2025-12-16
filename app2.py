from baybe import Campaign
from baybe.objectives import SingleTargetObjective, DesirabilityObjective
from baybe.parameters import NumericalDiscreteParameter, NumericalContinuousParameter, CategoricalParameter, SubstanceParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from typing import Dict, List, Tuple, Optional, Union
import streamlit as st
import pandas as pd
import json
from io import StringIO
from datetime import datetime
import matplotlib.pyplot as plt
import shap
from streamlit_shap import st_shap

def convert_substance_variable(
    sub_dict: Dict[str, str],
    name: str,
) -> SubstanceParameter:
    """
    Create a BayBE SubstanceParameter from a name-to-SMILES mapping.

    Args:
        sub_dict: Mapping of substance labels to SMILES strings.
        name: Name of the substance parameter.

    Returns:
        A BayBE SubstanceParameter using Mordred descriptors.
    """
    return SubstanceParameter(name, data=sub_dict, encoding="MORDRED")


def convert_categorical_variable(
    cat_list: List[str],
    name: str,
) -> CategoricalParameter:
    """
    Create a BayBE CategoricalParameter.

    Args:
        cat_list: List of allowed categorical values.
        name: Name of the categorical parameter.

    Returns:
        A BayBE CategoricalParameter.
    """
    return CategoricalParameter(name, values=cat_list)


def convert_discrete_numerical_variable(
    num_list: List[float],
    name: str,
) -> NumericalDiscreteParameter:
    """
    Create a BayBE NumericalDiscreteParameter.

    Args:
        num_list: Discrete numeric values the parameter may take.
        name: Name of the numerical parameter.

    Returns:
        A BayBE NumericalDiscreteParameter.
    """
    return NumericalDiscreteParameter(name, values=num_list)


def convert_continuous_numerical_variable(
    bounds_tuple: Tuple[float, float],
    name: str,
) -> NumericalContinuousParameter:
    """
    Create a BayBE NumericalContinuousParameter.

    Args:
        bounds_tuple: Lower and upper bounds (min, max).
        name: Name of the numerical parameter.

    Returns:
        A BayBE NumericalContinuousParameter.
    """
    return NumericalContinuousParameter(name, bounds=bounds_tuple)


def convert_objective_variable(
    name: str,
    mode: str,
    bounds: List[float],
) -> NumericalTarget:
    """
    Create a normalized ramp objective target.

    Args:
        name: Objective name (must match measurement column).
        mode: Optimization mode ("min" or "max").
        bounds: Lower and upper cutoff values for normalization.

    Returns:
        A BayBE NumericalTarget.

    Raises:
        ValueError: If bounds are invalid.
    """
    descending = mode.lower() == "min"

    if bounds[0] >= bounds[1]:
        raise ValueError(f"Objective '{name}' has invalid bounds")

    return NumericalTarget.normalized_ramp(
        name,
        cutoffs=(bounds[0], bounds[1]),
        descending=descending,
    )


def convert_params(
    cat_var_dict: Dict[str, List[str]],
    sub_var_dict: Dict[str, Dict[str, str]],
    num_disc_var_dict: Dict[str, List[float]],
    num_cont_var_dict: Dict[str, Tuple[float, float]],
    obj_dict: Dict[str, Dict[str, Union[str, List[float]]]],
) -> Tuple[List, List[NumericalTarget]]:
    """
    Convert user-defined parameter and objective dictionaries into
    BayBE parameter and target objects.

    Args:
        cat_var_dict: Categorical variable definitions.
        sub_var_dict: Substance variable definitions.
        num_disc_var_dict: Discrete numerical variable definitions.
        num_cont_var_dict: Continuous numerical variable definitions.
        obj_dict: Objective definitions including mode and bounds.

    Returns:
        A tuple of (parameters, objectives).
    """
    parameters: List = []
    objectives: List[NumericalTarget] = []

    for name, values in cat_var_dict.items():
        parameters.append(convert_categorical_variable(values, name))

    for name, values in sub_var_dict.items():
        parameters.append(convert_substance_variable(values, name))

    for name, values in num_disc_var_dict.items():
        parameters.append(convert_discrete_numerical_variable(values, name))

    for name, bounds in num_cont_var_dict.items():
        parameters.append(convert_continuous_numerical_variable(bounds, name))

    for name, values in obj_dict.items():
        objectives.append(
            convert_objective_variable(
                name=name,
                mode=values["mode"],
                bounds=values["bounds"],
            )
        )

    return parameters, objectives


def create_campaign(
    categorical_variables_dict: Dict[str, List[str]],
    substance_variables_dict: Dict[str, Dict[str, str]],
    disc_numerical_variables_dict: Dict[str, List[float]],
    cont_numerical_variables_dict: Dict[str, Tuple[float, float]],
    objective_dict: Dict[str, Dict[str, Union[str, List[float]]]],
    weights: Optional[List[int]],
) -> str:
    """
    Create a BayBE campaign and serialize it to JSON.

    Args:
        categorical_variables_dict: Categorical variable definitions.
        substance_variables_dict: Substance variable definitions.
        disc_numerical_variables_dict: Discrete numerical variable definitions.
        cont_numerical_variables_dict: Continuous numerical variable definitions.
        objective_dict: Objective definitions.
        weights: Objective weights for multi-objective optimization.

    Returns:
        JSON-serialized BayBE campaign.
    """
    parameters, objectives = convert_params(
        categorical_variables_dict,
        substance_variables_dict,
        disc_numerical_variables_dict,
        cont_numerical_variables_dict,
        objective_dict,
    )

    searchspace = SearchSpace.from_product(parameters=parameters)

    if len(objectives) > 1:
        objective = DesirabilityObjective(targets=objectives, weights=weights)
    else:
        objective = SingleTargetObjective(target=objectives[0])

    campaign = Campaign(searchspace=searchspace, objective=objective)
    return campaign.to_json()


def recommend_reactions(
    campaign: str,
    df: Optional[pd.DataFrame],
    batch_size: int,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Generate a batch of recommended experiments.

    Args:
        campaign: JSON-serialized BayBE campaign.
        df: Optional DataFrame of previous measurements.
        batch_size: Number of experiments to recommend.

    Returns:
        A tuple of (recommendations DataFrame, updated campaign JSON).
    """
    if not campaign:
        st.error("Please upload valid file")
        return None, None

    campaign_recreate = Campaign.from_json(campaign)
    target_names = [t.name for t in campaign_recreate.targets]

    if df is not None:
        campaign_recreate.add_measurements(df)

    recommendations = campaign_recreate.recommend(batch_size=batch_size)

    for target in target_names:
        recommendations[target] = None

    return recommendations, campaign_recreate.to_json()


def create_categorical_fields(num_variables):
    variable_dict = {}
    for i in range(num_variables):
        with st.container(border=True, key=f"cat_var_{i}"):
            variable_name = st.text_input(f"Variable {i + 1} title:", placeholder = 'E.g. base treament')
            variable_values = st.text_input(f"Variable {i + 1} categories (comma-separated):", placeholder= "ground, unground")
            
        values = [value.strip() for value in variable_values.split(',')]
        variable_dict[variable_name] = values
    return variable_dict

def create_substance_fields(num_variables):
    variable_dict = {}
    for i in range(num_variables):
        with st.container(border=True, key=f"sub_var_{i}"):
            variable_name = st.text_input(f"Variable {i + 1} title:", placeholder = 'E.g. solvent')
            variable_values = st.text_input(f"Variable {i + 1} names (comma-separated):", placeholder= "methanol, ethanol, etc.")
            variable_smile_values = st.text_input(f"Variable {i + 1} SMILE strings (comma-separated):", placeholder= "CO, CCO, etc.")

        keys = [value.strip() for value in variable_values.split(',')]
        values = [value.strip() for value in variable_smile_values.split(',')]
        variable_dict[variable_name] = dict(zip(keys, values))
    return variable_dict

def create_discrete_numerical_fields(num_numerical_variables):
    variable_dict = {}
    for i in range(num_numerical_variables):
        with st.container(border=True, key=f"disc_num_var_{i}"):
            variable_name = st.text_input(f"Variable {i + 1} name:", placeholder = 'E.g. temperature')
            variable_values = st.text_input(f"Variable {i + 1} values (comma-separated):", placeholder= "40,60,80")
            
        values = [float(value.strip()) for value in variable_values.split(',')]
        variable_dict[variable_name] = values
    return variable_dict

def create_continuous_numerical_fields(num_numerical_variables):
    variable_dict = {}
    for i in range(num_numerical_variables):
        with st.container(border=True, key=f"cont_num_var_{i}"):
            variable_name = st.text_input(f"Variable {i + 1} name:", placeholder = 'E.g. equivalents')
            variable_lower_bound = st.number_input(f"Lower bound of continuous variable {i + 1}", value=0.0, format="%0.1f")
            variable_upper_bound = st.number_input(f"Upper bound of continuous variable {i + 1}", value=1.0, format="%0.1f")
            variable_dict[variable_name] = (variable_lower_bound, variable_upper_bound)
    return variable_dict

def create_objective_fields(num_objective_variables):
    objective_dict = {}
    for i in range(num_objective_variables):
        values = {}
        with st.container(border=True, key=f"obj_var_{i}"):
            variable_name = st.text_input(f"Objective {i + 1} name:", placeholder='Yield')
            variable_mode = st.selectbox(f"Objective {i + 1} mode:", options=["max", "min"])
            variable_lower_bound = st.number_input(f"Lower bound of objective {i + 1}", value=0)
            variable_upper_bound = st.number_input(f"Upper bound of objective {i + 1}", value=100)
        values["mode"] = variable_mode
        values["bounds"] = [variable_lower_bound,variable_upper_bound]
        objective_dict[variable_name] = values
    return objective_dict

def upload_file(key):
    uploaded_files = st.file_uploader("Choose a " + key + " file", key = key)
    
    if uploaded_files is not None and uploaded_files.name.split('.')[1] == 'json':
        stringio = StringIO(uploaded_files.getvalue().decode("utf-8"))
        data = stringio.read()
        return data
    
    if uploaded_files is not None and uploaded_files.name.split('.')[1] == 'csv':
        raw = uploaded_files.read().decode("utf-8", errors="replace")
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        df = pd.read_csv(StringIO(raw))
        return df

def recommend_input():
    past_recommendation = st.toggle('Include existing reaction data')
    if past_recommendation:
        df = upload_file(key='Reactions data CSV')        
        return df
    return None

def get_current_round(campaign):
    campaign_recreate = Campaign.from_json(campaign)
    info = campaign_recreate.measurements
    try:
        return info["BatchNr"].max()
    except KeyError:
        return 0

def plot_learning_curve(campaign,objective_dict):
    campaign_recreate = Campaign.from_json(campaign)
    info = campaign_recreate.measurements
    try:
        num_rounds = info["BatchNr"].max()
    except KeyError:
        st.warning("Insufficient rounds performed to plot optimisation curve")
        return None
    if num_rounds > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        for obj in objective_dict:
            values = objective_dict[obj]
            if values["mode"].lower() == 'max':
                data = info.groupby('BatchNr')[obj].max().reset_index()
            else:
                data = info.groupby('BatchNr')[obj].min().reset_index()
            ax.plot(data["BatchNr"], data[obj], marker='o', label=obj)
        ax.set_title('Best Objective Outcome(s) vs. Round Number')
        ax.set_xlabel('Round Number')
        ax.set_xticks(data["BatchNr"])
        ax.set_ylabel('Objective Variable Value')
        ax.legend()
        st.pyplot(fig, clear_figure=True)
        return None
    st.warning("Insufficient rounds performed to plot optimisation curve")
    return None

def show_SHAP(campaign):
    campaign_recreate = Campaign.from_json(campaign)
    data = campaign_recreate.measurements[[p.name for p in campaign_recreate.parameters]]
    model = lambda x: campaign_recreate.get_surrogate().posterior(x).mean

    explainer = shap.Explainer(model, data)
    shap_values = explainer(data)
    st_shap(shap.plots.bar(shap_values), width=600)

def main():
    
    st.image('./catsci-logo.svg', width=200)  # Adjust width as needed
    st.title("Bayesian Reaction Optimizer")

    with st.expander("User Guide"):
        st.markdown("""
        ## Overview
        
        The Bayesian Reaction Optimizer is an interactive tool for designing and optimizing chemical reactions using Bayesian optimization powered by BayBE.
        
        Instead of exploring reaction conditions by trial and error, the tool learns from previous experiments and recommends the most informative next reactions to run. It supports categorical variables (e.g. solvent, catalyst), numerical variables (e.g. temperature, equivalents), and structure-aware substance variables (via molecular descriptors). One or more experimental objectives—such as yield or selectivity—can be optimized simultaneously.
        
        The workflow is fully iterative: define your reaction space, run suggested experiments, upload the results, and generate improved recommendations in subsequent rounds.
        
        ---
        ## How to Use the Tool
        ### 1. Define the Reaction Space
        - Specify reaction parameters:
          - **Categorical variables** (e.g. base treatment, atmosphere)
          - **Substance variables** (e.g. solvent or ligand, defined by SMILES)
          - **Numerical variables** (either discrete *or* continuous)
        - Define one or more **objectives** (e.g. yield, conversion), including:
          - Optimization mode (`max` or `min`)
          - Expected lower and upper bounds
        > **Note:** Discrete and continuous numerical variables cannot be mixed in the same campaign.
        ---
        ### 2. Generate a Campaign
        - Click *Generate* to create the Bayesian optimization campaign.
        - Download the generated campaign JSON file — this represents the full reaction design space and optimization state.
        ---
        ### 3. Get Reaction Recommendations
        - Upload the latest campaign JSON.
        - (Optional) Upload a CSV containing results from previously run reactions.
        - Specify how many new reactions you want to run.
        - Click *Get recommendations* to receive a ranked list of suggested experiments.
        ---
        ### 4. Run Experiments and Iterate
        - Perform the recommended reactions in the lab.
        - Record results (including objective values and batch number).
        - Upload the updated results in the next round to refine future recommendations.
        ---
        ### 5. Analyze Progress (Optional)
        - Visualize learning curves showing objective improvement across rounds.
        - View feature importance (SHAP analysis) to understand which parameters most influence the model.
        ---
        ## Typical Workflow
        1. Define parameters and objectives  
        2. Generate campaign  
        3. Run suggested reactions  
        4. Upload results  
        5. Repeat until optimal conditions are found
        """)

    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
    if "campaign_generated" not in st.session_state:
        st.session_state.campaign_generated = False
    if "recommendations_made" not in st.session_state:
        st.session_state.recommendations_made = False
    
    st.divider()

    campaign_name = st.text_input("Enter a campaign name", value="", key="campaign_name")
    
    st.header("Outline Parameters and Objective(s)")
    
    with st.container(border=True, key="cat_vars"):
        st.subheader("Categorical Variables")
        
        num_categorical_variables = st.number_input("How many **categorical** variables do you have?", min_value=0, value=0, key = 'cat')
        categorical_variables_dict = create_categorical_fields(num_categorical_variables)

    with st.container(border=True, key="sub_vars"):
        st.subheader("Substance Variables")
    
        num_sub_variables = st.number_input("How many **substance-type categorical** variables do you have?", min_value=0, value=0, key = 'sub')
        substance_variables_dict = create_substance_fields(num_sub_variables)

    with st.container(border=True, key="disc_num_vars"):
        st.subheader("Discrete Numerical Variables")
            
        num_disc_numerical_variables = st.number_input("How many **discrete numerical** variables do you have?", min_value=0, value=0, key = 'num_disc')
        disc_numerical_variables_dict = create_discrete_numerical_fields(num_disc_numerical_variables)

    with st.container(border=True, key="cont_num_vars"):
        st.subheader("Continuous Numerical Variables")
    
        num_cont_numerical_variables = st.number_input("How many **continuous numerical** variables do you have?", min_value=0, value=0, key = 'num_cont')
    
        if (num_disc_numerical_variables > 0) and (num_cont_numerical_variables > 0):
            st.error("This tool does not support mixing discrete and continuous numerical variables - please use one type or the other exclusively.")
            st.stop()
        
        cont_numerical_variables_dict = create_continuous_numerical_fields(num_cont_numerical_variables)

    with st.container(border=True, key="objs"):
        st.subheader("Objectives")
        
        num_objectives = st.number_input("How many **objective** variables do you have", min_value= 1, value= 1, key = 'obj')
        objective_dict = create_objective_fields(num_objectives)
    
        if num_objectives > 1:
            objective_weights = st.text_input("Target Objective weights (comma-separated and should sum to 100):", placeholder= "50,50")
            weights = [int(value.strip()) for value in objective_weights.split(',')]
        
            if num_objectives != len(weights):
                st.warning("Please make sure there are the same number of objectives as objective weights.")
        else:
            weights = None

    st.divider()
    st.header("Create Reaction Space")

    if st.button("Generate"):
        with st.spinner('Processing...'):
            st.session_state.campaign_json = create_campaign(categorical_variables_dict, substance_variables_dict, 
                                            disc_numerical_variables_dict, cont_numerical_variables_dict, 
                                            objective_dict, weights)

            st.session_state.campaign_generated = True

    if st.session_state.campaign_generated:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{now}_{campaign_name}_initial.json'
        st.download_button(f"Download {filename}", st.session_state.campaign_json, file_name=filename)

    st.divider()
    st.header("Recommend Reactions")

    campaign_previous = upload_file(key='latest campaign JSON')
    
    batch_size = st.number_input("Number of reactions to suggest", min_value=1, value=5, key='batch')
    df = recommend_input()
    
    if st.button("Get recommendations"):
        st.session_state.reactions, st.session_state.new_campaign = recommend_reactions(campaign_previous, df, batch_size)
        if st.session_state.reactions is not None and st.session_state.new_campaign is not None:
            st.data_editor(st.session_state.reactions)
            st.session_state.recommendations_made = True

    if st.session_state.recommendations_made and "new_campaign" in st.session_state and "reactions" in st.session_state:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_round = get_current_round(st.session_state.new_campaign)
        json_filename = f'{now}_{campaign_name}_round{current_round}.json'
        reactions_filename = f'{now}_{campaign_name}_round{current_round}_reactions.csv'
        
        st.download_button(f"Download round {current_round} JSON file",
                           st.session_state.new_campaign,
                           file_name=json_filename)
        
        st.download_button(f"Download recommended round {current_round} reactions",
                           st.session_state.reactions.to_csv(index=False, lineterminator='\r\n').encode('utf-8'),
                           file_name=reactions_filename,
                           mime='text/csv')

        try:
            plot_learning_curve(st.session_state.new_campaign, objective_dict)
        except ValueError:
            None

        if st.toggle("Show SHAP values"):
            show_SHAP(st.session_state.new_campaign)
            
if __name__ == "__main__":
    main()
