from baybe import Campaign
from baybe.objectives import SingleTargetObjective, DesirabilityObjective
from baybe.parameters import NumericalDiscreteParameter, NumericalContinuousParameter, CategoricalParameter, SubstanceParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
import streamlit as st
import pandas as pd
import json
import streamlit as st
from io import StringIO
from datetime import datetime
import matplotlib.pyplot as plt
import shap
from streamlit_shap import st_shap

def convert_substance_variable(sub_dict, name):
    """_summary_

    Args:
        cat_dict (_type_): _description_
        name (_type_): _description_
    """
    return SubstanceParameter(name, data=sub_dict, encoding="MORDRED")


def convert_categorical_variable(cat_list, name):
    """_summary_

    Args:
        cat_list (_type_): _description_
        name (_type_): _description_
    """
    return CategoricalParameter(name, values=cat_list)


def convert_discrete_numerical_variable(num_list, name):
    """_summary_

    Args:
        num_list (_type_): _description_
    """
    return NumericalDiscreteParameter(name, values=num_list)


def convert_continuous_numerical_variable(bounds_tuple, name):
    """_summary_

    Args:
        bounds_tuple (_type_): _description_
    """
    return NumericalContinuousParameter(name, bounds=bounds_tuple)


def convert_objective_variable(name, mode, bounds):
    """_summary_

    Args:
        name (_type_): _description_
        mode (_type_): _description_
        bounds (_type_): _description_
    """

    min_mode = False
    
    if mode.lower() == "min":
        min_mode = True

    target = NumericalTarget.normalized_ramp(name, cutoffs=(bounds[0], bounds[1]), descending=min_mode)
    
    return target


def convert_params(cat_var_dict, sub_var_dict, num_disc_var_dict, num_cont_var_dict, obj_dict) -> list:
    parameters = []
    objectives = []
    """_summary_

    Args:
        cat_var_dict (_type_): _description_
        sub_var_dict (_type_): _description_
        num_disc_var_dict (_type_): _description_
        num_cont_var_dict (_type_): _description_
        obj_dict (_type_): _description_

    Returns:
        list: _description_
    """
    for cat in cat_var_dict:
        
        variable = convert_categorical_variable(cat_list=cat_var_dict[cat], name=cat)
        parameters.append(variable)

    for sub in sub_var_dict:
        
        variable = convert_substance_variable(sub_dict=sub_var_dict[sub], name=sub)
        parameters.append(variable)

    for num in num_disc_var_dict:
        
        variable = convert_discrete_numerical_variable(num_list=num_disc_var_dict[num], name=num)
        parameters.append(variable)

    for num in num_cont_var_dict:
        
        variable = convert_continuous_numerical_variable(bounds_tuple=num_cont_var_dict[num], name=num)
        parameters.append(variable)
    
    for obj in obj_dict:
        values = obj_dict[obj]
        target = convert_objective_variable(name=obj, mode=values["mode"].upper(), bounds=values["bounds"])
        objectives.append(target)

    return parameters, objectives


def create_campaign(categorical_variables_dict, substance_variables_dict, 
                    disc_numerical_variables_dict, cont_numerical_variables_dict, 
                    objective_dict, weights):
    """_summary_

    Args:
        categorical_variables_dict (_type_): _description_
        substance_variables_dict (_type_): _description_
        disc_numerical_variables_dict (_type_): _description_
        cont_numerical_variables_dict (_type_): _description_
        objective_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    parameters, objectives = convert_params(categorical_variables_dict, substance_variables_dict, 
                                            disc_numerical_variables_dict, cont_numerical_variables_dict, 
                                            objective_dict)
    searchspace = SearchSpace.from_product(parameters=parameters)
    if len(objectives) > 1:
        objective = DesirabilityObjective(targets=objectives, weights=weights)
    else:
        objective = SingleTargetObjective(target=objectives[0])

    campaign = Campaign(searchspace=searchspace, objective=objective)

    return campaign.to_json()


def recommend_reactions(campaign, df, batch_size)-> pd.DataFrame:
    recommendations = None
    campaign_recreate = None
    if campaign:
        campaign_recreate = Campaign.from_json(campaign)
        campaign_json = json.loads(campaign)
        # Retrieve the objective
        objective = campaign_json.get("objective", {})

        # Check for "target" and "targets" in the objective
        if "target" in objective:
            # If "target" is a dictionary, convert it to a list containing that dictionary
            if isinstance(objective["target"], dict):
                target_list = [objective["target"]]
            else:
                st.error("The target value is not a dictionary.")
        elif "targets" in objective:
            # If "targets" is a list, use it directly
            if isinstance(objective["targets"], list):
                target_list = objective["targets"]
            else:
                st.error("Error in objective")
        
        target_names = [target["name"] for target in target_list]

        if df is not None:
            campaign_recreate.add_measurements(df)
        try:
            st.write(f"Acquisition function: {campaign_recreate.get_acquisition_function()}")
            st.write(f"Surrogate function: {campaign_recreate.get_surrogate()}")
        except:
            None
        recommendations = campaign_recreate.recommend(batch_size=batch_size)
        for target_column in target_names:
            recommendations[target_column] = None
        
    else:
        st.error("Please upload valid file")

    return recommendations, campaign_recreate.to_json() if campaign_recreate else None

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
            
        values = [value.strip() for value in variable_values.split(',')]
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
            ### **Overview**
            This tool helps you **design and optimize chemical reactions** using Bayesian Optimization via **BayBE**.  
            It suggests which experiments to run next to improve one or more objectives — for example, yield or selectivity.
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
            objective_weights = st.text_input("Target Objective weights (comma-separated):", placeholder= "50,50")
            vals = objective_weights.split(',')
            weights = [int(value.strip()) for value in vals if value.strip().isdigit()]
        
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
