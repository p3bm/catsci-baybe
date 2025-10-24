import streamlit as st
from utils import convert_params, create_campaign, recommend_reactions
from baybe.searchspace import SearchSpace
from baybe import Campaign
from baybe.objective import Objective
from baybe.targets import NumericalTarget
from baybe.recommenders import RandomRecommender, FPSRecommender, KMeansClusteringRecommender, SequentialGreedyRecommender, TwoPhaseMetaRecommender
# from baybe.strategies import TwoPhaseStrategy
from baybe.surrogates import (
    BayesianLinearSurrogate,
    GaussianProcessSurrogate,
    NGBoostSurrogate,
    RandomForestSurrogate,
)
import json
from io import StringIO
import pandas as pd
from datetime import datetime


ACQ_FUNCTION = "qEI"
ALLOW_REPEATED_RECOMMENDATIONS = True
ALLOW_RECOMMENDING_ALREADY_MEASURED = True


# Map the function names to the actual functions using a dictionary
strategy_functions_first = {
    'Random': RandomRecommender(),
    'Farthest Point Sampling': FPSRecommender(),
    'KMEANS Clustering': KMeansClusteringRecommender(),
}

strategy_functions_second = {
    "Gaussian Process" : GaussianProcessSurrogate(),
    "Random Forest": RandomForestSurrogate(),
    "NGBoost": NGBoostSurrogate(),
    "Bayesian Linear": BayesianLinearSurrogate(),
}

def create_categorical_fields(num_variables):
    variable_dict = {}
    for i in range(num_variables):
        variable_name = st.text_input(f"Variable {i + 1} title:", placeholder = 'E.g. base treament')
        variable_values = st.text_input(f"Variable {i + 1} categories (comma-separated):", placeholder= "ground, unground")

        values = [value.strip() for value in variable_values.split(',')]

        variable_dict[variable_name] = values
    # st.write(variable_dict)
    return variable_dict

def create_substance_fields(num_variables):
    variable_dict = {}
    for i in range(num_variables):
        variable_name = st.text_input(f"Variable {i + 1} title:", placeholder = 'E.g. solvent')
        variable_values = st.text_input(f"Variable {i + 1} names (comma-separated):", placeholder= "methanol, ethanol, etc.")
        variable_smile_values = st.text_input(f"Variable {i + 1} SMILE strings (comma-separated):", placeholder= "CO, CCO, etc.")

        keys = [value.strip() for value in variable_values.split(',')]
        values = [value.strip() for value in variable_smile_values.split(',')]
    
        variable_dict[variable_name] = dict(zip(keys, values))
    # st.write(variable_dict)
    return variable_dict

def create_discrete_numerical_fields(num_numerical_variables):
    variable_dict = {}
    for i in range(num_numerical_variables):
        variable_name = st.text_input(f"Variable {i + 1} name:", placeholder = 'E.g. temperature')
        variable_values = st.text_input(f"Variable {i + 1} values (comma-separated):", placeholder= "40,60,80")
        values = [value.strip() for value in variable_values.split(',')]

        variable_dict[variable_name] = values
    return variable_dict

def create_continuous_numerical_fields(num_numerical_variables):
    variable_dict = {}
    for i in range(num_numerical_variables):
        variable_name = st.text_input(f"Variable {i + 1} name:", placeholder = 'E.g. equivalents')
        variable_values = st.text_input(f"Variable {i + 1} lower and upper bounds (comma-separated):", placeholder= "0.8,2.0")
        bounds = [value.strip() for value in variable_values.split(',')]
        try:
            bounds = (bounds[0], bounds[1])
        except IndexError:
            bounds = (None, None)
        variable_dict[variable_name] = bounds

    for key in variable_dict:
        if len(variable_dict[key]) > 2:
            st.error("The continuous categorical variable requires only lower and upper bound values.")
            
    return variable_dict

def create_objective_fields(num_objective_variables):
    objective_dict = {}
    for i in range(num_objective_variables):
        variable_name = st.text_input(f"Objective {i + 1} name:", placeholder = 'Yield')
        variable_values = st.text_input(f"Objective {i + 1} mode (comma-separated):", placeholder= "max")
        # weights = st.number_input()
        values = [value.strip() for value in variable_values.split(',')]

        objective_dict[variable_name] = values
    # st.write(objective_dict)
    return objective_dict


def upload_file(key):
    uploaded_files = st.file_uploader("Choose a " + key + " file", key = key)
    
    if uploaded_files is not None and uploaded_files.name.split('.')[1] == 'json':
        stringio = StringIO(uploaded_files.getvalue().decode("utf-8"))
        data = stringio.read()
        # st.json(data)
        return data
    
    if uploaded_files is not None and uploaded_files.name.split('.')[1] == 'csv':
        df = pd.read_csv(uploaded_files)
        return df
        

def recommend_input():
    past_recommendation = st.toggle('Include existing reaction data')
    if past_recommendation:
        df = upload_file(key='Reactions data CSV')
        return df

def show_stats(campaign,recommendations):
    campaign_recreate = Campaign.from_json(campaign)
    campaign_json = json.loads(campaign)
    st.write(campaign_json.measurements)
    st.table(campaign_json.posterior_stats(recommendations))
    return None

def main():
    #st.set_page_config(page_title=None, page_icon="🧪", layout="wide")
    
    st.image('./catsci-logo.svg', width=200)  # Adjust width as needed
    st.title("Bayesian Reaction Optimizer")

    with st.expander("User Guide"):
        st.markdown("""
            ### **Overview**
            This tool helps you **design and optimize chemical reactions** using Bayesian Optimization via **BayBE**.  
            It suggests which experiments to run next to improve one or more objectives — for example, yield or selectivity.
            
            ---
            
            ### **1️⃣ Define Your Variables**
            Describe all the factors that define the reaction space:
            
            - **Categorical variables** – qualitative choices such as whether a base is ground or unground.  
            - **Substance variables** – chemical components that can be represented by SMILES strings (e.g. solvents, additives).  
            - **Discrete numerical variables** – numeric values with fixed options.  
            - **Continuous numerical variables** – a range of numeric values.
            
            ---
            
            ### **2️⃣ Define Your Objective(s)**
            Specify what to optimize, such as yield, selectivity, or ee.
            The mode can be set to `max` or `min` to maimise or minimise the chosen objective as required.
            If multiple objectives are used, the weighting of the importance of each objective can be specified.
            Make sure there are the same number of weights as objectives!
            Note: the objectives currently expect values within the bounds 0-100.
            
            ---
            
            ### **3️⃣ Choose Recommender Strategy**
            Pick two recommenders that guide the optimization process:
            
            - **Initial recommender** *(used before any data exists)*:
              - `Random` – Picks random conditions (pure exploration).  
              - `Farthest Point Sampling` – Picks conditions far apart to cover the space evenly.  
              - `KMEANS Clustering` – Picks representative, evenly distributed conditions.
            
            - **Surrogate recommender** *(used after data is available)*:
              - `Gaussian Process` – Smooth model, good for continuous variables.  
              - `Random Forest` – Robust and fast, works well with mixed variable types.  
              - `NGBoost` – Flexible model for complex relationships.  
              - `Bayesian Linear` – Simple, effective for small datasets.
            
            ---
            
            ### **4️⃣ Generate the Campaign**
            Once inputs are ready:
            1. Click **“Generate”** to create your optimization campaign.  
            2. Download the resulting **`*_campaign.json`** file to save your setup.
            
            ---
            
            ### **5️⃣ Get New Recommendations**
            1. Upload your latest **`*_campaign.json`** file.
            2. Set the **number of new experiments to suggest**.  
            3. Click **“Get recommendations.”**  
            4. View, edit, and download:
               - Suggested experiments as **`reactions.csv`**  
               - Updated campaign file as **`*_campaign.json`**
            
            ---

            ### **6️⃣ Add Experimental Data (Optional)**
            If you’ve already run experiments:
            1. Toggle **“Include existing reaction data.”**  
            2. Upload your results as a `.csv` file.  
               - Columns must match your variable and objective names.  
            3. The optimizer will use these results to improve its model.
            
            ---
            
            ### **7️⃣ Iterative Optimization**
            After each batch of experiments:
            1. Add new results to your data file.  
            2. Upload the updated `.csv` and `.json`.  
            3. Request new recommendations.  
            Repeat until your objectives stop improving.

            ---
            
            ### **Known Issues**
            - Does not yet support specifying a tolerance values on experimental variables.
            - Does not support the mixed use of discrete and continuous numerical variables (stick to one type or the other).
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
        
        num_objectives = st.number_input("How many **objective** variables do you have", min_value= 0, value= 0, key = 'obj')
        objective_dict = create_objective_fields(num_objectives)
    
        if num_objectives > 1:
            objective_weights = st.text_input("Target Objective weights (comma-separated):", placeholder= "50,50")
            vals = objective_weights.split(',')
            weights = [int(value.strip()) for value in vals if value.strip().isdigit()]
        
            if num_objectives != len(weights):
                st.warning("Please make sure there are the same number of objectives as objective weights.")
        else:
            weights = None

    with st.container(border=True, key="recomms"):
        st.subheader("Select Recommenders")
        
        initial_recommender = st.selectbox(
            'Select a stratgey to use for the initial recommendations:',
            ('Random', 'Farthest Point Sampling', 'KMEANS Clustering'))
        
        second_recommender = st.selectbox(
            "Select a surrogate model type to recommend new reactions when reaction data becomes available:",
            ("Gaussian Process", "Random Forest", "NGBoost", "Bayesian Linear"))

    strategy = TwoPhaseMetaRecommender(
                    initial_recommender= strategy_functions_first[initial_recommender],
                    recommender=SequentialGreedyRecommender(
                        surrogate_model= strategy_functions_second[second_recommender], acquisition_function=ACQ_FUNCTION
                    ),)
    
    st.divider()
    st.header("Create Reaction Space")

    if st.button("Generate"):
        with st.spinner('Processing...'):
            
            campaign_json = create_campaign(categorical_variables_dict, substance_variables_dict, 
                                            disc_numerical_variables_dict, cont_numerical_variables_dict, 
                                            objective_dict, strategy, weights)

            st.session_state.campaign_generated = True

    if st.session_state.campaign_generated:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Download campaign JSON", campaign_json, file_name= f'{now}_campaign.json')

    st.divider()
    st.header("Recommend Reactions")

    campaign_previous = upload_file(key='Campaign JSON')
    
    batch_reactions = st.number_input("Number of reactions to suggest", min_value= 1, value= 1, key = 'batch')
    df = recommend_input()
    
    if st.button("Get recommendations"):
        reactions, new_campaign = recommend_reactions(campaign_previous, df, batch_reactions)
        if reactions is not None and new_campaign is not None:
            st.data_editor(reactions)
            st.session_state.recommendations_made = True

    if st.session_state.recommendations_made:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Download JSON file", new_campaign, file_name= f"{now}_campaign.json")
        st.download_button("Download recommended reactions", reactions.to_csv().encode('utf-8'), file_name= 'reactions.csv', mime= 'text/csv')
        show_stats(new_campaign,reactions)

if __name__ == "__main__":
    main()
