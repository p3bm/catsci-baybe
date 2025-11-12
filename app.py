import streamlit as st
from utils import create_campaign, recommend_reactions
from baybe import Campaign
from baybe.recommenders import (
    RandomRecommender,
    FPSRecommender,
    KMeansClusteringRecommender,
    BotorchRecommender,
    TwoPhaseMetaRecommender
)
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
import matplotlib.pyplot as plt

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
        values = {}
        with st.container(border=True, key=f"obj_var_{i}"):
            variable_name = st.text_input(f"Objective {i + 1} name:", placeholder='Yield')
            variable_mode = st.selectbox(f"Objective {i + 1} mode:", options=["max", "min"])
            variable_lower_bound = st.number_input(f"Lower bound of objective {i + 1}", value=0)
            variable_upper_bound = st.number_input(f"Upper bound of objective {i + 1}", value=100)
        values["mode"] = variable_mode
        values["bounds"] = (variable_lower_bound,variable_upper_bound)
        objective_dict[variable_name] = values
    return objective_dict

def upload_file(key):
    uploaded_files = st.file_uploader("Choose a " + key + " file", key = key)
    
    if uploaded_files is not None and uploaded_files.name.split('.')[1] == 'json':
        stringio = StringIO(uploaded_files.getvalue().decode("utf-8"))
        data = stringio.read()
        return data
    
    if uploaded_files is not None and uploaded_files.name.split('.')[1] == 'csv':
        df = pd.read_csv(uploaded_files)
        return df

def recommend_input():
    past_recommendation = st.toggle('Include existing reaction data')
    if past_recommendation:
        df = upload_file(key='Reactions data CSV')
        return df
        
def plot_learning_curve(campaign,objective_dict):
    campaign_recreate = Campaign.from_json(campaign)
    info = campaign_recreate.measurements
    num_batches = max(info["BatchNr"].values.to_list())
    if num_batches > 1:
        data_to_plot = {}
        fig, ax = plt.subplots(figsize=(8, 5))
        for obj, values in objective_dict:
            if values["mode"].lower() == 'max':
                data_to_plot[obj] = info.groupby('BatchNr')[obj].max().reset_index()
            else:
                data_to_plot[obj] = info.groupby('BatchNr')[obj].min().reset_index()
        
            ax.plot(max_yield_per_batch['BatchNr'], max_yield_per_batch['Yield'], marker='o', label=obj)
        ax.set_title('Learning Curve: Max Yield per Batch')
        ax.set_xlabel('Round Number')
        ax.set_ylabel('Max Yield')
        ax.legend()
        st.pyplot(fig)
        return None
    st.warning("Insufficient rounds performed to plot optimisation curve")
    return None

def main():
    
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

    with st.container(border=True, key="recomms"):
        st.subheader("Campaign Settings")
        
        initial_recommender = st.selectbox(
            'Select a stratgey to use for the initial recommendations:',
            ('Random', 'Farthest Point Sampling', 'KMEANS Clustering'))
        
        second_recommender = st.selectbox(
            "Select a surrogate model type to recommend new reactions when reaction data becomes available:",
            ("Gaussian Process", "Random Forest", "NGBoost", "Bayesian Linear"))

        acq_functions_single = {
            "Expected Improvement" : "EI",
            "quasi Expected Improvement" : "qEI",
            "quasi Noisy Expected Improvement" : "qNEI",
            "Log Expected Improvement" : "LogEI",
            "quasi Log Expected Improvement" : "qLogEI",
            "Upper Confidence Bound" : "UCB",
            "quasi Upper Confidence Bound" : "qUCB"
        }

        acq_functions_multi = {
            "quasi Expected Hypervolume Improvement" : "qEHVI",
            "quasi Noisy Expected Hypervolume Improvement" : "qNEHVI",
            "quasi Log Expected Hypervolume Improvement" : "qLEHVI"
        }
        
        if num_objectives == 1:
            ACQ_FUNCTION = st.selectbox("Select an acquisition function:", options=[key for key in acq_functions_single], index=1, key="single_obj_acq_func")
        else:
            ACQ_FUNCTION = st.selectbox("Select an acquisition function:", options=[key for key in acq_functions_multi], index=1, key="multi_obj_acq_func")

    strategy = TwoPhaseMetaRecommender(
                    initial_recommender=strategy_functions_first[initial_recommender],
                    recommender=BotorchRecommender(
                        surrogate_model= strategy_functions_second[second_recommender], acquisition_function=acq_functions[ACQ_FUNCTION]
                        )
                    )
    
    st.divider()
    st.header("Create Reaction Space")

    if st.button("Generate"):
        with st.spinner('Processing...'):
            try:
                st.session_state.campaign_json = create_campaign(categorical_variables_dict, substance_variables_dict, 
                                                disc_numerical_variables_dict, cont_numerical_variables_dict, 
                                                objective_dict, strategy, weights)

                st.session_state.campaign_generated = True
            except ValueError:
                st.error("You may have forgotten to enter a name and optimisation mode for the objective variable(s). Check this and try again.")
                st.stop()

    if st.session_state.campaign_generated:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Download campaign JSON", st.session_state.campaign_json, file_name= f'{now}_{campaign_name}.json')

    st.divider()
    st.header("Recommend Reactions")

    campaign_previous = upload_file(key='latest campaign JSON')
    
    batch_reactions = st.number_input("Number of reactions to suggest", min_value= 1, value= 1, key = 'batch')
    df = recommend_input()
    
    if st.button("Get recommendations"):
        try:
            st.session_state.reactions, st.session_state.new_campaign = recommend_reactions(campaign_previous, df, batch_reactions)
        except NotEnoughPointsLeftError:
            st.error("The number of recommendations entered exceeds the number of parameter combinations left to explore!")
            st.stop()
        if st.session_state.reactions is not None and st.session_state.new_campaign is not None:
            st.data_editor(st.session_state.reactions)
            st.session_state.recommendations_made = True

    if st.session_state.recommendations_made and "new_campaign" in st.session_state and "reactions" in st.session_state:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Download JSON file", st.session_state.new_campaign, file_name= f"{now}_{campaign_name}.json")
        st.download_button("Download recommended reactions", st.session_state.reactions.to_csv().encode('utf-8'), file_name= f'{now}_{campaign_name}_reactions.csv', mime= 'text/csv')

        plot_learning_curve(st.session_state.new_campaign)
        
if __name__ == "__main__":
    main()
