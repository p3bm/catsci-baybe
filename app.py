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
import shap
from streamlit_shap import st_shap


# Safe debugging helpers for BayBE - put this near the top of your app (after imports)
import types
import inspect
import numpy as np
import torch
import traceback

# ---------------------
# Helper: restore original if accidentally overwritten
# ---------------------
def safe_restore(module, name, backup_name):
    """If backup exists on the module, restore the original attribute."""
    if hasattr(module, backup_name):
        setattr(module, name, getattr(module, backup_name))
        delattr(module, backup_name)

# ---------------------
# 1) Safely patch BayesianRecommender._setup_botorch_acqf
#    - Do NOT allow double-patching (guard with attribute)
#    - Keep a true copy of the original under a private backup name
# ---------------------
from baybe.recommenders.pure.bayesian.base import BayesianRecommender

if not hasattr(BayesianRecommender, "_dbg_original__setup_botorch_acqf"):
    # store the original method (unbound function)
    BayesianRecommender._dbg_original__setup_botorch_acqf = BayesianRecommender._setup_botorch_acqf

def _dbg_setup_botorch_acqf(self, searchspace, objective, measurements, pending):
    # Print only a few lines (Streamlit or stdout)
    try:
        print("\n=== DEBUG: _setup_botorch_acqf ===")
        # measurements.targets might be a special object; convert cautiously
        try:
            mt = np.asarray(measurements.targets)
            print("measurements.targets np.shape:", mt.shape)
            if mt.size <= 40:
                print("measurements.targets sample:", mt)
            else:
                print("measurements.targets sample (first 10):", mt[:10])
        except Exception as e:
            print("Could not show measurements.targets:", e)

        # show the objective descriptor (type / repr)
        try:
            print("objective type:", type(objective))
            try:
                # Show objective internal description if available
                if hasattr(objective, "to_dict"):
                    print("objective.to_dict():", objective.to_dict())
                else:
                    print("objective repr:", repr(objective))
            except Exception:
                pass
        except Exception:
            pass

    except Exception as dbg_exc:
        print("DEBUG helper error in _dbg_setup_botorch_acqf:", dbg_exc)
        traceback.print_exc()

    # finally call the original implementation (the true original, not the wrapper)
    return BayesianRecommender._dbg_original__setup_botorch_acqf(self, searchspace, objective, measurements, pending)

# install wrapper
BayesianRecommender._setup_botorch_acqf = _dbg_setup_botorch_acqf

# ---------------------
# 2) Safely inspect Objective.to_botorch outputs without breaking its contract
#    - We won't replace the returned object with a function.
#    - We'll wrap Objective.to_botorch to call original and then *inspect* the returned object,
#      but then return the *same* object unchanged so that attrs validators are satisfied.
# ---------------------
from baybe.objectives.base import Objective

if not hasattr(Objective, "_dbg_original__to_botorch"):
    Objective._dbg_original__to_botorch = Objective.to_botorch

def _dbg_to_botorch(self):
    # call original to obtain the real BoTorch objective object (or None)
    obj = Objective._dbg_original__to_botorch(self)

    try:
        print("\n=== DEBUG: Objective.to_botorch() was called ===")
        print("Objective instance type:", type(self))
        # print short summary of what was returned (type and attributes)
        print("returned type:", type(obj))
        # If it's a BoTorch object with `__call__` or `.forward`, try a dry call with a tiny tensor
        if obj is not None:
            # build a tiny test tensor - do not try large tensors
            # need to derive number of outputs expected: we can try (1, n_targets)
            # infer n_targets from self if possible
            try:
                # best-effort: if the objective has attribute 'n_targets' or similar
                n_targets = None
                if hasattr(self, "n_targets"):
                    n_targets = int(getattr(self, "n_targets"))
                elif hasattr(self, "targets"):
                    n_targets = int(len(getattr(self, "targets")))
                # fallback to 2 if unknown (safe small tensor)
                if n_targets is None:
                    n_targets = 2

                test_x = torch.zeros((1, n_targets))
                print("Attempting dry-call obj(test_x) with test_x.shape:", tuple(test_x.shape))
                try:
                    out = obj(test_x)
                    # try to print shape if possible
                    try:
                        out_np = out.detach().cpu().numpy()
                        print("obj(test_x) -> numpy shape:", out_np.shape)
                        if out_np.size <= 40:
                            print("obj(test_x) -> sample:", out_np)
                        else:
                            print("obj(test_x) -> sample(first 10):", out_np.flatten()[:10])
                    except Exception:
                        try:
                            out_np2 = np.asarray(out)
                            print("obj(test_x) -> np.asarray shape:", out_np2.shape)
                        except Exception as e2:
                            print("Could not convert obj output to numpy:", e2)
                except Exception as call_exc:
                    print("Calling obj(test_x) raised exception (may be expected):")
                    traceback.print_exc()
            except Exception as e:
                print("Failed preparing/doing dry-call on objective:", e)
    except Exception as e_top:
        print("DEBUG Objective.to_botorch wrapper hit exception:", e_top)
        traceback.print_exc()

    # return the original object unchanged so BayBE's validators are happy
    return obj

# install wrapper
Objective.to_botorch = _dbg_to_botorch

# ---------------------
# 3) Clean up any previously-created accidental globals / wrappers
#    If you previously put other debug names on these classes, remove them.
# ---------------------
# (No-op here intentionally; if you created other debug names earlier, remove them manually.)



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
            variable_lower_bound = st.number_input(f"Lower bound of continuous variable {i + 1}", value=0)
            variable_upper_bound = st.number_input(f"Upper bound of continuous variable {i + 1}", value=1)
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
        df = pd.read_csv(uploaded_files)
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
            acq_function_name = st.selectbox("Select an acquisition function:", options=[key for key in acq_functions_single], index=1, key="single_obj_acq_func")
            acq_function = acq_functions_single[acq_function_name]
        else:
            acq_function_name = st.selectbox("Select an acquisition function:", options=[key for key in acq_functions_multi], index=1, key="multi_obj_acq_func")
            acq_function = acq_functions_multi[acq_function_name]
    
    strategy = TwoPhaseMetaRecommender(
                    initial_recommender=strategy_functions_first[initial_recommender],
                    recommender=BotorchRecommender(
                        surrogate_model= strategy_functions_second[second_recommender], acquisition_function=acq_function
                        )
                    )
    
    st.divider()
    st.header("Create Reaction Space")

    if st.button("Generate"):
        with st.spinner('Processing...'):
            st.session_state.campaign_json = create_campaign(categorical_variables_dict, substance_variables_dict, 
                                            disc_numerical_variables_dict, cont_numerical_variables_dict, 
                                            objective_dict, strategy, weights)

            st.session_state.campaign_generated = True

    if st.session_state.campaign_generated:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Download campaign JSON", st.session_state.campaign_json, file_name= f'{now}_{campaign_name}_initial.json')

    st.divider()
    st.header("Recommend Reactions")

    campaign_previous = upload_file(key='latest campaign JSON')
    
    batch_size = st.number_input("Number of reactions to suggest", min_value= 1, value= 1, key = 'batch')
    df = recommend_input()
    
    if st.button("Get recommendations"):
        st.session_state.reactions, st.session_state.new_campaign = recommend_reactions(campaign_previous, df, batch_size)
        if st.session_state.reactions is not None and st.session_state.new_campaign is not None:
            st.data_editor(st.session_state.reactions)
            st.session_state.recommendations_made = True

    if st.session_state.recommendations_made and "new_campaign" in st.session_state and "reactions" in st.session_state:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        st.download_button("Download JSON file",
                           st.session_state.new_campaign,
                           file_name= f"{now}_{campaign_name}_round{get_current_round(st.session_state.new_campaign)}.json")
        
        st.download_button("Download recommended reactions",
                           st.session_state.reactions.to_csv(index=False).encode('utf-8'),
                           file_name= f'{now}_{campaign_name}_round{get_current_round(st.session_state.new_campaign)}_reactions.csv',
                           mime= 'text/csv')

        try:
            plot_learning_curve(st.session_state.new_campaign, objective_dict)
        except ValueError:
            None

        if st.toggle("Show SHAP values"):
            show_SHAP(st.session_state.new_campaign)
            
if __name__ == "__main__":
    main()
