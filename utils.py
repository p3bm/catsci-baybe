from baybe import Campaign
from baybe.objectives import SingleTargetObjective, DesirabilityObjective
from baybe.parameters import NumericalDiscreteParameter, NumericalContinuousParameter, CategoricalParameter, SubstanceParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
import streamlit as st
import pandas as pd
import json

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

    st.write(NumericalTarget.TRANSFORMATIONS)
    target = NumericalTarget(name=name, minimize=min_mode)
    target.clamp(min=bounds[0], max=bounds[1])
    
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
                    objective_dict, strategy, weights):
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

    campaign = Campaign(searchspace=searchspace, objective=objective, recommender=strategy)

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

        if df is None:
            recommendations = campaign_recreate.recommend(batch_size=batch_size)
            for target_column in target_names:
                recommendations[target_column] = None
        else:
            campaign_recreate.add_measurements(df)
            recommendations = campaign_recreate.recommend(batch_size=batch_size)
            for target_column in target_names:
                recommendations[target_column] = None
        
    else:
        st.error("Please upload valid file")

    return recommendations, campaign_recreate.to_json() if campaign_recreate else None
