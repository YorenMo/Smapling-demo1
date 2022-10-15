# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:21:09 2022

@author: Yoren Mo
"""

import altair as alt

def get_chart_shap_R(data):
    
    hover = alt.selection_single(
        fields=["num"],
        nearest=True,
        on="mouseover",
        empty="none",
    )
    

    lines = (
        alt.Chart(data, title="Remove datas")
        .mark_line()
        .encode(
            x= alt.X("num",scale=alt.Scale(domain=[0, 600])),
            y= alt.Y("value", scale=alt.Scale(domain=[0, 12])),
            color=alt.Color("method", scale=alt.
                    Scale(domain=['H','L'], range=['red','grey']))
            # strokeDash="method",
        )
    )
    
    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)
       
    # Draw a rule at the location of the selection
    
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="num",
            y="value",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("num", title="training datas"),
                alt.Tooltip("value", title="value(MAE)"),
            ],
        )
        .add_selection(hover)
    
    )
    
    return (lines + points + tooltips).interactive()

def get_chart_shap_A(data):
    
    hover = alt.selection_single(
        fields=["num"],
        nearest=True,
        on="mouseover",
        empty="none",
    )
    

    lines = (
        alt.Chart(data, title="Add datas")
        .mark_line()
        .encode(
            x= alt.X("num",scale=alt.Scale(domain=[0, 600])),
            y= alt.Y("value", scale=alt.Scale(domain=[0, 12])),
            color=alt.Color("method", scale=alt.
                    Scale(domain=['H','L'], range=['red','grey']))
            # strokeDash="method",
        )
    )
    
    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)
       
    # Draw a rule at the location of the selection
    
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="num",
            y="value",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("num", title="training datas"),
                alt.Tooltip("value", title="value(MAE)"),
            ],
        )
        .add_selection(hover)
    
    )
    
    return (lines + points + tooltips).interactive()