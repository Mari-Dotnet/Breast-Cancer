from flask import Flask,request,render_template,app
import pandas as pd
import numpy as np
import pickle
import os

application=Flask(__name__)
app=application

from src.pipeline.prediction_pipeline import predic_pipeline,Customerdata


@app.route("/")
def helloworld():
    return render_template('index.html')

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    result=""
    if (request.method=="POST"):
        data=Customerdata(
        meanradius=float(request.form.get('meanradius')),
        meantexture=float(request.form.get('meantexture')),
        meansmoothness=float(request.form.get('meansmoothness')),
        meancompactness=float(request.form.get('meancompactness')),
        meansymmetry=float(request.form.get('meansymmetry')),
        meanfractaldimension=float(request.form.get('meanfractaldimension')),
        radiuserror=float(request.form.get('radiuserror')),
        textureerror=float(request.form.get('textureerror')),
        smoothnesserror=float(request.form.get('smoothnesserror')),
        compactnesserror=float(request.form.get('compactnesserror')),
        concavityerror=float(request.form.get('concavityerror')),
        concavepointserror=float(request.form.get('concavepointserror')),
        symmetryerror=float(request.form.get('symmetryerror')),
        fractaldimensionerror=float(request.form.get('fractaldimensionerror')),
        worstsmoothness=float(request.form.get('worstsmoothness')),
        worstsymmetry=float(request.form.get('worstsymmetry')),
        worstfractaldimension=float(request.form.get('worstfractaldimension'))
        )

        final_data=data.get_dataas_dataframe()
        predictdata=predic_pipeline()
        predict=predictdata.Predictvalues(final_data)
        if predict[0]==1:
            result="Breast cancer"
        else:
            result="No-Breast cancer"
        return render_template('single_prediction.html',result=result)
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
