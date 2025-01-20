import pandas as pd
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues
from sklearn import datasets

reference_data = datasets.load_iris(as_frame=True).frame
reference_data = reference_data.rename(
    columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "target",
    }
)

current_data = pd.read_csv("prediction_database.csv")
current_data = current_data.drop(columns=["time"])
current_data = current_data.rename(
    columns={
        current_data.columns[0]: "sepal_length",
        current_data.columns[1]: "sepal_width",
        current_data.columns[2]: "petal_length",
        current_data.columns[3]: "petal_width",
        current_data.columns[4]: "target",
    }
)


report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("reports/report_drift.html")

report = Report(metrics=[DataQualityPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("reports/report_quality.html")

report = Report(metrics=[TargetDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("reports/report_target_drift.html")

data_test = TestSuite(tests=[TestNumberOfMissingValues()])
data_test.run(reference_data=reference_data, current_data=current_data)
result = data_test.as_dict()
print(result)
print("All tests passed: ", result["summary"]["all_passed"])
