import pandas as pd
import numpy as np
from typing import Any

# Define the categorical bins based on the metadata
gender_bins = {"1": "Menn", "2": "Kvinner"}

# Generate synthetic data
np.random.seed(42)
num_samples = 100

synthetic_data = pd.DataFrame({
    "Tid": np.random.choice(["2021", "2022", "2023"], num_samples),
    "UtdanningOppl": np.random.choice(list(range(1,19)), num_samples),
    "Kjonn": np.random.choice(list(gender_bins.keys()), num_samples),
    "Alder": np.random.randint(15, 67, num_samples),  # Ages between 15 and 66
    "syss_student": np.random.choice(["01", "02", "03", "04"], num_samples),
    "Deltakere": 1
})

# Generate a random duplication factor for each row (simulating different numbers of people in groups)
np.random.seed(42)

def to_range(x: str):
    start, stop = x.split("-")
    return (int(start), int(stop))

alder_cat_1_str = ["15-24", "25-34", "35-44", "45-54", "55-66"]
alder_cat_2_str = ["15-21", "22-30", "31-40", "41-50", "51-66"]
alder_cat_3_str = ["15-30", "31-45", "46-66"]

def categorize_age(x: pd.Series, cat: list[str]):
    y = x.astype("str")
    for age_range in cat:
        start, stop = to_range(age_range)
        y.loc[(x >= start) & (x <= stop)] = age_range

    return y

syss_stud_cat_1 = {"01": ["01", "02"], "02": ["03", "04"]}
syss_stud_cat_2 = {"03": ["02"], "04": ["04"]}

def categorize_nominal(x: pd.Series, cat: dict):
    y = x.astype("str")
    y[:] = ""
    for key, value in cat.items():
        y.loc[x.isin(value)] = key

    return y

synthetic_data["age_1"] = categorize_age(synthetic_data["Alder"], alder_cat_1_str)
synthetic_data["age_2"] = categorize_age(synthetic_data["Alder"], alder_cat_2_str)
synthetic_data["age_3"] = categorize_age(synthetic_data["Alder"], alder_cat_3_str)
synthetic_data["age_total"] = "Total"

synthetic_data["syss_student_1"] = categorize_nominal(synthetic_data["syss_student"], syss_stud_cat_1)
synthetic_data["syss_student_2"] = categorize_nominal(synthetic_data["syss_student"], syss_stud_cat_2)
synthetic_data["syss_student_total"] = "Total"
synthetic_data["n"] = 1

tbl = (
    synthetic_data.groupby(["Kjonn", 
                            "age_1", "age_2", "age_3", "age_total",
                            "syss_student_1", "syss_student_2", "syss_student_total",
                            ]).agg({"n": "sum"}).reset_index()
    .melt(id_vars = ["Kjonn", "syss_student_1", "syss_student_2", "syss_student_total", "n"], 
          value_vars = ["age_1", "age_2", "age_3", "age_total"],
          var_name = "age_group", value_name = "alder")
    .melt(id_vars = ["Kjonn", "n", "age_group", "alder"],
          value_vars = ["syss_student_1", "syss_student_2", "syss_student_total"],
          var_name = "syss_student_group", value_name = "syss_student")
    .drop(labels=["age_group", "syss_student_group"], axis=1)
    .groupby(["Kjonn", "alder", "syss_student"]).agg({"n": "sum"}).reset_index()
)

# testing:
# 7 rows
synthetic_data.loc[(synthetic_data["Alder"] >= 15) &
                   (synthetic_data["Alder"] <= 24) &
                   synthetic_data["syss_student"].isin(["01", "02"]), :]
tbl.loc[(tbl["alder"] == "15-24") & (tbl["syss_student"] == "01"), :]


# Writing a genaralized function
def parse_mapping(mapping, x: pd.Series):
    if isinstance(mapping, str) and mapping == "__ALL__":
        return x.unique()
    else:
        return mapping


def all_combos_non_exclusive_agg(df, groupcols, 
                                 category_mappings: dict[str, dict[Any, Any]], 
                                 valuecols=None, aggargs=None, 
                                 fillna_dict=None, keep_empty=False, grand_total=True): # not implemented yet
    df = df.copy()[groupcols + valuecols + list(category_mappings.keys())]

    pivot_vars: list[str] = list(category_mappings.keys())
    pivot_names: dict[str, list[str]] = {}
    all_pivot_names: list[str] = []

    for var in category_mappings.keys():
        ncat = len(category_mappings[var])

        pivot_names[var] = ["__" + var + "__" + str(i) for i in range(ncat)]
        all_pivot_names  = all_pivot_names + pivot_names[var]

        x = df[var].astype("str")
        x[:] = "__NA__" # using pd.NA does not work as intended with .melt()

        for i, mapping in enumerate(category_mappings[var].items()):
            y = x.copy()
            newval = mapping[0]
            oldvals = parse_mapping(mapping[1], x=df[var])
            pivot_name = pivot_names[var][i]

            y.loc[df[var].isin(oldvals)] = newval
            df[pivot_name] = y
  
    tbl = df.groupby(groupcols + all_pivot_names).agg(aggargs).reset_index()

    id_vars: set[str] = set(groupcols + all_pivot_names + valuecols)
    for var in category_mappings.keys():
        id_vars = id_vars - set(pivot_names[var])
        tbl = tbl.melt(id_vars = id_vars, 
                       value_vars = pivot_names[var],
                       var_name = "__variable__", value_name = var)
        tbl = tbl.loc[tbl[var] != "__NA__", :].drop(labels=["__variable__"], axis=1)
        id_vars = id_vars.union([var])

    return tbl.groupby(groupcols + pivot_vars).agg(aggargs).reset_index()


category_mappings = {
    "Alder": {
        "15-24": range(15, 25),
        "25-34": range(25, 35),
        "35-44": range(35, 45),
        "45-54": range(45, 55),
        "55-66": range(55, 67),
        "15-21": range(15, 22),
        "22-30": range(22, 31),
        "31-40": range(31, 41),
        "41-50": range(41, 51),
        "51-66": range(51, 67),
        "15-30": range(15, 31),
        "31-45": range(31, 46),
        "46-66": range(46, 67),
        # "Total": range(15, 67)
        "Total": "__ALL__"
    },
    "syss_student": {
        "01": ["01", "02"], 
        "02": ["03", "04"],
        "03": ["02"],
        "04": ["04"],
        # "Total": ["01", "02", "03", "04"]
        "Total": "__ALL__"
    },
    "Kjonn": {
        "Menn": ["1"],
        "Kvinner": ["2"],
        "Begge": ["1", "2"]
    }
}


tbl = all_combos_non_exclusive_agg(synthetic_data, 
                             groupcols = [],
                             category_mappings=category_mappings,
                             valuecols = ["n"],
                             aggargs={"n": "sum"})

tbl.loc[(tbl["Alder"] == "15-24") & (tbl["syss_student"] == "01"), :]
# 7 rows
synthetic_data.loc[(synthetic_data["Alder"] >= 15) &
                   (synthetic_data["Alder"] <= 24) &
                   synthetic_data["syss_student"].isin(["01", "02"]), :]
#>      Tid  UtdanningOppl Kjonn  Alder syss_student  Deltakere age_total syss_student_total  n
#> 24  2021              1     1     22           02          1     Total              Total  1
#> 46  2022             11     2     20           01          1     Total              Total  1
#> 62  2022              3     1     17           01          1     Total              Total  1
#> 67  2022              5     1     17           02          1     Total              Total  1
#> 76  2022              7     1     16           01          1     Total              Total  1
#> 79  2022             17     2     16           01          1     Total              Total  1
#> 84  2021              2     2     23           02          1     Total              Total  1
