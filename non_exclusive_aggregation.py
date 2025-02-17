import pandas as pd
import numpy as np
from typing import Any
from collections.abc import Callable
from itertools import product

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
    "n": 1
})


# Writing a genaralized function
def parse_mapping(mapping, x: pd.Series):
    if isinstance(mapping, str) and mapping == "__ALL__":
        return x.unique()
    else:
        return mapping


def all_combos_non_exclusive_agg(df: pd.DataFrame, 
                                 groupcols: list[str], 
                                 category_mappings: dict[str, dict[Any, Any]], 
                                 valuecols: list[str] = [], 
                                 aggargs: None | dict[str, Any] | Callable | str | list  = None,
                                 totalcodes: None | dict[str, str] = None, 
                                 keep_empty: bool = False, 
                                 grand_total: bool = True): # not implemented yet
    """Generate all aggregation levels for a set of columns in a dataframe, for non-exclusive categories.

    Creates aggregations over all combinations of categorical variables specified in `groupcols`
    and applies aggregation functions on `valuecols`. Allows for inclusion of grand totals
    and customized fill values for missing groups, similar to "proc means" in SAS. categories are 
    defined by a dictionary of mappings in `category_mappings`, and can be non-exclusive.

    Args:
        df: DataFrame to aggregate.
        groupcols: List of columns to group by.
        category_mappings: Dictionary of dictionaries, where each key is a column name and each value is a dictionary of mappings. 
            '__ALL__' can be used to indicate 'all values' in a column, and is used for totals.
        valuecols: List of columns to apply aggregation functions on. Defaults to None, in which case all numeric columns are used.
        aggargs: Dictionary or function specifying aggregation for each column in `valuecols`. If None, defaults to 'sum' for each column in `valuecols`.
        totalcodes: Dictionary specifying values to use as labels representing totals in each column.
        keep_empty: If True, preserves empty groups in the output.
        grand_total: Dictionary or string to indicate a grand total row. If a dictionary, the values are applied in each corresponding `groupcols`.

    Returns:
        DataFrame with all aggregation levels for the specified columns.

    Examples:
        >>> # Define the categorical bins based on the metadata
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
                "n": 1
            })

        >>> category_mappings = {
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
                },
                "syss_student": {
                    "01": ["01", "02"], 
                    "02": ["03", "04"],
                    "03": ["02"],
                    "04": ["04"],
                },
                "Kjonn": {
                    "Menn": ["1"],
                    "Kvinner": ["2"],
                }
            }

    >>> totalcodes = {
                "Alder": "Total",
                "syss_student": "Total",
                "Kjonn": "Begge"
        }

    >>> all_combos_non_exclusive_agg(synthetic_data, 
                                     groupcols = [],
                                     category_mappings=category_mappings,
                                     totalcodes=totalcodes,
                                     valuecols = ["n"],
                                     aggargs={"n": "sum"},
                                     grand_total=True)
    """                                
    all_cols: list[str] = groupcols + valuecols + list(category_mappings.keys())
    df = df.copy()[all_cols]

    if totalcodes: 
        for var, code in totalcodes.items():
            category_mappings[var][code] = "__ALL__"

    pivot_vars: list[str] = list(category_mappings.keys())
    pivot_names: dict[str, list[str]] = {}
    all_pivot_names: list[str] = []
    

    # fill in default for the rest, used for `grand_total`
    if not totalcodes:
        totalcodes = {}
    for var in pivot_vars + groupcols:
        if var not in totalcodes.keys():
            totalcodes[var] = "Total"


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
  
    tbl: pd.DataFrame = df.groupby(groupcols + all_pivot_names).agg(aggargs).reset_index()

    id_vars: set[str] = set(groupcols + all_pivot_names + valuecols)
    for var in category_mappings.keys():
        id_vars = id_vars - set(pivot_names[var])
        tbl = tbl.melt(id_vars = list(id_vars), 
                       value_vars = pivot_names[var],
                       var_name = "__variable__", value_name = var)
        tbl = tbl.loc[tbl[var] != "__NA__", :].drop(labels=["__variable__"], axis=1)
        id_vars = id_vars.union([var])

    tbl = tbl.groupby(groupcols + pivot_vars).agg(aggargs).reset_index()

    if grand_total:
        total_df = df.copy()
        grouping: list[str] = groupcols + pivot_vars

        for var in grouping:
            total_df[var] = totalcodes[var]
        tbl = pd.concat((tbl, total_df.groupby(grouping).agg(aggargs).reset_index()))
        tbl = tbl.reset_index(drop=True)

    if keep_empty:
        grouping = groupcols + pivot_vars
        all_combos = list(product(*[tbl[v].unique() for v in grouping]))
        all_combos_df = pd.DataFrame(np.array(all_combos), columns=grouping)

        tbl = pd.merge(all_combos_df, tbl, on=grouping, how="left")

    return tbl



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
    },
    "syss_student": {
        "01": ["01", "02"], 
        "02": ["03", "04"],
        "03": ["02"],
        "04": ["04"],
    },
    "Kjonn": {
        "Menn": ["1"],
        "Kvinner": ["2"],
    }
}


totalcodes = {
        "Alder": "Total",
        "syss_student": "Total",
        "Kjonn": "Begge"
}


tbl = all_combos_non_exclusive_agg(synthetic_data, 
                             groupcols = [],
                             category_mappings=category_mappings,
                             totalcodes=totalcodes,
                             valuecols = ["n"],
                             aggargs={"n": "sum"},
                             grand_total=True)
tbl.loc[tbl["n"].isna()]
#> Out[39]: 
#> Empty DataFrame
#> Columns: [Alder, syss_student, Kjonn, n]
#> Index: []

tbl = all_combos_non_exclusive_agg(synthetic_data, 
                             groupcols = [],
                             category_mappings=category_mappings,
                             totalcodes=totalcodes,
                             valuecols = ["n"],
                             aggargs={"n": "sum"},
                             grand_total=True, 
                             keep_empty=True)
tbl.loc[tbl["n"].isna()]
#> Out[40]: 
#>     Alder syss_student    Kjonn   n
#> 7   15-21           03  Kvinner NaN
#> 55  22-30           04  Kvinner NaN
#> 68  25-34           03     Menn NaN
