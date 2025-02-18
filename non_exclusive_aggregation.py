import pandas as pd
import numpy as np
from typing import Any
from collections.abc import Callable
from itertools import product


def parse_mapping(mapping: list[Any] | str, x: pd.Series) -> list[Any] | str:
    if isinstance(mapping, str) and mapping == "__ALL__":
        return list(x.unique())
    else:
        return mapping


def all_combos_non_exclusive_agg(df: pd.DataFrame, 
                                 groupcols: list[str] = [], 
                                 category_mappings: dict[str, dict[str, list[Any] | str]] = {}, 
                                 valuecols: list[str] = [], 
                                 aggargs: None | dict[str, Any] | Callable | str | list  = None,
                                 totalcodes: None | dict[str, str] = None, 
                                 keep_empty: bool = False, 
                                 grand_total: bool = True) -> pd.DataFrame:
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

    for col in groupcols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        items: list[Any] = list(df[col].unique())
        keys: list[str] = [str(x) for x in items]
        values: list[list[Any]] = [[x] for x in items]
        mapping: dict[str, list[Any] | str] = dict(zip(keys, values))

        if col not in category_mappings.keys():
            category_mappings[col] = mapping


    all_cols: list[str] = valuecols + list(category_mappings.keys())
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
    for var in pivot_vars:
        if var not in totalcodes.keys():
            totalcodes[var] = "Total"


    for var in category_mappings.keys():
        ncat: int = len(category_mappings[var])

        pivot_names[var] = ["__" + var + "__" + str(i) for i in range(ncat)]
        all_pivot_names: list[str]  = all_pivot_names + pivot_names[var]

        x: pd.Series = df[var].astype("str")
        x[:] = "__NA__" # using pd.NA does not work as intended with .melt()

        for i, pairmap in enumerate(category_mappings[var].items()):
            y: pd.Series = x.copy()
            newval: str = pairmap[0]
            oldvals: list[Any] | str | Any = parse_mapping(pairmap[1], x=df[var])
            pivot_name: str = pivot_names[var][i]

            y.loc[df[var].isin(oldvals)] = newval
            df[pivot_name] = y
  
    tbl: pd.DataFrame = df.groupby(all_pivot_names).agg(aggargs).reset_index()

    id_vars: set[str] = set(all_pivot_names + valuecols)
    for var in category_mappings.keys():
        id_vars = id_vars - set(pivot_names[var])
        tbl = tbl.melt(id_vars = list(id_vars), 
                       value_vars = pivot_names[var],
                       var_name = "__variable__", value_name = var)
        tbl = tbl.loc[tbl[var] != "__NA__", :].drop(labels=["__variable__"], axis=1)
        id_vars = id_vars.union([var])

    tbl = tbl.groupby(pivot_vars).agg(aggargs).reset_index()

    if grand_total:
        total_df: pd.DataFrame = df.copy()

        for var in pivot_vars:
            total_df[var] = totalcodes[var]
        tbl = pd.concat((tbl, total_df.groupby(pivot_vars).agg(aggargs).reset_index()))
        tbl = tbl.reset_index(drop=True)

    if keep_empty:
        all_combos: list[Any] = list(product(*[tbl[v].unique() for v in pivot_vars]))
        all_combos_df: pd.DataFrame = pd.DataFrame(np.array(all_combos), columns=pivot_vars)

        tbl = pd.merge(all_combos_df, tbl, on=pivot_vars, how="left")

    return tbl


if __name__ == "__main__":
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
            "01-02": ["01", "02"], 
            "03-04": ["03", "04"],
            "02": ["02"],
            "04": ["04"],
        },
    #    "Kjonn": {
    #        "Menn": ["1"],
    #        "Kvinner": ["2"],
    #    }
    }
    
    
    totalcodes = {
            "Alder": "Total",
            "syss_student": "Total",
            "Kjonn": "Begge"
    }
    
    
    tbl = all_combos_non_exclusive_agg(synthetic_data, 
                                 groupcols = ["Kjonn"],
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
    synthetic_data.loc[(synthetic_data["Alder"] >= 15) &
                       (synthetic_data["Alder"] <= 21) &
                       synthetic_data["syss_student"].isin(["01", "02"]), :]
    
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


    test_mappings = {
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
            "Total": range(0, 100)
        },
        "syss_student": {
            "01-02": ["01", "02"], 
            "03-04": ["03", "04"],
            "02": ["02"],
            "04": ["04"],
            "Total": ["01", "02", "03", "04"]
        },
        "Kjonn": {
            "Menn": ["1"],
            "Kvinner": ["2"],
            "Begge": ["1", "2"]
        }
    }

    def print_indented(x, w = 4):
        pad = " " * w
        print(pad + str(x).replace("\n", "\n" + pad))

    def print_thin_sep(w=57):
        print("-" * w)

    def print_thick_sep(w=57):
        print("=" * w)

    tbl = all_combos_non_exclusive_agg(synthetic_data,
                                       groupcols = [],
                                       category_mappings=test_mappings,
                                       valuecols = ["n"],
                                       aggargs={"n": "sum"})

    cat_alder = test_mappings["Alder"]
    cat_student = test_mappings["syss_student"]
    cat_kjonn = test_mappings["Kjonn"]

    for alder in cat_alder:
        for student in cat_student:
            for kjonn in cat_kjonn:
                query = synthetic_data.loc[(synthetic_data["Alder"].isin(cat_alder[alder])) &
                                           (synthetic_data["syss_student"].isin(cat_student[student])) &
                                           (synthetic_data["Kjonn"].isin(cat_kjonn[kjonn])), :]
                n_observed = query.shape[0]
                n_predicted = tbl.loc[(tbl["Alder"] == alder) &
                                      (tbl["syss_student"] == student) &
                                      (tbl["Kjonn"] == kjonn), "n"].values
                n_predicted = 0 if len(n_predicted) == 0 else n_predicted[0]

                print_thick_sep()
                print(f"\tQuery: {alder}, {student}, {kjonn}")
                print(f"\tObserved: {n_observed}, Predicted: {n_predicted}, Diff: {n_observed - n_predicted}")
                print_thick_sep()
                print("Table:")
                print_indented(tbl.loc[(tbl["Alder"] == alder) &
                              (tbl["syss_student"] == student) &
                              (tbl["Kjonn"] == kjonn), :])
                print_thin_sep()
                print("Data:")
                print_indented(query)
                print_thin_sep()
                print("\n\n")
                assert n_observed == n_predicted





