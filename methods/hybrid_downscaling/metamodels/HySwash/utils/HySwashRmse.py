import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.datamining.pca import PCA
from bluemath_tk.interpolation.rbf import RBF


# Divide the cases in train and test group
def shuffle_cases(cases, n_fold):
    """
    Shuffle cases for cross-validation.
    """
    shuffled_cases = cases.copy()  # Dont modify the original n_cases
    np.random.seed(2)
    np.random.shuffle(shuffled_cases)
    split_arrays = np.array_split(shuffled_cases, n_fold)
    return split_arrays


def rmse(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    # Apply the mask
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    # Compute RMSE only on valid pairs
    rmse = np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2))

    return rmse


def print_rmse(n_cases, postprocessed_output, mda, vars):
    import warnings

    warnings.filterwarnings("ignore", message="The NumPy module was reloaded*")
    rmse_df_list = []
    rmse_Ru2 = []
    folds = 5
    n_cases_list = np.arange(50, n_cases, 50)
    rmse_dict = {}
    for var in vars:
        rmse_dict[var] = []

    for cases_i in n_cases_list:
        # Select train and test data
        cases = np.arange(0, cases_i)
        cases_folds = shuffle_cases(cases, 5)

        train_cases = np.concatenate(cases_folds[:4])
        test_cases = cases_folds[4]

        postprocessed_output = postprocessed_output.copy(deep=True)

        train_data = postprocessed_output.isel(case_num=train_cases)
        test_data = postprocessed_output.isel(case_num=test_cases)

        pca = {}
        rbf = {}

        # Calculate RMSE for each variable
        for var in vars:
            if var == "Ru2":
                # RunUp
                Ru2 = train_data[["Ru2"]].Ru2.values
                # # Convert Ru2 numpy array to DataFrame
                rbf_Ru = RBF()
                rbf_Ru.logger.disabled = True
                Ru2_df = pd.DataFrame(Ru2, columns=["Ru2"])
                rbf_Ru.fit(
                    subset_data=mda.centroids.iloc[train_data["case_num"].values, :],
                    target_data=Ru2_df,
                )
                reconstructed_Ru2 = rbf_Ru.predict(dataset=test_data_df)
                rmse_Ru2s = rmse(reconstructed_Ru2.Ru2.values, test_data["Ru2"].values)
                rmse_dict["Ru2"].append(rmse_Ru2s)

            else:
                # Create RBF from train data

                # Get PCA components from the Hs field of the train data
                pca[var] = PCA()
                pca[var].logger.disabled = True
                _pcs_trained = pca[var].fit_transform(
                    data=train_data,
                    vars_to_stack=[var],
                    coords_to_stack=["Xp"],
                    pca_dim_for_rows="case_num",
                    value_to_replace_nans={var: 0.0},
                )

                # Get the relationship between the preditands (Hs, Hs_Lo, Hv, Nv and Dv)
                # and the predictors (PC1, PC2, PC3, PC4, ....)
                rbf[var] = RBF()
                rbf[var].logger.disabled = True
                rbf[var].fit(
                    subset_data=mda.centroids.iloc[train_data["case_num"].values, :],
                    target_data=pca[var].pcs_df,
                    num_workers=20,
                )

                # Get predicted Hrms from Test data
                # Spatial Reconstruction: Test data
                test_data_df = mda.centroids.iloc[test_data["case_num"].values, :]
                pcs_test_predicted = rbf[var].predict(
                    dataset=test_data_df,
                )
                pcs_test_predicted_ds = xr.Dataset(
                    {"PCs": (["case_num", "n_component"], pcs_test_predicted.values)},
                    coords={
                        "case_num": (["case_num"], test_data["case_num"].values),
                        "n_component": (
                            ["n_component"],
                            range(1, len(pcs_test_predicted.columns) + 1),
                        ),
                    },
                )

                reconstructed_test_data = pca[var].inverse_transform(
                    PCs=pcs_test_predicted_ds
                )
                rmse_values = []
                for i in reconstructed_test_data.case_num.values:
                    rmse_var = rmse(
                        reconstructed_test_data.sel(case_num=i)[var].values,
                        test_data.sel(case_num=i)[var].values,
                    )
                    rmse_values.append(rmse_var)
                mean_rmse = np.mean(rmse_values)
                rmse_dict[var].append(mean_rmse)

    plt.figure(figsize=(8, 5))
    for var in vars:
        plt.plot(n_cases_list, rmse_dict[var], marker="o", label=var)
    plt.xlabel("Number of Cases")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Number of Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
