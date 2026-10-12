
import json
import urllib.parse
import urllib.request
import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. PROJECT SETTINGS
# ==================================================

LATITUDE = 12.9716
LONGITUDE = 77.5946

START_YEAR = 2000
END_YEAR = 2025

DATA_FILE = "climate_data.csv"
GRAPH_FILE = "climate_graph.png"


# ==================================================
# 2. DOWNLOAD BENGALURU CLIMATE DATA
# ==================================================

def download_climate_data():

    print("Connecting to NASA POWER...")
    print("Downloading Bengaluru temperature data...")

    parameters = {
        "parameters": "T2M",
        "community": "RE",
        "longitude": LONGITUDE,
        "latitude": LATITUDE,
        "start": START_YEAR,
        "end": END_YEAR,
        "format": "JSON"
    }

    base_url = (
        "https://power.larc.nasa.gov/"
        "api/temporal/monthly/point?"
    )

    url = base_url + urllib.parse.urlencode(parameters)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BengaluruClimateProject/1.0"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    monthly_data = data["properties"]["parameter"]["T2M"]

    records = []

    for date, temperature in monthly_data.items():

        # NASA monthly data may include an annual summary
        # whose month field is 13. Skip it.
        year = int(date[:4])
        month = int(date[4:6])

        if month < 1 or month > 12:
            continue

        if temperature is None or temperature <= -900:
            continue

        records.append({
            "Year": year,
            "Month": month,
            "Temperature_C": float(temperature)
        })

    if not records:
        raise ValueError("No valid monthly temperature data found.")

    monthly_df = pd.DataFrame(records)

    # Calculate annual averages only for years with
    # all 12 valid monthly observations.
    annual_records = []

    for year, group in monthly_df.groupby("Year"):

        if group["Month"].nunique() != 12:
            continue

        annual_temperature = group["Temperature_C"].mean()

        annual_records.append({
            "Year": int(year),
            "Annual_Mean_Temperature_C": annual_temperature
        })

    df = pd.DataFrame(annual_records)

    if len(df) < 5:
        raise ValueError(
            "Not enough complete years of data were returned."
        )

    df = df.sort_values("Year").reset_index(drop=True)

    df.to_csv(DATA_FILE, index=False)

    print("\nData downloaded successfully!")
    print("Saved file:", DATA_FILE)
    print("\nFirst five rows:")
    print(df.head())

    print("\nYears available:", df["Year"].min(),
          "to", df["Year"].max())

    return df


# ==================================================
# 3. SOLVE LEAST SQUARES USING SVD
# ==================================================

def solve_using_svd(A, y):

    # Singular Value Decomposition:
    # A = U * Sigma * V_transpose
    U, singular_values, Vt = np.linalg.svd(
        A, full_matrices=False
    )

    tolerance = (
        np.finfo(float).eps
        * max(A.shape)
        * singular_values[0]
    )

    inverse_values = np.zeros_like(singular_values)

    valid = singular_values > tolerance
    inverse_values[valid] = 1 / singular_values[valid]

    # Least-squares solution:
    # beta = V * Sigma_pseudoinverse * U_transpose * y
    beta = Vt.T @ (inverse_values * (U.T @ y))

    return beta


# ==================================================
# 4. BUILD LINEAR AND QUADRATIC MODELS
# ==================================================

def fit_models(df):

    years = df["Year"].to_numpy(dtype=float)

    temperatures = df[
        "Annual_Mean_Temperature_C"
    ].to_numpy(dtype=float)

    # Center and scale years for numerical stability.
    year_mean = years.mean()
    year_scale = years.std()

    x = (years - year_mean) / year_scale

    # Linear model: T = b0 + b1*x
    A_linear = np.column_stack([
        np.ones(len(x)),
        x
    ])

    # Quadratic model: T = b0 + b1*x + b2*x^2
    A_quadratic = np.column_stack([
        np.ones(len(x)),
        x,
        x**2
    ])

    beta_linear = solve_using_svd(
        A_linear, temperatures
    )

    beta_quadratic = solve_using_svd(
        A_quadratic, temperatures
    )

    # Verify the linear solution using NumPy's
    # independent least-squares solver.
    beta_check, *_ = np.linalg.lstsq(
        A_linear, temperatures, rcond=None
    )

    print("\nSVD VERIFICATION")
    print("----------------")
    print(
        "Maximum coefficient difference:",
        np.max(np.abs(beta_linear - beta_check))
    )

    return (
        years,
        temperatures,
        year_mean,
        year_scale,
        beta_linear,
        beta_quadratic
    )


# ==================================================
# 5. PREDICT TEMPERATURE
# ==================================================

def predict_temperature(
    target_year,
    year_mean,
    year_scale,
    beta_linear,
    beta_quadratic
):

    x = (target_year - year_mean) / year_scale

    linear = beta_linear[0] + beta_linear[1] * x

    quadratic = (
        beta_quadratic[0]
        + beta_quadratic[1] * x
        + beta_quadratic[2] * x**2
    )

    return linear, quadratic


# ==================================================
# 6. CALCULATE RMSE
# ==================================================

def calculate_rmse(actual, predicted):

    return np.sqrt(np.mean((actual - predicted) ** 2))


# ==================================================
# 7. INTERPOLATION AND EXTRAPOLATION
# ==================================================

def estimate_year(
    target_year,
    first_year,
    last_year,
    year_mean,
    year_scale,
    beta_linear,
    beta_quadratic
):

    linear, quadratic = predict_temperature(
        target_year,
        year_mean,
        year_scale,
        beta_linear,
        beta_quadratic
    )

    if first_year < target_year < last_year:
        method = "INTERPOLATION"
    elif target_year < first_year or target_year > last_year:
        method = "EXTRAPOLATION"
    else:
        method = "ESTIMATE AT OBSERVED RANGE BOUNDARY"

    print(f"\n{method}: {target_year}")
    print(f"Linear estimate:    {linear:.3f} °C")
    print(f"Quadratic estimate: {quadratic:.3f} °C")

    return linear, quadratic


# ==================================================
# 8. DRAW AND SAVE THE GRAPH
# ==================================================

def plot_results(
    years,
    temperatures,
    year_mean,
    year_scale,
    beta_linear,
    beta_quadratic
):

    plot_years = np.arange(
        int(years.min()),
        2051
    )

    linear_values = []
    quadratic_values = []

    for year in plot_years:

        linear, quadratic = predict_temperature(
            year,
            year_mean,
            year_scale,
            beta_linear,
            beta_quadratic
        )

        linear_values.append(linear)
        quadratic_values.append(quadratic)

    plt.figure(figsize=(12, 7))

    plt.scatter(
        years,
        temperatures,
        label="NASA POWER annual averages"
    )

    plt.plot(
        plot_years,
        linear_values,
        label="Linear least-squares model"
    )

    plt.plot(
        plot_years,
        quadratic_values,
        label="Quadratic least-squares model"
    )

    plt.axvline(
        years.max(),
        linestyle="--",
        label="Last observed year"
    )

    plt.xlabel("Year")
    plt.ylabel("Annual mean temperature (°C)")
    plt.title(
        "Bengaluru Temperature Analysis Using SVD"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(GRAPH_FILE, dpi=300)
    plt.show()

    print("\nGraph saved as:", GRAPH_FILE)


# ==================================================
# 9. MAIN PROGRAM
# ==================================================

def main():

    try:

        # Download data.
        df = download_climate_data()

        # Fit mathematical models.
        (
            years,
            temperatures,
            year_mean,
            year_scale,
            beta_linear,
            beta_quadratic
        ) = fit_models(df)

        # Calculate fitted values for observed years.
        x = (years - year_mean) / year_scale

        linear_fitted = (
            beta_linear[0] + beta_linear[1] * x
        )

        quadratic_fitted = (
            beta_quadratic[0]
            + beta_quadratic[1] * x
            + beta_quadratic[2] * x**2
        )

        # Compare the models using RMSE.
        linear_rmse = calculate_rmse(
            temperatures, linear_fitted
        )

        quadratic_rmse = calculate_rmse(
            temperatures, quadratic_fitted
        )

        print("\nMODEL COMPARISON")
        print("----------------")
        print(f"Linear RMSE:    {linear_rmse:.4f} °C")
        print(f"Quadratic RMSE: {quadratic_rmse:.4f} °C")

        if linear_rmse < quadratic_rmse:
            print("Lower training RMSE: Linear model")
        elif quadratic_rmse < linear_rmse:
            print("Lower training RMSE: Quadratic model")
        else:
            print("Both models have equal training RMSE.")

        # Demonstrate interpolation at an interior year
        # by temporarily removing that year's observation.
        middle_index = len(years) // 2
        missing_year = int(years[middle_index])

        keep = years != missing_year

        interpolation_value = np.interp(
            missing_year,
            years[keep],
            temperatures[keep]
        )

        print("\nINTERPOLATION DEMONSTRATION")
        print("---------------------------")
        print("Year treated as missing:", missing_year)
        print(
            f"Interpolated value: {interpolation_value:.3f} °C"
        )
        print(
            "Observed value:     "
            f"{temperatures[middle_index]:.3f} °C"
        )

        # Estimate selected future years.
        print("\nFUTURE EXTRAPOLATION ESTIMATES")
        print("------------------------------")

        for year in [2030, 2040, 2050]:

            estimate_year(
                year,
                years.min(),
                years.max(),
                year_mean,
                year_scale,
                beta_linear,
                beta_quadratic
            )

        # Save and display graph.
        plot_results(
            years,
            temperatures,
            year_mean,
            year_scale,
            beta_linear,
            beta_quadratic
        )

        print("\nPROJECT COMPLETED SUCCESSFULLY!")

    except Exception as error:

        print("\nPROJECT ERROR")
        print("-------------")
        print(type(error).__name__ + ":", error)
        print(
            "\nCheck your internet connection, "
            "NASA POWER availability, and API response."
        )


if __name__ == "__main__":
    main()