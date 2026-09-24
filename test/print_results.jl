using CSV
using DataFrames
using Statistics

# Folder containing the CSV files
folder = ""

# Iterate over all CSV files in the folder
for file in readdir(folder)

    # Only process CSV files
    if endswith(lowercase(file), ".csv")

        filepath = joinpath(folder, file)

        # Read CSV
        df = CSV.read(filepath, DataFrame)

        # Select only rows where iterations > 0
        valid = df.iterations .> 0

        # Calculate averages
        avg_time = mean(df.time[valid])
        avg_iterations = mean(df.iterations[valid])

        # Get filename without extension
        filename = splitext(file)[1]

        # Print results
        println(
            "$(filename): " *
            "average time = $(round(avg_time, digits=4)), " *
            "average iterations = $(round(avg_iterations, digits=2))"
        )
    end
end