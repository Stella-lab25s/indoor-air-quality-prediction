# Setup workload
options(encoding = "UTF-8")
if (.Platform$OS.type == "windows") {
  Sys.setlocale("LC_ALL", "Chinese (Simplified)_China.936")
}

# Load packages
required_packages <- c(
  "stringi",
  "stringr",
  "tidyverse",
  "lubridate",
  "readxl",
  "scales",
  "gridExtra",
  "forecast",
  "zoo"
)

# Check and load packages
for(package in required_packages){
  if(!require(package, character.only = TRUE)){
    print(paste("Package", package, "is not installed. Please run package_install.r first."))
    quit(save = "no", status = 1)
  }
}

# Create Output Structure Function
create_output_structure <- function (){
  # create basic directory
    dirs <- c(
      "output",
      "output/figure",
      "output/report",
      "output/data"
    )
  # create each directory (if does not exsit)
  for (dir in dirs){
    if(!dir.exists(dir)){
      dir.create(dir, recursive = TRUE)
      print(paste("Create directory:", dir))
    }
  }
}

# Read data function
read_air_quality_data <- function(file_path) {
  tryCatch({
    data <- read.csv(file_path,
                     fileEncoding = "UTF-8",
                     check.names = FALSE)

    # Transfer timestamp
    data$timestamp <- as.POSIXct(data$timestamp)
    data$date <- as.Date(data$timestamp)
    data$hour <- hour(data$timestamp)

    return(data)

  }, error = function(e) {
    print(paste("Error reading data:", e$message))
    return(NULL)
  })
}

# Create visualization function
create_visualizations <- function(data) {
    # Setup English language
    Sys.setlocale("LC_TIME", "English")

    # Setup figure path
    figure_path <- "output/figures"

    # create white theme
    white_theme <- theme_minimal() +
        theme(
            panel.background = element_rect(fill = "white", color = NA),
            plot.background = element_rect(fill = "white", color = NA),
            legend.background = element_rect(fill = "white", color = NA),
            panel.grid.major = element_line(color = "grey90"),
            panel.grid.minor = element_line(color = "grey95")
        )

    # CO2 Daily variation chart
    daily_pattern <- ggplot(data, aes(x = hour, y = co2)) +
        geom_smooth(method = "loess") +
        geom_point(alpha = 0.1) +
        theme_minimal() +
        labs(
            title = "Daily CO2 Pattern",
            x = "Hour of Day (London)",
            y = "CO2 (ppm)"
        )

    # Save figure(CO2 daily variation)
    ggsave(
        file.path(figure_path, "daily_pattern.png"),
        daily_pattern,
        width = 10,
        height = 6,
        bg = "white"
    )

    # Correlation plot
    cor_matrix <- cor(data[c("co2", "temp", "humid", "voc", "pm25", "pm10")])
    correlation_plot <- ggplot(data = reshape2::melt(cor_matrix)) +
        geom_tile(aes(x = Var1, y = Var2, fill = value)) +
        scale_fill_gradient2(
            low = "blue",
            mid = "white",
            high = "red",
            midpoint = 0
        ) +
        white_theme +
         labs(
            title = "Parameter Correlations",
            fill = "Correlation"
        ) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))

    # Save figure
    ggsave(
        file.path(figure_path, "correlation_plot.png"),
        correlation_plot,
        width = 8,
        height = 8,
        bg = "white"
    )

    # Time series figure
    time_series <- ggplot(data, aes(x = timestamp, y = co2)) +
        geom_line(color = "black") +
        theme_minimal() +
        scale_x_datetime(
            labels = date_format("%b %d"),
            date_breaks = "1 day"
        ) +
        labs(
            title = "CO2 Time Series",
            x = "Time (London)",
            y = "CO2 (ppm)"
        ) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))

    # Save figure
    ggsave(
        file.path(figure_path, "time_series.png"),
        time_series,
        width = 12,
        height = 6,
        bg = "white"
    )

    print(paste("Image save at:", normalizePath(figure_path)))

    return(list(
        daily = daily_pattern,
        correlation = correlation_plot,
        timeseries = time_series
    ))
}

# Basic analysis function -基础分析函数
basic_analysis <- function(data) {
  if (is.null(data)) return(NULL)

  tryCatch({
    # Basic statisticians
    stats <- summary(data)

    # Carbon Dionxide analysis
    co2_stats <- data %>%
      group_by(date) %>%
      summarise(
        mean_co2 = mean(co2, na.rm = TRUE),
        max_co2 = max(co2, na.rm = TRUE),
        min_co2 = min(co2, na.rm = TRUE)
      )

    return(list(
      basic_stats = stats,
      co2_analysis = co2_stats
    ))

  }, error = function(e) {
    print(paste("Error in basic analysis:", e$message))
    return(NULL)
  })
}

# Main funtion
main <- function() {
  # Setup English enviroment
  Sys.setlocale("LC_TIME", "English")
  # Setup workload directory
  setwd("C:/Users/Liu Meishan/PycharmProjects/air_quality_analysis")
  print(paste("Current working directory:", getwd()))

  # Create Output directory structure
  create_output_structure()

  # List files
  print("Files in current directory:")
  print(list.files())

  # Read data
  data <- read_air_quality_data("Bedroom.csv")

  if (!is.null(data)) {
    print("Data loaded successfully")
    print(paste("Number of rows:", nrow(data)))
    print(paste("Number of columns:", ncol(data)))

    # Run basic analysis
    results <- basic_analysis(data)

    # Create visualization
    plots <- create_visualizations(data)

    if (!is.null(results)) {
      # Print result
      print("Basic statistics:")
      print(results$basic_stats)

      print("CO2 analysis:")
      print(results$co2_analysis)
    }

    print("Analysis Completed, please check result in 'output' folder.")
  }


  # Add advanced analysis function
  advanced_analysis <- function (data){
    # 1. CO2 concentration analysis
    co2_analysis <- function (data){
      group_by(date) %>%
        summarize(
          exceeded_minutes = sum(co2>1000) *5,  # 5 min interval
          exceeded_percentages = mean(co2>1000) *100
        )
      # CO2 Average level(per hour) Statistical analysis
      hourly_co2 <- data %>%
        group_by(hour) %>%
        summarize(
          mean_co2 = mean(co2),
          sd_co2 = sd(co2)
        )
      return(list(
        exceeded = co2_exceeded,
        hourly = hourly_co2
      ))
    }
    # 2. Ventilation effectiveness Analysis
    ventilation_analysis <- function (data){
    # Calculate CO2 decline rate
    data <- data %>%
      group_by(date) %>%
      mutate(
        co2_change = c(NA, diff(co2)),
        ventilation_event = co2_change < -50 # Define ventilation event
      )
    # Statistiction of ventilation event
    vent_stats <- data %>%
      filter(ventilation_event) %>%
      group_by(date) %>%
      summarise(
        num_events = n(),
        mean_reduction = mean(abs(co2_change), na.rm = TRUE)
      )

    return(vent_stats)
    }

    # 3. Generate visualizations
    create_visualizations <- function (data){
      # CO2 daily variation chart
      daily_pattern <- ggplot(data, aes(x = hour, y = co2)) +
        geom_smooth(method = "loess") +
        geom_point(alpha = 0.1) +
        theme_minimal() +
        labs(
        title = "Daily CO2 Pattern",
        x = "Hour of Day",
        y = "CO2 (ppm)"
      )

      # Heat map of environmental parameter correlation
      cor_matrix <- cor(data[c("co2","temp","humid","voc","pm25","pm10")])
      correlation_polt <- ggplot(data = reshape::melt(cor_matrix)) +
         geom_tile(aes(x = Var1, y = Var2, fill = value)) +
         scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
         theme_minimal() +
         labs(title = "Parameter Correlations")

      # Save graph
      ggsave("daily_partten.png", daily_pattern)
      ggsave("correlation_plt.png", correlation_polt)

      return(list(
        daily = daily_pattern,
        correlation = correlation_polt
      ))
    }

    # 4. Statistical tests
    statistical_tests <- function (data){
      # CO2 concentration normality test
      shapiro_test <- shapiro.test(data$co2)

      # Correlation test between parameters
      cor_tests <- list()
      params <- c("co2","temp", "humid", "voc", "pm25", "pm10")
      for (i in 1: (length(params)-1)){
        for (j in (i+1): (length(params))){
          test_result <- cor.test(data[[params[i]]], data[[params[j]]])
          cor_test[[paste(params[i], params[j], sep="_")]] <- test_result
        }
      }
      return(list(
        normality = shapiro_test,
        correlations = cor_tests
      ))
    }
    # Execute all analysis
    results <- list(
      co2 = co2_analysis(data),
      ventilation = ventilation_analysis(data),
      visualization = create_visualizations(data),
      statistics = statistical_tests(data)
    )

    # Generate Analysis report
    sink("analysis_report.txt")
    cat("Air Quality Analysis Report\n")
    cat("=========================\n\n")

    cat("1. CO2 Analysis\n")
    cat("---------------\n")
    print(results$co2$exceeded)
    cat("\nHourly CO2 Patterns:\n")
    print(results$co2$hourly)

    cat("\n2. Ventilation Analysis\n")
    cat("----------------------\n")
    print(results$ventilation)

    cat("\n3. Statistical Tests\n")
    cat("-------------------\n")
    print(results$statistics$normality)

    sink()

    return(results)
  }

  # Modify the main function
  main <- function (){
    setwd("C:/Users/Liu Meishan/PycharmProjects/air_quality_analysis")
    data <- read_air_quality_data("Bedroom.csv")

    if (! is.null(data)){
      # Basic analysis
      basic_results <- basic_analysis(data)

      # Advanced anaysis
      advanced_results <- advanced_analysis(data)

      # Print the key finding
      print("Key Findings:")
      print("-------------")

      # CO2 exceedance
      exceeded_summary <- advanced_results$co2$exceeded %>%
      summarise(
        total_exceeded_minutes = sum(exceeded_minutes),
        avg_exceeded_percentage = mean(exceeded_percentage)
      )
    print(paste("Total minutes with CO2 > 1000ppm:", exceeded_summary$total_exceeded_minutes))
    print(paste("Average percentage of time exceeded:", round(exceeded_summary$avg_exceeded_percentage, 2), "%"))

      # Ventilation effectiveness
    vent_summary <- advanced_results$ventilation %>%
      summarise(
        total_events = sum(num_events),
        avg_reduction = mean(mean_reduction)
      )
    print(paste("Total ventilation events:", vent_summary$total_events))
    print(paste("Average CO2 reduction per event:", round(vent_summary$avg_reduction, 2), "ppm"))
    }
  }
}


# Run main function
tryCatch({
  main()
}, error = function(e) {
  print(paste("Error in main function:", e$message))
})