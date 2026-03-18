# Set mirror source
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

# Setup encoding
options(encoding = "UTF-8")

# Setup language enviroment
if (.Platform$OS.type == "windows") {
  Sys.setlocale("LC_ALL", "Chinese (Simplified)_China.936")
} else {
  Sys.setlocale("LC_ALL", "zh_CN.UTF-8")
}

install.packages("tidyverse")
install.packages("lubridate")
install.packages("readxl")
install.packages("scales")
install.packages("gridExtra")
install.packages("forecast")
install.packages("zoo")

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(readxl)
  library(scales)
  library(gridExtra)
  library(forecast)
  library(zoo)
})

tryCatch({
  data <- read.csv("Bedroom.csv",
                   fileEncoding = "UTF-8",
                   check.names = FALSE)

  print("Data Read Successfully！")
  print(summary(data))

}, error = function(e) {
  print(paste("Error:", e$message))
})