
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
options(encoding = "UTF-8")

if (.Platform$OS.type == "windows") {
  Sys.setlocale("LC_ALL", "Chinese (Simplified)_China.936")
}

install.packages("stringi")
install.packages("stringr")

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

new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

for(package in required_packages){
  tryCatch({
    library(package, character.only = TRUE)
    print(paste("Successfully loaded:", package))
  }, error = function(e){
    print(paste("Error loading", package, ":", e$message))
  })
}

print(paste("R version:", R.version.string))
print("Loaded package versions:")
for(package in required_packages){
  if(package %in% installed.packages()[,"Package"]){
    print(paste(package, ":", packageVersion(package)))
  }
}