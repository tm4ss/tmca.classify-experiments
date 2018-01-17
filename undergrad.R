undergrad <- function (control = "control.txt", stem = T, strip.tags = T, 
          ignore.case = T, table.file = "tablefile.txt", threshold = 0.01, 
          pyexe = NULL, sep = NULL, printit = TRUE, fullfreq = FALSE, 
          python3 = FALSE, alphanumeric.only = TRUE, textincontrol = FALSE, 
          remove.regex = NULL) 
{
  os.type <- .Platform$OS.type
  if (is.null(pyexe)) 
    pyexe <- "python"
  if (python3 == FALSE) {
    call <- paste(system.file("makerfile", package = "ReadMe"), sep = "")
  }
  else {
    call <- paste(system.file("makerfile3-0.py", package = "ReadMe"), sep = "")
  }
  if (!stem) 
    call <- paste(call, "--no-stem")
  if (!strip.tags) 
    call <- paste(call, "--tags")
  if (!ignore.case) 
    call <- paste(call, "--case-sensitive")
  if (!printit) 
    call <- paste(call, "--silent")
  if (alphanumeric.only) 
    call <- paste(call, "--alphanumeric-only")
  if (textincontrol) 
    call <- paste(call, "--in-control-file")
  if (is.data.frame(control)) {
    write.table(control, "readmetmpctrl.txt", row.names = FALSE, 
                quote = FALSE)
    control <- "readmetmpctrl.txt"
  }
  if (!is.null(sep)) {
    call <- paste(call, " --separator \",\"", sep = "")
  }
  if (!is.null(remove.regex)) {
    call <- paste(call, " --remove-regex ", "'", remove.regex, 
                  "'", sep = "")
  }
  call <- paste(call, "--control-file", paste(control, sep = ""))
  call <- paste(call, "--table-file", paste(table.file, sep = ""))
  call <- paste(call, "--threshold", threshold)
  print(paste(pyexe, call))
  sysres <- system(paste(pyexe, call))
  if (sysres == -1) {
    if (os.type == "unix") 
      stop("Python pyexe must be installed and on system path.")
    pyexe <- NULL
    for (i in 10:50) {
      pyexe <- paste("/python", i, "/python.exe", sep = "")
      if (file.exists(pyexe)) {
        warning(paste("Python not on path. Using", pyexe))
        break
      }
    }
    if (is.null(pyexe)) {
      stop("Python pyexe must be installed and on system path.")
    }
    sysres <- system(paste(pyexe, call))
  }
  if (sysres != 0) {
    stop("Python module failed. Aborting undergrad.")
  }
  tab <- read.csv(table.file)
  ret <- list()
  ret$trainingset <- tab[tab$TRAININGSET == 1, ]
  ret$testset <- tab[tab$TRAININGSET == 0, ]
  cnames <- colnames(tab)
  ncols <- length(cnames)
  formula <- paste(cnames[4], "+...+", cnames[ncols], "~TRUTH", 
                   sep = "")
  formula <- as.formula(formula)
  ret$formula <- formula
  ret$features <- 15
  ret$n.subset <- 300
  ret$prob.wt <- 1
  ret$boot.se <- FALSE
  ret$nboot = 300
  ret$printit = printit
  return(ret)
}
