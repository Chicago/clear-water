# The usePackage() function checks to see if the package is installed, if the 
# the package is not installed then the package is installed. 
#Once the package is installed, or was previously installed
#it is put to use through the require() statement.

usePackage <- function(p) 
{
  if (!is.element(p, installed.packages()[,1]))
    install.packages(p, dep = TRUE)
  require(p, character.only = TRUE)
}

