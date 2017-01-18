conf_matrix<-function(var1,var2,show=TRUE){
  conf_matrix<-table(var1,var2)
  fn<- conf_matrix[1,2]/(conf_matrix[1,1]+conf_matrix[1,2])
  tp<- conf_matrix[2,2]/(conf_matrix[2,1]+conf_matrix[2,2])
  precision<- conf_matrix[2,2]/(conf_matrix[1,2]+conf_matrix[2,2])
  if(show==TRUE)
  {
    print(table(var1,var2))
    cat("False Negative Rate =",fn,
        "\nTrue Positive Rate =",tp,
        "\nPrecision = ", precision)
  }
  #else
  #{
  #  my_list<- list("tpr"=tp,"fnr"=fn,"prec"=precision)
  #}
  
  return(invisible(list("tpr"=tp,"fnr"=fn,"prec"=precision)))
}
