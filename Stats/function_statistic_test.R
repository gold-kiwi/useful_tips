test_statistic = function(x,mu,sigma){
	return ( (mean(x) - mu) / sqrt(sigma / length(x)) )
}

critical_region_norm = function(alpha,two_side=TRUE){
	if (two_side){
		return ( list(qnorm( alpha/2) , qnorm(alpha/2,lower.tail=FALSE)) )
	}
	else{
		return ( qnorm( alpha)  )		
	}
}

p_value_norm = function(test_statistic,two_side=TRUE){
	if (two_side){
		if (test_statistic < 0){
			return ( pnorm(test_statistic) + pnorm((-1) * test_statistic,lower.tail=FALSE) )
		}
		else{
			return ( (-1) * pnorm(test_statistic) + pnorm(test_statistic,lower.tail=FALSE) )
			
		}
	}
	else{
		return ( pnorm(test_statistic))
	}
}

critical_region_t = function(alpha,df,two_side=TRUE){
	if (two_side){
		return ( list(qt(alpha/2,df) , qnorm(alpha/2,df,lower.tail=FALSE)) )
	}
	else{
		return ( qnorm( alpha,df)  )		
	}
}

p_value_t = function(test_statistic,df,two_side=TRUE){
	if (two_side){
		if (test_statistic < 0){
			return ( pt(test_statistic,df) + pt((-1) * test_statistic,df,lower.tail=FALSE) )
		}
		else{
			return ( (-1) * pt(test_statistic,df) + pt(test_statistic,df,lower.tail=FALSE) )
			
		}
	}
	else{
		return ( pt(test_statistic,df))
	}
}

test_statistic_cor = function(x1,x2){
	corr = cor(x1,x2)
	return ( corr * sqrt(length(x1)-2) / sqrt(1-corr^2) )
}



