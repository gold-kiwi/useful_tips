source("/Users/Kagesuke/Documents/gold-kiwi/practice/stats/function_statistic_test.R")

psycho_test = c(13,14,7,12,10,6,8,15,4,14,9,6,10,12,5,12,8,8,12,15)
print(psycho_test)
print(summary(psycho_test))
ts = test_statistic(psycho_test,12,10)
print(ts)

print(critical_region_norm(0.05))

print(p_value_norm(ts))

ts = test_statistic(psycho_test,12,var(psycho_test))
print(ts)

print(critical_region_t(0.05,length(psycho_test)-1))

print(p_value_t(ts,length(psycho_test)-1))

print(t.test(psycho_test,mu=12))

st_test1 = c(6,10,6,10,5,3,5,9,3,3,11,6,11,9,7,5,8,7,7,9)
st_test2 = c(10,13,8,15,8,6,9,10,7,3,18,14,18,11,12,5,7,12,7,7)
