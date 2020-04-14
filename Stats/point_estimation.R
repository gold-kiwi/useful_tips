view_sample_distribution = function(n,iter,mean,sd,file_name){
	sample_mean = numeric(length=iter)
	for (i in 1:iter){
	sample = rnorm(n=n,mean=mean,sd=sd)
	sample_mean[i] = mean(sample)
	}
	png(file_name)
	hist(sample_mean,freq=FALSE)
	curve(dnorm(x,mean=mean,sd=sqrt(sd/n)),from=-3,to=3,add=TRUE)
	dev.off()
}


tall = c(165.2,175.9,161.7,174.2,172.1,163.3,170.9,170.6,168.4,171.3)
print(mean(tall))
print(var(tall))

n = 6
dice = ceiling(runif(n=n,min=0,max=6))
print(table(dice))

#n = 6000000
#dice = ceiling(runif(n=n,min=0,max=6))
#print(table(dice))

#barplot(c(2/3,1/3),names.arg=c("men","female"))


#curve(dnorm(x,mean=0,sd=1),from=-4,to=4)
#curve(dnorm(x,mean=1,sd=1),add=TRUE)
#curve(dnorm(x,mean=0,sd=2),add=TRUE)

print(rnorm(n=5,mean=50,sd=10))
print(rnorm(n=5,mean=50,sd=10))

sample = rnorm(n=5,mean=50,sd=10)
#hist(sample)

sample = rnorm(n=10000,mean=50,sd=10)
#hist(sample)

sample_mean = numeric(length=10000)

for (i in 1:10000){
	sample = rnorm(n=10,mean=50,sd=10)
	sample_mean[i] = mean(sample)
}
#hist(sample_mean)

abs_error_se5 = ifelse(abs(sample_mean-50)<=5,1,0)
print(table(abs_error_se5))
#hist(sample_mean,freq=FALSE)
#curve(dnorm(x,mean=50,sd=sqrt(10)),add=TRUE)

if (dev.cur()>1) dev.off()

sample_mean_2 = numeric(length=5000)
for (i in 1:5000){
	sample_2 = rnorm(n=20,mean=50,sd=10)
	sample_mean_2[i] = mean(sample_2)
}
#hist(sample_mean_2,freq=FALSE)
#curve(dnorm(x,mean=50,sd=sqrt(5)),add=TRUE)


n = 1
iter = 5000
mean = 0
sd = 1
view_sample_distribution(n,iter,mean,sd,"plot_1.png")

n = 4
iter = 5000
mean = 0
sd = 1
view_sample_distribution(n,iter,mean,sd,"plot_4.png")

n = 9
iter = 5000
mean = 0
sd = 1
view_sample_distribution(n,iter,mean,sd,"plot_9.png")

n = 16
iter = 5000
mean = 0
sd = 1
view_sample_distribution(n,iter,mean,sd,"plot_16.png")

n = 25
iter = 5000
mean = 0
sd = 1
view_sample_distribution(n,iter,mean,sd,"plot_25.png")


curve(dnorm(x,mean=0,sd=sqrt(1/25)),-3,3)
curve(dnorm(x,mean=0,sd=sqrt(1/16)),-3,3,add=TRUE)
curve(dnorm(x,mean=0,sd=sqrt(1/9)),-3,3,add=TRUE)
curve(dnorm(x,mean=0,sd=sqrt(1/4)),-3,3,add=TRUE)
curve(dnorm(x,mean=0,sd=sqrt(1/1)),-3,3,add=TRUE)


