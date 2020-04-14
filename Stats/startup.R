time_a = c(60,100,50,40,50,230,120, 240,200,30)
time_b = c(50,60,40,50,100,80,30,20,100,120)
hist_a = hist(time_a,col='#ff00ff40',border='#ff00ff')
#hist_b = hist(time_b)
hist_ab = hist(time_b,col='#0000ff40',border='#0000ff',add=T)
labels = c('time_a','time_b')
legend('topright',legend=labels,col=c('#ff00ff40','#0000ff40'),pch=15,cex=1)
text(hist_a$mid,hist_a$counts,hist_a$counts,cex=0.8,pos=3,col='red')
text(hist_a$mid,hist_a$counts-max(hist_a$counts)*0.02,hist_ab$counts,cex=0.8,pos=3,col='blue')
mean_a = mean(time_a)
mean_b = mean(time_b)
sd_a = sd(time_a)
sd_b = sd(time_b)
std_a = (time_a - mean_a) / sd_a
std_b = (time_b - mean_b) / sd_b


