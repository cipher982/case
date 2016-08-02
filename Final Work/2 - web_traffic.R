traffic <- read.csv('case/traffic.csv')
install.packages("tseries")
library(tseries)


## convert into date format
traffic$Month <- as.Date(as.character(traffic$Month),format="%m/%d/%Y")

traffic2 <- ts(traffic$Number.of.Visits, start = 2006, end = c(2015,4), frequency = 12)
summary(traffic2)

plot(traffic2)

abline(reg=lm(traffic2~time(traffic2)))

plot(aggregate(traffic2,FUN=mean))

adf.test(diff(log(traffic2)), alternative="stationary", k=0)

acf(log(traffic2))

acf(diff(log(traffic2)))

pacf(diff(log(traffic2)))

fit <- arima(log(traffic2), c(0, 1, 1),seasonal = list(order = c(0, 1, 1), period = 12))

pred <- predict(fit, n.ahead = 12*2)

traffic3 <- window(traffic2, 2013)

ts.plot(traffic3,2.718^pred$pred, log = "y", lty = c(1,3), col = "blue", xlab = 'Year', ylab = "Visits")