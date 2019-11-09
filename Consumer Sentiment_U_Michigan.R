# Consumer Sentiment, UMICH
cs.vec=read.csv("UMCSENT2.csv")

#ACF, PACF
acf(cs2)
tsdisplay(cs2)

# Original Data - CS1
cs1=ts(cs.vec[,2],start=1978,deltat=1/12)
cs1

# Plot the original data
plot.ts(cs1,xlab="time", ylab="Consumer Sentiment, UMICH")

plot(decompose(cs1)) #look at seasonality

# Differencing the Data - CS2
cs2=ts(diff(cs.vec[,2]),start=1978,deltat=1/12)
cs2

# Plot the differenced data
plot.ts(cs2,xlab="time", ylab="Consumer Sentiment, UMICH (Differenced)")

plot(decompose(cs2)) #look at seasonality


# Log the Data (not necessary but good to try) - CS3
cs3=ts(diff(log(cs.vec[,2])),start=1978,deltat=1/12)
cs3

# Plot the log-differenced data
plot.ts(cs3,xlab="time", ylab="Consumer Sentiment, UMICH (Log-differenced)")

plot(decompose(cs3)) #look at seasonality

# Model the data - Model #1, Auto Arima 
train_data=window(cs3,end=c(2016,12))
test_data=window(cs3,start=c(2017,1))
train_data
test_data

library(tidyverse)
forecast_value=auto.arima(train_data) %>%forecast(h=22)
forecast_value$mean %>%autoplot()+geom_point()
ts.plot(forecast_value)

#compare test to forecast data
install.packages("plotly")
library(plotly)
cbind(test_data,forecast_value$mean) %>%autoplot()+geom_point()+autolayer(train_data)
ggplotly()

# Model 2 - HW (better forecast)
forecast_value=hw(train_data,h=72) %>%forecast()
forecast_value$mean %>%autoplot()+geom_point()
ggplotly()

#compare test to forecast data
install.packages("plotly")
library(plotly)
cbind(test_data,forecast_value$mean) %>%autoplot()+geom_point()
cbind(test_data,forecast_value$mean) %>%autoplot()+geom_point()+autolayer(train_data)
ggplotly()

# Use original data (HW seems best)
cs1
forecast_value=hw(cs1,h=60) %>%forecast()
forecast_value$mean %>%autoplot()+geom_point()
ggplotly()


forecast_value



