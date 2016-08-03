library(ggplot2)

## In R it is simpler to just save the excel as a CSV and then load in the data.
clients <- read.csv("case/clients.csv")

clients2 <- subset(clients, Gender %in% c("M","F"))

states <- tolower(clients$State)
states[states=="florida"]<- "fl"
states[states=="alabama"]<- "al"
states[states=="south carolina"]<- "sc"
states[states=="louisiana"]<- "la"
states[states=="georgia"]<- "ga"
states_main <- c("ga","la","al","sc","fl")
states <- subset(states, states %in% states_main)

qplot(clients2$Gender, main = "Gender", fill = clients2$Gender)
qplot(clients$Age, main = "Age", fill = 'blue')
qplot(clients$Education.Level, main = "Education Level", fill = clients$Education.Level)
qplot(states, main = "State", fill = states)