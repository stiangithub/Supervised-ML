library(tidyverse)
a = read_csv("train.csv")

number_of_app<-length(unique(a$app))
number_of_device<-length(unique(a$device))
number_of_os<-length(unique(a$os))
number_of_channel<-length(unique(a$channel))
number_of_ip<-length(unique(a$ip))
number_of_click_time<-length(unique(a$click_time))


df<-data.frame(number_of_app,number_of_channel,number_of_click_time,
                 number_of_device,number_of_ip,number_of_os)

df%>%
  ggplot()+geom_bar(mapping = aes(x = )