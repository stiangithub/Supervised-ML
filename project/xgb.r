library(modelr)
library(tidyverse)
song_extra<-read_csv("~/Desktop/DS5110/project/project/song_extra_info.csv")
song<-read_csv("~/Desktop/DS5110/project/project/songs.csv")
train<-read_csv("~/Desktop/DS5110/project/project/train.csv")
members<-read_csv("~/Desktop/DS5110/project/project/members.csv")
train_song<-left_join(train, song,by="song_id")
train_song_member<-left_join(train_song,members,by="msno")


test<-read_csv("~/Desktop/DS5110/project/project/test.csv")
test<-left_join(test, song,by="song_id")
test<-left_join(test, members,by="msno")

##xgboost


#calculate each user's mean target value.
msno_influ_train_1<-train_song_member %>% group_by(msno) %>%summarise ( msno_repeat=mean(target,na.rm=T) )
#calculate each song's mean target value
songid_influ_train_1<-train_song_member %>% group_by(song_id) %>%summarise ( song_repeat=mean(target,na.rm=T) )
#calculate each artist's mean target value
artist_influ_train_1<-train_song_member %>% group_by(artist_name) %>%summarise(artist_repeat = mean(target,na.rm=T))
#calculate each composer's mean target value
composer_influ_train_1<-train_song_member%>%group_by(composer)%>% summarise (composer_repeat = mean(target,na.rm=T))
#calculate each lyricist's mean target value
lyricist_influ_train_1<-train_song_member%>%group_by(lyricist)%>%summarise (lyricist_repeat=mean(target,na.rm=T))


train_song_member_1<-left_join(train_song_member,msno_influ_train_1,by = "msno")
train_song_member_1<-left_join(train_song_member_1,songid_influ_train_1 ,by = "song_id")
train_song_member_1<-left_join(train_song_member_1,artist_influ_train_1 ,by = "artist_name")
train_song_member_1<-left_join(train_song_member_1,composer_influ_train_1 ,by = "composer")
train_song_member_1<-left_join(train_song_member_1,lyricist_influ_train_1 ,by = "lyricist")


test<-left_join(test,msno_influ_train_1,by = "msno")
test<-left_join(test,songid_influ_train_1 ,by = "song_id")
test<-left_join(test,artist_influ_train_1 ,by = "artist_name")
test<-left_join(test,composer_influ_train_1 ,by = "composer")
test<-left_join(test,lyricist_influ_train_1 ,by = "lyricist")



dataset<-select(train_song_member_1,
                msno_repeat,song_repeat,artist_repeat,composer_repeat,lyricist_repeat,target)
test<-select(test,
             msno_repeat,song_repeat,artist_repeat,composer_repeat,lyricist_repeat,id)

set.seed(13)
dataset = na.omit(dataset)

dataset2<-dataset
test2<-test

library(plyr)
into_factor <- function(x){
  
  if(class(x) == "factor"){
    n = length(x)
    data.fac = data.frame(x = x,y = 1:n)
    output = model.matrix(y~x,data.fac)[,-1]
    ## Convert factor into dummy variable matrix
  }else{
    output = x
    ## if x is numeric, output is x
  }
  output
}

dataset2 = colwise(into_factor)(dataset2)
dataset2 = do.call(cbind,dataset2)
dataset2 = as.data.frame(dataset2)

test2 = colwise(into_factor)(test2)
test2 = do.call(cbind,test2)
test2 = as.data.frame(test2)

label <- as.matrix(dataset2[,6,drop =F])

data <- as.matrix(dataset2[,-6,drop =F])

label2<-as.matrix(test2[,6,drop = F])

test_data<-as.matrix(test2[,-6,drop=F])


library(xgboost)

xgmat <- xgb.DMatrix(data, label = label, missing = -10000)
xgtest<-xgb.DMatrix(test_data, label = label2)

param <- list("objective" = "binary:logistic","bst:eta" = 1,"bst:max_depth" = 2,
              "eval_metric" = "logloss","silent" = 1,"nthread" = 16 ,"min_child_weight" =1.45)
nround =275

bst = xgb.train(param, xgmat, nround )

res2 = predict(bst,xgtest)


xgb_answer<-data.frame(id = label2, target = res2)
xgb_answer$id<-as.integer(xgb_answer$id)

write_csv(xgb_answer,"xgb_answer.csv")