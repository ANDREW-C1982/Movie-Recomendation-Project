# Importing necessary libraries

library(tidyverse)
library(caret)
library(dslabs)
library(dplyr)
library(ggplot2)
library(matrixStats)
library(data.table)
library(lubridate)

# Data pre-processing

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
str(movies) # view the variables of the movies dataset

summary(movies) # view the summaries of the movies dataset
head(movies) # view the first six lines of the movies dataset

summary(ratings) # view the summaries of the movies dataset
head(ratings) # view the first six lines of the movies dataset

# if using R 3.6 or earlier
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# data pre-processing
# summary
summary(movielens) # view the summaries of the movielens dataset
str(movielens) # view the variables of the movielens dataset
# number of unique users that provided ratings
movielens %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))
# Most watched movies
Top_5_movies <- movielens %>%
  dplyr::count(movieId,title) %>%
  top_n(5) %>%
  pull(movieId,title)
Top_5_movies        

# Visualizations
#Total Views of the Top Films in Movielens
movielens%>%
  group_by(title)%>%
  summarise(n=n())%>%
  top_n(7)%>%
  ggplot(aes(title,n))+geom_bar(stat="identity", fill = 'green')+geom_text(aes(label=n), vjust=-0.3, size=3.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Total Views of the Top Films")

n_distinct(movielens$movieId) # calculate the number of unique novies
n_distinct(movielens$userId) # calculate the number of unique users

# movie rating distribution
movielens %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "blue") + 
  scale_x_log10() + 
  ggtitle("Movie ratings distribution")
# user rating distribution
movielens %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "orange") + 
  scale_x_log10() +
  ggtitle("User ratings distribution")

# Loading and evaluating the training data set:edx
# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

table (complete.cases (edx)) #Checking for missing values

str (edx) # view the variables of the edx dataset

# Data exploration

#Exploring the userId variable
summary (edx$userId)
# how many different users are in the train_set dataset
n_distinct(edx$userId)
#Boxplot for userId variable
boxplot (userId ~ rating, data = edx,
         main = "rating levels based on the userId of an individual",
         xlab = "rating", ylab = "userId", col = "salmon")
#Histogram for userId variable
qplot(userId,data=edx,margins = TRUE,binwidth = 2000,colour="red")

#Exploring the movieId variable
summary (edx$movieId)
# how many different movies are in the train_set dataset
n_distinct(edx$movieId)
#Boxplot for movieId variable
boxplot (movieId ~ rating, data = edx,
         main = "rating levels based on the movieId of an individual",
         xlab = "rating", ylab = "movieId", col = "salmon")
#Histogram for movieId variable
qplot(movieId,data=edx,margins = TRUE,binwidth = 2000,colour="red")

#Exploring categoricals:genres and title
# common genres
genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

#Total views of the Top Films by Title
edx%>%
  group_by(title)%>%
  summarise(n=n())%>%
  top_n(7)%>%
  ggplot(aes(title,n))+geom_bar(stat="identity", fill = 'steelblue')+geom_text(aes(label=n), vjust=-0.3, size=3.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Total Views of the Top Films")
#Total views of the Top Films by genres
edx%>%
  group_by(genres)%>%
  summarise(n=n())%>%
  top_n(7)%>%
  ggplot(aes(genres,n))+geom_bar(stat="identity", fill = 'red')+geom_text(aes(label=n), vjust=-0.3, size=3.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Total Views of the Top Films by Genres")

# building the model

# Loading and evaluating the training data set:edx

#least squares estimate of the mean mu
mu <- mean(edx$rating) # calculates mean rating
mu

#average movie rating-effect on the training model:
#movie-specific effect on the model:
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu))
#histogram for average rating for movies
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

#user-specific effect on the model:
#user-specific effect on the model:
user_avgs <- edx %>%
  left_join(movie_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu))
#histogram for average rating for user
user_avgs %>% qplot(b_u, geom ="histogram", bins = 10, data = ., color = I("green"))

#genres-specific effect on the model:
#genres-specific effect on the model:
genres_avgs <-edx%>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i -b_u- mu))
#histogram for average rating for user
genres_avgs %>% qplot(b_g, geom ="histogram", bins = 50, data = ., color = I("yellow"))


#First, let’s create a database that connects movieId to movie title:
movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()
head(movie_titles)

# Here are the 10 best movies according to our estimate and how often they are rated:
edx %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# Here are the 10 worst movies according to our estimate and how often they are rated:
edx %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

#training the model
#training the model
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <-edx%>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i -b_u- mu)/(n()+l))
  predicted_ratings <-
    validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})

# the optimal λ_edx
qplot(lambdas, rmses)  
lambda <- lambdas[which.min(rmses)]
lambda

#calculating RMSE
Regularized_rmse_results<-data.frame(method="Avg movie rating + UserId-effect  + genres-effect ",RMSE = min(rmses))
Regularized_rmse_results
