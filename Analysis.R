library(tidyverse)
library(tidytext)
library(tidymodels)
library(zeallot)
library(keras)

## Create Empty Placeholder for data
Training_Data <- list(Training_Set = NA, Training_Dictionary = NA, Training_Matrix = NA, Training_Label = NA)
Validation_Data <- list(Validation_Set = NA, Validation_Matrix = NA, Validation_Label = NA)
Testing_Data <- list(Testing_Set = NA, Testing_Matrix = NA)

## Load in the data
Training_Data$Training_Set <- read_csv("SexComment.csv", name_repair = str_to_title) %>%
  select(Index, Comment_text, Label) %>%
  rename(ID = Index, Review = Comment_text, Rating = Label)
Testing_Data$Testing_Set <- read_csv("SexWeibo.csv", name_repair = str_to_title) %>%
  mutate(ID = 1:nrow(.)) %>%
  select(ID, Weibo_text) %>%
  rename(Review = Weibo_text)

## Split the training data to create a validation data
Validation_Data$Validation_Set <- Training_Data$Training_Set %>%
  initial_split(., prop = 0.95, strata = "Rating") %>%
  testing()
Training_Data$Training_Set <- Training_Data$Training_Set %>%
  anti_join(., Validation_Data$Validation_Set, by = colnames(.))

## Create a dictionary for all the words that appeared in the training data
Training_Data$Training_Dictionary <- Training_Data$Training_Set %>%
  unnest_tokens(input = Review, output = Review_Word, token = "words") %>%
  count(Review_Word, sort = T) %>%
  mutate(
    n = NULL,
    Word_ID = 1:nrow(.)
  )

## Creating the training and validation labels
list(Training_Data$Training_Set, Validation_Data$Validation_Set) %>%
  map(., ~arrange(.x, ID)) %>%
  map(., ~`$`(.x, Rating)) %->%
  c(Training_Data$Training_Label, Validation_Data$Validation_Label)
list(Training_Data$Training_Set, Validation_Data$Validation_Set) %>%
  map(., ~select(.x, -Rating)) %->%
  c(Training_Data$Training_Set, Validation_Data$Validation_Set)

## Break up the sentences, one-hot-encoding the words using the dictionary
list(Training_Data$Training_Set, Validation_Data$Validation_Set, Testing_Data$Testing_Set) %>%
  map(., ~unnest_tokens(.x, input = Review, output = Review_Word, token = "words")) %>%
  map(., ~group_by(.x, ID)) %>%
  map(., ~summarise(.x, Review_Word = Review_Word, Word_Placement = paste0("Word_", 1:n()))) %>%
  map(., ungroup) %>%
  map(., ~pivot_wider(.x, names_from = Word_Placement, values_from = Review_Word)) %>%
  map(., ~arrange(.x, ID)) %>%
  map(., ~select(.x, starts_with("Word"))) %>%
  map(., ~map(.x, .f = ~left_join(.x %>% as_tibble() %>% rename(Review_Word = value), Training_Data$Training_Dictionary, by = "Review_Word"))) %>%
  map(., bind_cols) %>%
  map(
    .x = .,
    .f = ~`colnames<-`(.x, paste0(c("Word_", "Number_"), rep(1:(ncol(.x)/2), each = 2)))
  ) %>%
  map(., ~select(.x, starts_with("Number"))) %>%
  map(., as.matrix) %->%
  c(Training_Data$Training_Set, Validation_Data$Validation_Set, Testing_Data$Testing_Set)

## Check the dimensions of the data
list(Training_Data$Training_Set, Validation_Data$Validation_Set, Testing_Data$Testing_Set) %>%
  map(., dim)

## Turn the data into matrices
Training_Data$Training_Matrix <- matrix(
  rep(0, nrow(Training_Data$Training_Set) * nrow(Training_Data$Training_Dictionary)),
  nrow = nrow(Training_Data$Training_Set),
  ncol = nrow(Training_Data$Training_Dictionary)
)

Validation_Data$Validation_Matrix <- matrix(
  rep(0, nrow(Validation_Data$Validation_Set) * nrow(Training_Data$Training_Dictionary)),
  nrow = nrow(Validation_Data$Validation_Set),
  ncol = nrow(Training_Data$Training_Dictionary)
)

Testing_Data$Testing_Matrix <- matrix(
  rep(0, nrow(Testing_Data$Testing_Set) * nrow(Training_Data$Training_Dictionary)),
  nrow = nrow(Testing_Data$Testing_Set),
  ncol = nrow(Training_Data$Training_Dictionary)
)

for (i in 1:nrow(Training_Data$Training_Set)) {
  for (j in 1:ncol(Training_Data$Training_Set)) {
    if(is.na(Training_Data$Training_Set[i,j])) {
      next
    } else {
      Training_Data$Training_Matrix[i, Training_Data$Training_Set[i,j]] <- 1
    }
  }
  print(i)
}

for (i in 1:nrow(Validation_Data$Validation_Set)) {
  for (j in 1:ncol(Validation_Data$Validation_Set)) {
    if(is.na(Validation_Data$Validation_Set[i,j])) {
      next
    } else {
      Validation_Data$Validation_Matrix[i, Validation_Data$Validation_Set[i,j]] <- 1
    }
  }
  print(i)
}

for (i in 1:nrow(Testing_Data$Testing_Set)) {
  for (j in 1:ncol(Testing_Data$Testing_Set)) {
    if(is.na(Testing_Data$Testing_Set[i,j])) {
      next
    } else {
      Testing_Data$Testing_Matrix[i, Testing_Data$Testing_Set[i,j]] <- 1
    }
  }
  print(i)
}

remove(i, j)

## Check the dimensions of the data
list(Training_Data$Training_Matrix, Validation_Data$Validation_Matrix, Testing_Data$Testing_Matrix) %>%
  map(., dim)

list(Training_Data$Training_Label, Validation_Data$Validation_Label) %>%
  map(., length)

## Train the neural network
Neural_Network_Model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "softmax", input_shape = c(ncol(Training_Data$Training_Matrix))) %>%
  layer_dense(units = 4, activation = "softmax") %>%
  layer_dense(units = 1, activation = "hard_sigmoid")

Neural_Network_Model %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.01),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

Training_History <- Neural_Network_Model %>% fit(
  Training_Data$Training_Matrix,
  Training_Data$Training_Label,
  epochs = 3,
  batch_size = 1024,
  validation_data = list(Validation_Data$Validation_Matrix, Validation_Data$Validation_Label)
)

remove(Neural_Network_Model)