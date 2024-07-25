#https://www.tidytextmining.com/ngrams
install.packages("tidyverse")
install.packages("tidytext")
install.packages("janitor")
library(tidyverse)
library(tidytext)
library(janitor)
data <- read_csv("~/Desktop/intern/cleaned_data/alldata.csv", show_col_types = FALSE)
data <- clean_names(data)
ngrams <- data %>%
  unnest_tokens(ngram, title, token = "ngrams", n = 1)#for bi, tri and more grams, edit n=number
ngram_counts <- ngrams %>%
  count(ngram, sort = TRUE)
print(ngram_counts, n = 2000)
write.table(ngram_counts, file = "~/Desktop/intern/n_gram/unigram.txt", row.names = FALSE, sep = "\t")#change txt name

