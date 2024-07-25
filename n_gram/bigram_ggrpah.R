install.packages("ggraph")
install.packages("tidyverse")
install.packages("tidytext")
install.packages("janitor")
install.packages("igraph")
install.packages("showtext")
library(tidyverse)
library(tidytext)
library(janitor)
library(ggraph)
library(igraph)
library(showtext)
font_add_google("Noto Sans SC", "noto")
showtext_auto()
data <- read_csv("~/Desktop/intern/cleaned_data/alldata.csv", show_col_types = FALSE)
data <- clean_names(data)
bigrams <- data %>%
  unnest_tokens(bigram, title, token = "ngrams", n = 2)#change gram
bigram_counts <- bigrams %>%
  count(bigram, sort = TRUE)
bigram_separated <- bigrams %>%
  separate(bigram, into = c("word1", "word2"), sep = " ")
bigram_separated <- bigram_separated %>%
  filter(!is.na(word1) & !is.na(word2))#remove NA
bigram_graph <- bigram_separated %>%
  count(word1, word2, sort = TRUE) %>%
  filter(n > 20) # change frequency
bigram_igraph <- graph_from_data_frame(bigram_graph)
set.seed(2017)
bigram_plot <- ggraph(bigram_igraph, layout = "fr") + 
  geom_edge_link(color = "grey", show.legend = FALSE) + 
  geom_node_point(color = "black", size = 1) + 
  geom_node_text(aes(label = name), color = "black", vjust = 1, hjust = 1, family = "noto", size = 2) + 
  theme_void()
print(bigram_plot)

