# Decision Tree & Random Forest Tree Plot Extraction in R
### Complete Guide: Single Trees · Random Forest · ggplot2 · Custom Visualisation

> **Stack:** `rpart` · `rpart.plot` · `randomForest` · `ggplot2` · `ggraph` · `igraph` · `tidygraph` · `DiagrammeR` · `reprtree`

---

## Table of Contents

1. [Overview — Tree Plot Methods](#1-overview--tree-plot-methods)
2. [Part A — Single Decision Tree Plots](#2-part-a--single-decision-tree-plots)
   - 2a. [rpart + rpart.plot (easiest)](#2a-rpart--rpartplot-easiest)
   - 2b. [rpart + plot() base R](#2b-rpart--plot-base-r)
   - 2c. [party / partykit (conditional trees)](#2c-party--partykit-conditional-trees)
   - 2d. [DiagrammeR (interactive)](#2d-diagrammer-interactive)
3. [Part B — Extract Tree Data from randomForest](#3-part-b--extract-tree-data-from-randomforest)
   - 3a. [getTree() — raw node table](#3a-gettree--raw-node-table)
   - 3b. [Parse tree into edge/node data frames](#3b-parse-tree-into-edgenode-data-frames)
   - 3c. [reprtree — representative tree](#3c-reprtree--representative-tree)
4. [Part C — ggplot2 Tree Plots from randomForest](#4-part-c--ggplot2-tree-plots-from-randomforest)
   - 4a. [Build layout with ggraph + igraph](#4a-build-layout-with-ggraph--igraph)
   - 4b. [Pure ggplot2 from scratch (full control)](#4b-pure-ggplot2-from-scratch-full-control)
   - 4c. [ggplot2 horizontal tree layout](#4c-ggplot2-horizontal-tree-layout)
   - 4d. [Feature importance bar chart (ggplot2)](#4d-feature-importance-bar-chart-ggplot2)
   - 4e. [OOB error curve (ggplot2)](#4e-oob-error-curve-ggplot2)
5. [Part D — sparklyr / Databricks Tree Extraction](#5-part-d--sparklyr--databricks-tree-extraction)
6. [Save & Export Plots](#6-save--export-plots)
7. [Quick-Reference Cheatsheet](#7-quick-reference-cheatsheet)

---

## 1. Overview — Tree Plot Methods

```
WHICH METHOD TO USE?
─────────────────────────────────────────────────────────────────────

Single Decision Tree (rpart)
  ├── Quick view          → rpart.plot()           [best for reports]
  ├── Base R classic      → plot() + text()
  ├── Interactive / HTML  → DiagrammeR::grViz()
  └── Conditional tree    → partykit::plot()

Random Forest (randomForest)
  ├── One tree from RF    → getTree() → rpart.plot  [most practical]
  ├── Representative tree → reprtree::ReprTree()
  ├── ggplot2 custom      → getTree() → igraph → ggraph
  └── Pure ggplot2        → getTree() → parse → geom_segment + geom_label

Feature / Error Plots (ggplot2)
  ├── Feature importance  → importance() → ggplot2 geom_col
  ├── OOB error curve     → rf$err.rate  → ggplot2 geom_line
  └── Partial dependence  → partialPlot() data → ggplot2
```

---

## 2. Part A — Single Decision Tree Plots

### 2a. rpart + rpart.plot (easiest)

```r
install.packages(c("rpart", "rpart.plot"))
library(rpart)
library(rpart.plot)

# ── Train a single decision tree ─────────────────────────────────────
tree_model <- rpart(
  formula = Churn ~ tenure + MonthlyCharges + TotalCharges +
                    Contract + InternetService,
  data    = train,
  method  = "class",              # classification
  control = rpart.control(
    maxdepth  = 4,                # limit depth for readability
    minsplit  = 20,               # min obs to attempt a split
    minbucket = 10,               # min obs in terminal node
    cp        = 0.01              # complexity parameter (pruning)
  )
)

print(tree_model)                  # text summary
printcp(tree_model)                # cross-validation error table
plotcp(tree_model)                 # CP vs error plot

# ── Basic plot ────────────────────────────────────────────────────────
rpart.plot(tree_model)

# ── Styled plot ───────────────────────────────────────────────────────
rpart.plot(
  tree_model,
  type        = 4,       # 0–5: controls label style
                         # 4 = labels at all nodes, split labels above
  extra       = 106,     # 104=% in class, 106=% + n, 101=prob
  fallen.leaves = TRUE,  # align leaves at the bottom
  shadow.col  = "gray70",
  branch      = 0.5,     # branch curvature (0=V-shape, 1=right angle)
  gap         = 0,
  tweak       = 1.2,     # scale up text
  main        = "Churn Decision Tree (depth = 4)",
  col         = "gray20",
  border.col  = "gray40",
  # Colour nodes by class:
  box.palette = list(
    Active = "#E1F5EE",
    Churn  = "#FCEBEB"
  ),
  # Colour splits by feature:
  split.col   = "#0F6E56",
  split.font  = 2
)
```

#### `rpart.plot` type & extra quick reference

| `type` | What it shows |
|---|---|
| `0` | Split label below node |
| `1` | Split label below, class label inside |
| `2` | Split label below, class label inside + stats |
| `3` | Split label above branch, class inside |
| `4` | Split above branch, all labels — **most readable** |
| `5` | As type 4 + branch lines |

| `extra` | What it shows |
|---|---|
| `0` | Class label only |
| `1` | Predicted class + count |
| `100` | % of observations at node |
| `104` | % of class at node |
| `106` | % + n (obs count) — **recommended** |
| `108` | % + prob + n |

### 2b. rpart + plot() base R

```r
# ── Base R plot ───────────────────────────────────────────────────────
par(mar = c(1, 1, 2, 1))           # set margins
plot(
  tree_model,
  uniform   = TRUE,                # equal branch lengths
  compress  = TRUE,                # compact horizontal layout
  margin    = 0.1,                 # margin fraction
  branch    = 0.5,                 # branch angle
  main      = "Churn Decision Tree"
)
text(
  tree_model,
  use.n    = TRUE,                 # show n at each node
  all      = TRUE,                 # label all nodes (not just leaves)
  cex      = 0.75,                 # text size
  col      = "black",
  fancy    = FALSE                 # TRUE = ovals for internal, boxes for leaves
)

# ── Fancy version with boxes ─────────────────────────────────────────
plot(tree_model, uniform = TRUE, branch = 0.4, margin = 0.1)
text(tree_model, fancy = TRUE, fwidth = 0.4, fheight = 0.4,
     col = c("#0F6E56", "#A32D2D"), cex = 0.7)
```

### 2c. party / partykit (conditional trees)

```r
install.packages(c("party", "partykit"))
library(partykit)

# ── Conditional inference tree (unbiased splits) ─────────────────────
ctree_model <- ctree(
  Churn ~ tenure + MonthlyCharges + TotalCharges + Contract,
  data    = train,
  control = ctree_control(
    maxdepth = 4,
    mincriterion = 0.95           # 1 - p-value threshold for splitting
  )
)

# ── Default plot ──────────────────────────────────────────────────────
plot(ctree_model,
     main          = "Conditional Inference Tree — Churn",
     gp            = gpar(fontsize = 9),
     inner_panel   = node_inner,
     terminal_panel = node_barplot)

# ── Compact version ───────────────────────────────────────────────────
plot(ctree_model,
     type = "simple",             # "simple" or "extended"
     main = "Churn Conditional Tree (simple)")

# ── Extract tree structure as data frame ─────────────────────────────
ctree_df <- as.data.frame(ctree_model)
print(ctree_df)
```

### 2d. DiagrammeR (interactive HTML)

```r
install.packages("DiagrammeR")
library(DiagrammeR)

# ── Convert rpart tree to DOT language ───────────────────────────────
rpart_to_dot <- function(tree, digits = 3) {
  frame   <- tree$frame
  nodes   <- rownames(frame)
  is_leaf <- frame$var == "<leaf>"

  lines <- c("digraph tree {",
             "  node [shape=box, style=filled, fontname=Arial, fontsize=11]",
             "  edge [fontname=Arial, fontsize=9]")

  for (i in seq_along(nodes)) {
    node_id <- nodes[i]
    label   <- if (is_leaf[i]) {
      cls   <- levels(tree$y)[which.max(frame$yval2[i, ])]
      pct   <- round(max(frame$yval2[i, ]) / frame$n[i] * 100, 1)
      paste0('"', node_id, '" [label="', cls, '\\n', pct, '%\\nn=',
             frame$n[i], '", fillcolor="',
             ifelse(cls == "Churn", "#FCEBEB", "#E1F5EE"), '"]')
    } else {
      split_var <- as.character(frame$var[i])
      paste0('"', node_id, '" [label="', split_var,
             '\\nn=', frame$n[i], '", fillcolor="#F0F0EC"]')
    }
    lines <- c(lines, paste0("  ", label))
  }

  # Add edges using left/right daughter info from getTree()
  tree_tbl <- getTree(tree, labelVar = TRUE)
  for (r in seq_len(nrow(tree_tbl))) {
    if (tree_tbl[r, "left daughter"] > 0) {
      ld  <- tree_tbl[r, "left daughter"]
      rd  <- tree_tbl[r, "right daughter"]
      sp  <- tree_tbl[r, "split var"]
      pt  <- round(tree_tbl[r, "split point"], digits)
      lines <- c(lines,
                 paste0('  "', r, '" -> "', ld, '" [label="<= ', pt, '"]'),
                 paste0('  "', r, '" -> "', rd, '" [label="> ',  pt, '"]'))
    }
  }

  lines <- c(lines, "}")
  paste(lines, collapse = "\n")
}

# ── Render ────────────────────────────────────────────────────────────
dot_code <- rpart_to_dot(tree_model)
grViz(dot_code)

# ── Or use a simpler built-in approach ───────────────────────────────
# install.packages("rattle")
# library(rattle)
# fancyRpartPlot(tree_model, caption = "Churn Decision Tree")
```

---

## 3. Part B — Extract Tree Data from randomForest

### 3a. getTree() — raw node table

```r
library(randomForest)

set.seed(42)
rf_model <- randomForest(
  Churn ~ tenure + MonthlyCharges + TotalCharges +
           Contract + InternetService + PaymentMethod,
  data       = train,
  ntree      = 200,
  mtry       = 3,
  importance = TRUE
)

# ── Extract tree k (1-indexed) ────────────────────────────────────────
k <- 1    # which tree to extract
tree_k <- getTree(
  rf_model,
  k        = k,
  labelVar = TRUE   # use feature names instead of column indices
)

print(tree_k)
# Columns:
#   left daughter  : row index of left child  (-1 if terminal)
#   right daughter : row index of right child (-1 if terminal)
#   split var      : feature name used at this node (NA if terminal)
#   split point    : threshold value for the split
#   status         : 1 = internal node, -1 = terminal (leaf)
#   prediction     : predicted class at leaf (NA if internal)

# ── Summary of all trees ──────────────────────────────────────────────
tree_stats <- do.call(rbind, lapply(1:rf_model$ntree, function(k) {
  t <- getTree(rf_model, k = k, labelVar = TRUE)
  data.frame(
    tree     = k,
    n_nodes  = nrow(t),
    n_leaves = sum(t$status == -1),
    depth    = max(sapply(which(t$status == -1), function(leaf) {
      d <- 0; node <- leaf
      while (node != 1) {
        parent <- which(t[, "left daughter"]  == node |
                        t[, "right daughter"] == node)
        node   <- parent; d <- d + 1
      }
      d
    }))
  )
}))

summary(tree_stats)
cat("Median depth:", median(tree_stats$depth), "\n")
cat("Median nodes:", median(tree_stats$n_nodes), "\n")
```

### 3b. Parse tree into edge/node data frames

```r
# ── Full parser: getTree() → nodes_df + edges_df ─────────────────────
parse_rf_tree <- function(rf_model, k = 1) {

  tree_tbl <- getTree(rf_model, k = k, labelVar = TRUE)
  tree_tbl$node_id <- seq_len(nrow(tree_tbl))

  # ── Nodes ─────────────────────────────────────────────────────────
  nodes_df <- data.frame(
    id         = tree_tbl$node_id,
    is_leaf    = tree_tbl$status == -1,
    split_var  = as.character(tree_tbl[, "split var"]),
    split_pt   = round(tree_tbl[, "split point"], 3),
    prediction = tree_tbl$prediction,
    stringsAsFactors = FALSE
  )

  # Build label for each node
  nodes_df$label <- ifelse(
    nodes_df$is_leaf,
    paste0("Predict:\n", nodes_df$prediction),
    paste0(nodes_df$split_var, "\n<= ", nodes_df$split_pt)
  )

  # ── Edges ─────────────────────────────────────────────────────────
  internal <- tree_tbl[tree_tbl$status == 1, ]
  edges_df <- data.frame(
    from     = rep(internal$node_id, 2),
    to       = c(internal[, "left daughter"],
                 internal[, "right daughter"]),
    label    = c(rep("Yes (<=)", nrow(internal)),
                 rep("No  (>)",  nrow(internal))),
    stringsAsFactors = FALSE
  )

  list(nodes = nodes_df, edges = edges_df, raw = tree_tbl)
}

# ── Use it ────────────────────────────────────────────────────────────
tree_data  <- parse_rf_tree(rf_model, k = 1)
nodes_df   <- tree_data$nodes
edges_df   <- tree_data$edges

cat("Nodes:", nrow(nodes_df), "| Edges:", nrow(edges_df),
    "| Leaves:", sum(nodes_df$is_leaf), "\n")
print(head(nodes_df))
print(head(edges_df))
```

### 3c. reprtree — representative tree

```r
# reprtree finds the single tree in the forest most similar
# to the ensemble — the best "summary tree" of the whole forest

install.packages("reprtree")
library(reprtree)

set.seed(42)
repr <- ReprTree(
  rf        = rf_model,
  train     = train,
  metric    = "d2"      # "d2" (node distance) or "corr" (correlation)
)

# ── Plot the representative tree ──────────────────────────────────────
plot(repr,
     depth   = 4,       # limit display depth
     main    = "Representative Tree from Random Forest")

# ── Extract the representative tree index ────────────────────────────
repr_tree_idx <- attr(repr, "representative")
cat("Representative tree index:", repr_tree_idx, "\n")

# ── Convert to rpart object for rpart.plot ────────────────────────────
repr_rpart <- as.rpart(repr)
rpart.plot(
  repr_rpart,
  type          = 4,
  extra         = 106,
  fallen.leaves = TRUE,
  main          = "Representative Tree (rpart.plot style)",
  box.palette   = list(Active = "#E1F5EE", Churn = "#FCEBEB")
)
```

---

## 4. Part C — ggplot2 Tree Plots from randomForest

### 4a. Build layout with ggraph + igraph

```r
install.packages(c("igraph", "ggraph", "tidygraph"))
library(igraph)
library(ggraph)
library(tidygraph)
library(ggplot2)

# ── Step 1: Parse a single tree ───────────────────────────────────────
tree_data <- parse_rf_tree(rf_model, k = 1)   # from Section 3b
nodes_df  <- tree_data$nodes
edges_df  <- tree_data$edges

# ── Step 2: Build igraph object ───────────────────────────────────────
g <- graph_from_data_frame(
  d        = edges_df[, c("from", "to")],
  directed = TRUE,
  vertices = nodes_df
)

# ── Step 3: Convert to tidygraph ──────────────────────────────────────
tg <- as_tbl_graph(g)

# ── Step 4: Plot with ggraph ──────────────────────────────────────────
ggraph(tg, layout = "tree") +
  # Edges
  geom_edge_diagonal(
    aes(label = label),
    colour      = "gray50",
    width       = 0.5,
    arrow       = arrow(length = unit(2, "mm"), type = "closed"),
    end_cap     = circle(4, "mm"),
    label_size  = 2.5,
    label_colour = "gray30",
    angle_calc  = "along",
    label_dodge = unit(2.5, "mm")
  ) +
  # Internal nodes
  geom_node_point(
    aes(filter = !is_leaf),
    size  = 8,
    shape = 22,
    fill  = "#E6F1FB",
    color = "#185FA5",
    stroke = 0.8
  ) +
  # Leaf nodes
  geom_node_point(
    aes(filter = is_leaf,
        color  = prediction),
    size  = 8,
    shape = 21,
    stroke = 0.8
  ) +
  scale_color_manual(
    values = c("Active" = "#1D9E75", "Churn" = "#A32D2D"),
    name   = "Prediction"
  ) +
  # Node labels
  geom_node_text(
    aes(label = label),
    size    = 2.2,
    repel   = FALSE,
    fontface = "plain",
    color   = "gray15"
  ) +
  theme_graph(base_family = "sans") +
  theme(
    legend.position = "bottom",
    plot.title      = element_text(size = 13, face = "bold",
                                   color = "#0F6E56"),
    plot.subtitle   = element_text(size = 9, color = "gray50")
  ) +
  labs(
    title    = "Random Forest — Tree 1 Structure",
    subtitle = paste0("Nodes: ", nrow(nodes_df),
                      " | Leaves: ", sum(nodes_df$is_leaf))
  )
```

### 4b. Pure ggplot2 from scratch (full control)

```r
library(ggplot2)

# ── Step 1: Compute x/y positions for every node ─────────────────────
compute_tree_layout <- function(nodes_df, edges_df) {

  n       <- nrow(nodes_df)
  x_pos   <- numeric(n)
  y_pos   <- numeric(n)
  visited <- logical(n)

  # BFS to assign levels (y) and leaf ordering (x)
  level     <- integer(n)
  level[1]  <- 0
  queue     <- 1

  while (length(queue) > 0) {
    curr  <- queue[1]; queue <- queue[-1]
    children_rows <- edges_df[edges_df$from == curr, "to"]
    for (ch in children_rows) {
      level[ch] <- level[curr] + 1
      queue     <- c(queue, ch)
    }
  }

  # Assign x positions: leaves get sequential slots
  leaves     <- nodes_df$id[nodes_df$is_leaf]
  leaf_x     <- setNames(seq_along(leaves), leaves)

  # Internal nodes: centre above their children
  assign_x <- function(node) {
    children <- edges_df[edges_df$from == node, "to"]
    if (length(children) == 0) return(leaf_x[as.character(node)])
    child_x  <- sapply(children, assign_x)
    mean(child_x)
  }

  for (nd in nodes_df$id) {
    x_pos[nd] <- assign_x(nd)
    y_pos[nd] <- -level[nd]          # flip so root is at top
  }

  nodes_df$x <- x_pos
  nodes_df$y <- y_pos
  nodes_df
}

# ── Step 2: Compute layout ────────────────────────────────────────────
nodes_layout <- compute_tree_layout(nodes_df, edges_df)

# ── Step 3: Build edge coordinates ────────────────────────────────────
edge_coords <- merge(
  edges_df,
  nodes_layout[, c("id", "x", "y")],
  by.x = "from", by.y = "id"
) %>%
  dplyr::rename(x_from = x, y_from = y) %>%
  merge(nodes_layout[, c("id", "x", "y")],
        by.x = "to", by.y = "id") %>%
  dplyr::rename(x_to = x, y_to = y)

# ── Step 4: Plot ──────────────────────────────────────────────────────
ggplot() +
  # ── Edges ────────────────────────────────────────────────────────
  geom_segment(
    data = edge_coords,
    aes(x = x_from, y = y_from, xend = x_to, yend = y_to),
    color    = "gray60",
    linewidth = 0.5,
    arrow    = arrow(length = unit(0.15, "cm"), type = "closed")
  ) +
  # ── Edge labels ──────────────────────────────────────────────────
  geom_label(
    data = edge_coords,
    aes(x = (x_from + x_to) / 2,
        y = (y_from + y_to) / 2,
        label = label),
    size      = 2.5,
    fill      = "white",
    color     = "gray40",
    label.size = 0,
    label.padding = unit(0.1, "lines")
  ) +
  # ── Internal nodes (split boxes) ────────────────────────────────
  geom_tile(
    data  = nodes_layout[!nodes_layout$is_leaf, ],
    aes(x = x, y = y),
    width  = 0.85, height = 0.45,
    fill   = "#E6F1FB",
    color  = "#185FA5",
    linewidth = 0.4
  ) +
  # ── Leaf nodes (prediction boxes) ───────────────────────────────
  geom_tile(
    data  = nodes_layout[nodes_layout$is_leaf, ],
    aes(x = x, y = y,
        fill = prediction),
    width  = 0.85, height = 0.45,
    color  = "gray40",
    linewidth = 0.4
  ) +
  scale_fill_manual(
    values = c("Active" = "#E1F5EE", "Churn" = "#FCEBEB"),
    na.value = "#E6F1FB",
    name   = "Prediction"
  ) +
  # ── Node text labels ─────────────────────────────────────────────
  geom_text(
    data  = nodes_layout,
    aes(x = x, y = y, label = label),
    size  = 2.3,
    color = "gray15",
    lineheight = 0.9
  ) +
  # ── Styling ──────────────────────────────────────────────────────
  scale_y_continuous(expand = expansion(add = 0.5)) +
  scale_x_continuous(expand = expansion(add = 0.5)) +
  theme_void() +
  theme(
    legend.position  = "bottom",
    legend.title     = element_text(size = 9),
    plot.title       = element_text(size = 13, face = "bold",
                                    color = "#0F6E56", hjust = 0.5,
                                    margin = margin(b = 4)),
    plot.subtitle    = element_text(size = 9, color = "gray50",
                                    hjust = 0.5, margin = margin(b = 10)),
    plot.margin      = margin(15, 15, 15, 15)
  ) +
  labs(
    title    = "Random Forest — Tree 1 (ggplot2)",
    subtitle = paste0("Nodes: ", nrow(nodes_layout),
                      "  |  Leaves: ", sum(nodes_layout$is_leaf),
                      "  |  Max depth: ",
                      abs(min(nodes_layout$y)))
  )
```

### 4c. ggplot2 horizontal tree layout

```r
# ── Horizontal version: root on left, leaves on right ─────────────────
# Swap x and y in the layout

nodes_horiz        <- nodes_layout
nodes_horiz$x_orig <- nodes_layout$x
nodes_horiz$y_orig <- nodes_layout$y
nodes_horiz$x      <- -nodes_layout$y   # depth becomes x
nodes_horiz$y      <- nodes_layout$x    # leaf position becomes y

edge_horiz <- edge_coords %>%
  dplyr::mutate(
    x_from_h = -y_from, y_from_h = x_from,
    x_to_h   = -y_to,   y_to_h   = x_to
  )

ggplot() +
  geom_segment(
    data = edge_horiz,
    aes(x = x_from_h, y = y_from_h,
        xend = x_to_h, yend = y_to_h),
    color = "gray60", linewidth = 0.5,
    arrow = arrow(length = unit(0.12, "cm"), type = "closed")
  ) +
  geom_label(
    data = edge_horiz,
    aes(x = (x_from_h + x_to_h) / 2,
        y = (y_from_h + y_to_h) / 2,
        label = label),
    size = 2.3, fill = "white", color = "gray40",
    label.size = 0, label.padding = unit(0.08, "lines")
  ) +
  geom_tile(
    data  = nodes_horiz[!nodes_horiz$is_leaf, ],
    aes(x = x, y = y),
    width = 0.45, height = 0.7,
    fill = "#E6F1FB", color = "#185FA5", linewidth = 0.4
  ) +
  geom_tile(
    data  = nodes_horiz[nodes_horiz$is_leaf, ],
    aes(x = x, y = y, fill = prediction),
    width = 0.45, height = 0.7,
    color = "gray40", linewidth = 0.4
  ) +
  scale_fill_manual(
    values   = c("Active" = "#E1F5EE", "Churn" = "#FCEBEB"),
    na.value = "#E6F1FB", name = "Prediction"
  ) +
  geom_text(
    data  = nodes_horiz,
    aes(x = x, y = y, label = label),
    size = 2.2, color = "gray15", lineheight = 0.9
  ) +
  scale_x_continuous(
    breaks = unique(floor(nodes_horiz$x)),
    labels = paste("Depth", abs(unique(floor(nodes_horiz$x))))
  ) +
  theme_void() +
  theme(
    axis.text.x      = element_text(size = 8, color = "gray50"),
    legend.position  = "bottom",
    plot.title       = element_text(size = 13, face = "bold",
                                    color = "#0F6E56", hjust = 0.5),
    plot.margin      = margin(15, 15, 15, 15)
  ) +
  labs(
    title = "Random Forest — Horizontal Tree Layout (ggplot2)"
  )
```

### 4d. Feature importance bar chart (ggplot2)

```r
library(ggplot2)
library(dplyr)

# ── Extract both importance measures ─────────────────────────────────
imp_mda   <- importance(rf_model, type = 1)   # Mean Decrease Accuracy
imp_mdg   <- importance(rf_model, type = 2)   # Mean Decrease Gini

imp_df <- data.frame(
  feature  = rownames(imp_mda),
  MDA      = imp_mda[, 1],
  MDGini   = imp_mdg[, 1]
) %>%
  arrange(desc(MDA)) %>%
  mutate(feature = factor(feature, levels = rev(feature)))

# ── Plot 1: Side-by-side MDA + MDGini ────────────────────────────────
library(tidyr)
imp_long <- imp_df %>%
  pivot_longer(cols = c(MDA, MDGini),
               names_to = "metric", values_to = "value")

ggplot(imp_long, aes(x = feature, y = value, fill = metric)) +
  geom_col(position = "dodge", width = 0.7) +
  coord_flip() +
  scale_fill_manual(
    values = c("MDA" = "#1D9E75", "MDGini" = "#534AB7"),
    labels = c("MDA" = "Mean Decrease Accuracy",
               "MDGini" = "Mean Decrease Gini"),
    name   = "Importance metric"
  ) +
  labs(
    title    = "Random Forest Feature Importance",
    subtitle = paste0("ntree = ", rf_model$ntree,
                      " | mtry = ", rf_model$mtry),
    x = NULL, y = "Importance"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position  = "top",
    panel.grid.major.y = element_blank(),
    axis.text.y      = element_text(color = "gray20"),
    plot.title       = element_text(face = "bold", color = "#0F6E56"),
    plot.subtitle    = element_text(color = "gray50", size = 9)
  )

# ── Plot 2: Single MDA with value labels ─────────────────────────────
ggplot(head(imp_df, 15),
       aes(x = feature, y = MDA)) +
  geom_col(fill = "#1D9E75", width = 0.7) +
  geom_text(aes(label = round(MDA, 2)),
            hjust = -0.15, size = 3, color = "gray30") +
  coord_flip(ylim = c(0, max(imp_df$MDA) * 1.15)) +
  labs(
    title = "Top 15 Features — Mean Decrease Accuracy",
    x = NULL, y = "Mean Decrease in Accuracy"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.major.y = element_blank(),
    plot.title         = element_text(face = "bold", color = "#0F6E56")
  )

# ── Plot 3: Dot plot with uncertainty (sd from permutations) ─────────
# randomForest stores localImp when localImp=TRUE in the call
rf_localimp <- randomForest(
  Churn ~ tenure + MonthlyCharges + TotalCharges + Contract,
  data = train, ntree = 200, importance = TRUE, localImp = TRUE
)

local_sd <- apply(rf_localimp$localImportance, 1, sd)
local_mean <- rowMeans(rf_localimp$localImportance)

imp_err_df <- data.frame(
  feature = names(local_mean),
  mean    = local_mean,
  sd      = local_sd
) %>% arrange(desc(mean)) %>%
  mutate(feature = factor(feature, levels = rev(feature)))

ggplot(imp_err_df, aes(x = feature, y = mean)) +
  geom_point(color = "#1D9E75", size = 3) +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd),
                width = 0.25, color = "#1D9E75", alpha = 0.6) +
  coord_flip() +
  labs(
    title    = "Feature Importance with Variability",
    subtitle = "Mean ± SD across observations",
    x = NULL, y = "Local Importance"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.major.y = element_blank(),
    plot.title         = element_text(face = "bold", color = "#0F6E56")
  )
```

### 4e. OOB error curve (ggplot2)

```r
# ── Extract OOB error matrix ──────────────────────────────────────────
oob_df <- as.data.frame(rf_model$err.rate)
oob_df$ntree <- seq_len(nrow(oob_df))

# Reshape to long format
oob_long <- tidyr::pivot_longer(
  oob_df,
  cols      = c(OOB, Active, Churn),
  names_to  = "class",
  values_to = "error"
)

# ── Plot ──────────────────────────────────────────────────────────────
ggplot(oob_long, aes(x = ntree, y = error,
                     color = class, linewidth = class)) +
  geom_line(alpha = 0.9) +
  scale_color_manual(
    values = c("OOB"    = "gray30",
               "Active" = "#1D9E75",
               "Churn"  = "#A32D2D"),
    name   = "Error type"
  ) +
  scale_linewidth_manual(
    values = c("OOB" = 1.0, "Active" = 0.6, "Churn" = 0.6),
    guide  = "none"
  ) +
  # Mark optimal ntree
  geom_vline(
    xintercept = which.min(oob_df$OOB),
    linetype   = "dashed",
    color      = "gray50",
    linewidth  = 0.5
  ) +
  annotate("text",
    x      = which.min(oob_df$OOB) + 5,
    y      = max(oob_long$error),
    label  = paste0("Min OOB\n@ ntree=", which.min(oob_df$OOB)),
    size   = 3, hjust = 0, color = "gray40"
  ) +
  labs(
    title    = "Random Forest OOB Error vs Number of Trees",
    subtitle = paste0("Final OOB error: ",
                      round(oob_df$OOB[nrow(oob_df)] * 100, 2), "%"),
    x = "Number of Trees", y = "OOB Error Rate"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    plot.title      = element_text(face = "bold", color = "#0F6E56"),
    plot.subtitle   = element_text(color = "gray50", size = 9),
    panel.grid.minor = element_blank()
  )
```

---

## 5. Part D — sparklyr / Databricks Tree Extraction

```r
library(sparklyr)
sc <- spark_connect(method = "databricks")   # or "local"

# ── Train Spark RF ────────────────────────────────────────────────────
rf_spark <- ml_random_forest_classifier(
  x         = train_spark,
  formula   = label ~ tenure + MonthlyCharges + TotalCharges,
  num_trees = 100,
  max_depth = 5,
  seed      = 42
)

# ── Method 1: toDebugString — text representation ────────────────────
tree_str <- ml_stage(rf_spark) %>%
  invoke("trees") %>%
  .[[1]] %>%                        # first tree (0-indexed in Scala)
  invoke("toDebugString")
cat(tree_str)

# ── Method 2: Per-tree stats via Java invoke ──────────────────────────
trees_java  <- ml_stage(rf_spark) %>% invoke("trees")

tree_stats_spark <- do.call(rbind, lapply(seq_along(trees_java), function(i) {
  t <- trees_java[[i]]
  data.frame(
    tree     = i,
    depth    = invoke(t, "depth"),
    n_nodes  = invoke(t, "numNodes"),
    n_leaves = invoke(t, "numLeaves")     # available in Spark 3.2+
  )
}))

summary(tree_stats_spark)

# ── Method 3: Convert Spark tree to rpart for plotting ────────────────
# Best approach: train rpart on a collected sample, plot that

train_r_sample <- collect(sdf_sample(train_spark, fraction = 0.2, seed = 42))
train_r_sample$Churn <- factor(train_r_sample$label,
                                levels = c(0, 1),
                                labels = c("Active", "Churn"))

rpart_proxy <- rpart(
  Churn ~ tenure + MonthlyCharges + TotalCharges,
  data    = train_r_sample,
  method  = "class",
  control = rpart.control(maxdepth = 5, cp = 0.005)
)

# Plot the proxy tree (structure mirrors Spark RF trees)
rpart.plot(
  rpart_proxy,
  type          = 4,
  extra         = 106,
  fallen.leaves = TRUE,
  main          = "Proxy Decision Tree (representative of Spark RF)",
  box.palette   = list(Active = "#E1F5EE", Churn = "#FCEBEB")
)

# ── Method 4: Feature importance (Spark RF) ──────────────────────────
imp_spark <- ml_feature_importances(rf_spark, train_spark)

ggplot(head(imp_spark, 15),
       aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "#1D9E75", width = 0.7) +
  geom_text(aes(label = round(importance, 3)),
            hjust = -0.1, size = 3, color = "gray30") +
  coord_flip(ylim = c(0, max(imp_spark$importance) * 1.15)) +
  labs(
    title    = "Spark RF Feature Importance (Gini)",
    subtitle = paste0("num_trees = ", rf_spark$model$num_trees),
    x = NULL, y = "Importance"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.major.y = element_blank(),
    plot.title         = element_text(face = "bold", color = "#0F6E56")
  )

# ── Method 5: rawPrediction vote distribution ─────────────────────────
vote_dist <- ml_predict(rf_spark, test_spark) %>%
  mutate(
    prob_churn  = vector_to_array(probability)[2],
    vote_churn  = vector_to_array(rawPrediction)[2],  # trees voting Churn
    vote_active = vector_to_array(rawPrediction)[1]   # trees voting Active
  ) %>%
  select(label, prob_churn, vote_churn, vote_active) %>%
  collect()

# Plot vote distribution
ggplot(vote_dist, aes(x = vote_churn,
                       fill = factor(label, labels = c("Active", "Churn")))) +
  geom_histogram(bins = 40, alpha = 0.75, position = "identity") +
  scale_fill_manual(values = c("Active" = "#1D9E75", "Churn" = "#A32D2D"),
                    name = "Actual") +
  labs(
    title    = "Distribution of Churn Votes per Observation",
    subtitle = paste0("Total trees: ", rf_spark$model$num_trees,
                      "  |  High votes = confident Churn prediction"),
    x = paste0("Number of trees voting Churn (of ",
                rf_spark$model$num_trees, " total)"),
    y = "Count"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    plot.title      = element_text(face = "bold", color = "#0F6E56"),
    plot.subtitle   = element_text(color = "gray50", size = 9)
  )
```

---

## 6. Save & Export Plots

```r
# ── Save as PNG (high resolution) ────────────────────────────────────
ggsave(
  filename = "output/tree_plot.png",
  plot     = last_plot(),     # or assign your plot to a variable
  width    = 14,
  height   = 9,
  dpi      = 300,
  bg       = "white"
)

# ── Save as PDF (scalable, best for reports) ──────────────────────────
ggsave(
  filename = "output/tree_plot.pdf",
  plot     = last_plot(),
  width    = 14,
  height   = 9,
  device   = "pdf"
)

# ── Save rpart.plot directly ──────────────────────────────────────────
png("output/rpart_tree.png", width = 2400, height = 1600, res = 200)
rpart.plot(
  tree_model,
  type = 4, extra = 106, fallen.leaves = TRUE,
  main = "Churn Decision Tree"
)
dev.off()

# ── Save to DBFS (Databricks) ─────────────────────────────────────────
ggsave(
  filename = "/dbfs/tmp/tree_plot.png",
  plot     = last_plot(),
  width    = 14, height = 9, dpi = 300, bg = "white"
)
# Then log to MLflow
mlflow_log_artifact("/dbfs/tmp/tree_plot.png", "plots")

# ── Display in Databricks notebook (inline) ───────────────────────────
# ggplot objects auto-display when printed in a notebook cell
print(last_plot())

# For base R plots in Databricks use display():
# display(recordPlot())
```

---

## 7. Quick-Reference Cheatsheet

### Method comparison

| Method | Package | Input | Best for | Output |
|---|---|---|---|---|
| `rpart.plot()` | `rpart.plot` | `rpart` object | Quick, publication-ready | Static plot |
| `plot()` + `text()` | base R | `rpart` object | Simple, no deps | Static plot |
| `ctree()` plot | `partykit` | `ctree` object | Unbiased splits + stats | Static plot |
| `grViz()` | `DiagrammeR` | DOT string | Interactive HTML | HTML widget |
| `fancyRpartPlot()` | `rattle` | `rpart` object | Colourful quick view | Static plot |
| `ReprTree()` | `reprtree` | `randomForest` | Summary of whole forest | rpart-like |
| `ggraph()` | `ggraph` + `igraph` | edge/node df | Customised ggplot style | ggplot2 |
| Pure `ggplot2` | `ggplot2` | edge/node df | Full layout control | ggplot2 |

### getTree() column reference

| Column | Type | Meaning |
|---|---|---|
| `left daughter` | int | Row index of left child; `-1` if leaf |
| `right daughter` | int | Row index of right child; `-1` if leaf |
| `split var` | char | Feature name used at this node |
| `split point` | dbl | Threshold: go left if `feature <= split point` |
| `status` | int | `1` = internal node, `-1` = terminal leaf |
| `prediction` | char | Predicted class at leaf (`NA` for internal) |

### rpart.plot type + extra combinations

| Goal | `type` | `extra` |
|---|---|---|
| Clean publication figure | `4` | `0` |
| Show sample counts | `4` | `1` |
| Show class % + n | `4` | `106` |
| Show all probabilities | `4` | `108` |
| Very compact tree | `2` | `104` |

### Colour palettes for churn trees

```r
# Two-class palette (Active / Churn)
box.palette = list(Active = "#E1F5EE", Churn = "#FCEBEB")   # soft
box.palette = list(Active = "#1D9E75", Churn = "#A32D2D")   # vivid
box.palette = "RdYlGn"                                        # built-in

# Node fill in ggplot2
scale_fill_manual(values = c("Active" = "#E1F5EE", "Churn" = "#FCEBEB"))

# Edge colour by depth
scale_edge_color_gradientn(colours = c("#1D9E75", "#185FA5", "#A32D2D"))
```

### Common issues & fixes

| Problem | Cause | Fix |
|---|---|---|
| Tree too wide to read | `maxdepth` too large | Set `maxdepth = 4` in `rpart.control()` |
| `getTree()` returns numbers not names | `labelVar = FALSE` | Use `labelVar = TRUE` |
| `ggraph` layout looks wrong | Root at bottom | Use `layout = "tree"` or flip with `scale_y_reverse()` |
| `reprtree` not on CRAN | Package archived | `remotes::install_github("araastat/reprtree")` |
| Spark `toDebugString` hard to parse | Text only | Use `rpart` proxy on collected sample |
| Plot text overlaps on large trees | Too many nodes | Limit depth or use `tweak` argument in `rpart.plot` |
| `geom_segment` arrows wrong direction | x/y assignment | Verify `from = parent, to = child` in edge_coords |

---

*R Tree Plot Extraction · rpart · randomForest · ggplot2 · ggraph · sparklyr · Databricks*
